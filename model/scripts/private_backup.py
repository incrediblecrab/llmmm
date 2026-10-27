"""Encrypted private backup of the corpus files that git does not hold.

`raw-data/`, `model/data/` and the recovery bundle exist only on local disk. This packs each top-level folder
into a zstd tar, encrypts it with gpg (AES-256, passphrase `LLMMM_BACKUP_PASSPHRASE` from the login keychain
via `keychain-env`), splits it into 2 GB parts and uploads the parts to a private Hugging Face dataset. The Hub
only ever holds ciphertext; the passphrase never touches disk or argv.

    pack           write .artifacts/private-backup/ (encrypted parts, local manifest)
    verify         decrypt every local part and compare each file to the manifest
    upload         create the repo private, confirm it is private, upload the parts
    check-remote   compare the Hub's SHA-256 of every part to the local one
    restore DIR    download, decrypt and extract into DIR without overwriting, then check every file

Run `upload`, `check-remote` and `restore` under `keychain-env run --` so `HF_TOKEN` is set.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / ".artifacts" / "private-backup"
UPLOAD = OUT / "upload"
REPO = "incrediblecrab/llmmm-private-backup"
KEY = "LLMMM_BACKUP_PASSPHRASE"
SOURCES = ("raw-data", "model/data")
BUNDLE = ".artifacts/workspace-v2-20260909.tar.gz"
PART_BYTES = 2_000_000_000
CHUNK = 1 << 20
ENV = {**os.environ, "COPYFILE_DISABLE": "1"}

CARD = f"""---
license: other
viewer: false
---
# llmmm private backup

Encrypted backup of the llmmm training corpus (`raw-data/`, `model/data/` and the recovery bundle). Every file
is gpg-encrypted ciphertext; the passphrase is not stored here. Restore with
`keychain-env run -- model/scripts/private_backup.py restore DIR` from https://github.com/incrediblecrab/llmmm.
This repository must stay private: the source corpora are third-party data that the project does not redistribute.
"""


def units() -> list[tuple[str, list[str]]]:
    """One archive per top-level folder; loose files in a source folder share one archive."""
    found = []
    for source in SOURCES:
        children = sorted((ROOT / source).iterdir())
        found += [(f"{source}/{c.name}", [f"{source}/{c.name}"]) for c in children if c.is_dir() and not c.is_symlink()]
        loose = [f"{source}/{c.name}" for c in children if not c.is_dir() or c.is_symlink()]
        if loose:
            found.append((f"{source}/_files", loose))
    found.append((BUNDLE, [BUNDLE]))
    return found


def archive_name(unit: str) -> str:
    return unit.lstrip(".").replace("/", "__") + ".tar.zst.gpg"


def passphrase() -> bytes:
    return subprocess.run(["keychain-env", "get", KEY], capture_output=True, check=True).stdout.strip()


def gpg(mode: list[str], stdin, stdout) -> subprocess.Popen:
    """gpg reading the passphrase from an inherited pipe, so it is never in argv or a file."""
    read_fd, write_fd = os.pipe()
    os.write(write_fd, passphrase() + b"\n")
    os.close(write_fd)
    proc = subprocess.Popen(
        ["gpg", "--batch", "--quiet", "--pinentry-mode", "loopback", "--no-symkey-cache",
         "--passphrase-fd", str(read_fd), *mode],
        stdin=stdin, stdout=stdout, pass_fds=(read_fd,))
    os.close(read_fd)
    return proc


def encrypt_args() -> list[str]:
    return ["--symmetric", "--cipher-algo", "AES256", "--compress-algo", "none", "--output", "-"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_of(base: Path, paths: list[str]) -> dict[str, str]:
    """SHA-256 of every regular file, `symlink:<target>` for links; directories are implied."""
    entries = {}
    for rel in paths:
        top = base / rel
        walk = [(top.parent, [], [top.name])] if not top.is_dir() or top.is_symlink() else os.walk(top)
        for folder, dirs, files in walk:
            for name in list(files) + [d for d in dirs if (Path(folder) / d).is_symlink()]:
                path = Path(folder) / name
                key = str(path.relative_to(base))
                entries[key] = f"symlink:{os.readlink(path)}" if path.is_symlink() else sha256(path)
    return entries


def read_tsv(path: Path) -> dict[str, str]:
    return dict(reversed(line.split("\t", 1)) for line in path.read_text().splitlines())


def write_tsv(path: Path, entries: dict[str, str]) -> None:
    path.write_text("".join(f"{value}\t{key}\n" for key, value in sorted(entries.items())))


def parts_of(folder: Path, unit: str) -> list[Path]:
    return sorted(folder.glob(archive_name(unit) + ".*"))


def pack(_args) -> None:
    UPLOAD.mkdir(parents=True, exist_ok=True)
    if any(UPLOAD.glob("*.gpg*")):
        sys.exit(f"{UPLOAD} already holds parts; move them aside before packing again")
    manifest, part_hashes = {}, {}
    for unit, paths in units():
        manifest.update(manifest_of(ROOT, paths))
        tar = subprocess.Popen(["tar", "--no-mac-metadata", "-cf", "-", "-C", str(ROOT), *paths], stdout=subprocess.PIPE, env=ENV)
        zstd = subprocess.Popen(["zstd", "-q", "-T0", "-10", "-c"], stdin=tar.stdout, stdout=subprocess.PIPE)
        tar.stdout.close()
        enc = gpg(encrypt_args(), zstd.stdout, subprocess.PIPE)
        zstd.stdout.close()
        index, block = 0, enc.stdout.read(CHUNK)
        while block:
            part = UPLOAD / f"{archive_name(unit)}.{index:03d}"
            digest, written = hashlib.sha256(), 0
            with part.open("wb") as handle:
                while block and written < PART_BYTES:
                    handle.write(block)
                    digest.update(block)
                    written += len(block)
                    block = enc.stdout.read(min(CHUNK, PART_BYTES - written) or CHUNK)
            part_hashes[part.name] = digest.hexdigest()
            index += 1
        for proc in (tar, zstd, enc):
            if proc.wait():
                sys.exit(f"{proc.args[0]} failed for {unit}")
        print(f"packed {unit}: {index} part(s)", flush=True)
    write_tsv(OUT / "manifest.tsv", manifest)
    with (OUT / "manifest.tsv").open("rb") as plain, (UPLOAD / "manifest.tsv.gpg").open("wb") as sealed:
        if gpg(encrypt_args(), plain, sealed).wait():
            sys.exit("gpg failed for the manifest")
    write_tsv(OUT / "parts.tsv", part_hashes)
    (UPLOAD / "README.md").write_text(CARD)
    print(f"{len(manifest)} files, {len(part_hashes)} parts in {UPLOAD}")


def decrypted_stream(parts: list[Path]) -> tuple[subprocess.Popen, list[subprocess.Popen]]:
    cat = subprocess.Popen(["cat", *map(str, parts)], stdout=subprocess.PIPE)
    dec = gpg(["--decrypt"], cat.stdout, subprocess.PIPE)
    cat.stdout.close()
    zstd = subprocess.Popen(["zstd", "-q", "-d", "-c"], stdin=dec.stdout, stdout=subprocess.PIPE)
    dec.stdout.close()
    return zstd, [cat, dec, zstd]


def verify(_args) -> None:
    expected, seen = read_tsv(OUT / "manifest.tsv"), {}
    for unit, _paths in units():
        last, procs = decrypted_stream(parts_of(UPLOAD, unit))
        with tarfile.open(fileobj=last.stdout, mode="r|") as archive:
            for member in archive:
                if member.issym():
                    seen[member.name] = f"symlink:{member.linkname}"
                elif member.isfile():
                    digest = hashlib.sha256()
                    handle = archive.extractfile(member)
                    for block in iter(lambda: handle.read(CHUNK), b""):
                        digest.update(block)
                    seen[member.name] = digest.hexdigest()
        last.stdout.read()
        if any(proc.wait() for proc in procs):
            sys.exit(f"decrypting {unit} failed")
        print(f"verified {unit}", flush=True)
    for part, digest in read_tsv(OUT / "parts.tsv").items():
        if sha256(UPLOAD / part) != digest:
            sys.exit(f"part changed on disk: {part}")
    missing, extra = expected.keys() - seen.keys(), seen.keys() - expected.keys()
    wrong = [key for key in expected.keys() & seen.keys() if expected[key] != seen[key]]
    if missing or extra or wrong:
        sys.exit(f"mismatch: {len(missing)} missing, {len(extra)} unexpected, {len(wrong)} differ")
    print(f"all {len(expected)} files decrypt to their recorded SHA-256")


def upload(_args) -> None:
    api = HfApi()
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(REPO, repo_type="dataset").private:
        sys.exit(f"refusing to upload: {REPO} is not private")
    api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=UPLOAD, allow_patterns=["*.gpg*", "README.md"],
                      commit_message="Encrypted corpus backup")
    print(f"uploaded to https://huggingface.co/datasets/{REPO}")


def check_remote(_args) -> None:
    api = HfApi()
    if not api.repo_info(REPO, repo_type="dataset").private:
        sys.exit(f"{REPO} is not private")
    remote = {}
    for entry in api.list_repo_tree(REPO, repo_type="dataset", recursive=True, expand=True):
        if ".gpg" not in entry.path or not hasattr(entry, "size"):
            continue
        if getattr(entry, "lfs", None):
            remote[entry.path] = entry.lfs.sha256
        else:  # small files may be stored as plain git blobs, which carry no SHA-256
            remote[entry.path] = sha256(Path(hf_hub_download(REPO, entry.path, repo_type="dataset")))
    local = read_tsv(OUT / "parts.tsv")
    local["manifest.tsv.gpg"] = sha256(UPLOAD / "manifest.tsv.gpg")
    bad = sorted(name for name in local.keys() | remote.keys() if local.get(name) != remote.get(name))
    if bad:
        sys.exit(f"{len(bad)} part(s) differ or are missing on the Hub: {bad[:5]}")
    print(f"all {len(local)} encrypted files on the Hub match the local SHA-256")


def restore(args) -> None:
    target = Path(args.directory).resolve()
    chosen = [(unit, paths) for unit, paths in units() if not args.unit or unit in args.unit]
    patterns = ["manifest.tsv.gpg", *(archive_name(unit) + ".*" for unit, _ in chosen)]
    with tempfile.TemporaryDirectory(dir=target.parent) as staging:
        download = Path(snapshot_download(REPO, repo_type="dataset", local_dir=staging, allow_patterns=patterns))
        with (download / "manifest.tsv.gpg").open("rb") as sealed:
            manifest_proc = gpg(["--decrypt"], sealed, subprocess.PIPE)
            text = manifest_proc.stdout.read().decode()
        if manifest_proc.wait():
            sys.exit("could not decrypt the manifest: wrong passphrase?")
        expected = dict(reversed(line.split("\t", 1)) for line in text.splitlines())
        target.mkdir(parents=True, exist_ok=True)
        for unit, paths in chosen:
            last, procs = decrypted_stream(parts_of(download, unit))
            procs.append(subprocess.Popen(["tar", "-xk", "-f", "-", "-C", str(target)], stdin=last.stdout))
            last.stdout.close()
            if any(proc.wait() for proc in procs):
                sys.exit(f"restoring {unit} failed")
            want = {key: value for key, value in expected.items() if any(key == p or key.startswith(p + "/") for p in paths)}
            got = manifest_of(target, paths)
            if got != want:
                sys.exit(f"{unit}: restored files do not match the manifest")
            print(f"restored {unit}: {len(want)} files match", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("pack", pack), ("verify", verify), ("upload", upload), ("check-remote", check_remote)):
        commands.add_parser(name).set_defaults(handler=handler)
    restoring = commands.add_parser("restore")
    restoring.add_argument("directory")
    restoring.add_argument("--unit", action="append", help="restore only this unit, e.g. raw-data/01-recipenlg")
    restoring.set_defaults(handler=restore)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
