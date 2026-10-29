"""Private backup of the files that git does not hold.

`raw-data/`, `model/data/`, `.artifacts/` (with the recovery bundle) and the git-ignored files in
`model/results/runs/`, `model/training/` and `prior-study/data/` exist only on local disk. This packs each
top-level folder into a zstd tar, splits it into 2 GB parts and uploads the parts to a private Hugging Face
dataset, which must never be made public.

    pack           write .artifacts/private-backup/ (parts, local manifest); a folder whose parts already exist
                   is kept if its files and parts still match the manifest, so only new folders upload
    verify         decompress every local part and compare each file to the manifest
    upload         create the repo private, confirm it is private, upload the parts, delete Hub parts no longer listed
    check-remote   compare the Hub's SHA-256 of every part to the local one
    restore DIR    download, decompress and extract into DIR without overwriting, then check every file

`restore` takes the list of folders from the manifest on the Hub, so it works with the local copies deleted.
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
SOURCES = ("raw-data", "model/data", ".artifacts")  # one archive per child folder
FOLDERS = ("model/results/runs", "model/training", "prior-study/data")  # one archive of the git-ignored files
BUNDLE = ".artifacts/workspace-v2-20260909.tar.gz"
MANIFEST = "manifest.tsv"
PART = ".[0-9][0-9][0-9]"  # part suffix; also excludes the retired `.gpg.NNN` parts
PART_BYTES = 2_000_000_000
CHUNK = 1 << 20
ENV = {**os.environ, "COPYFILE_DISABLE": "1"}

CARD = f"""---
license: other
viewer: false
---
# llmmm private backup

Backup of the llmmm training corpus and the local files git does not hold (`raw-data/`, `model/data/`, `.artifacts/` with the recovery bundle, and the git-ignored files in `model/results/runs/`, `model/training/` and `prior-study/data/`), as zstd tar parts with a SHA-256 `{MANIFEST}` of every file. Restore with `keychain-env run -- model/scripts/private_backup.py restore DIR` from https://github.com/incrediblecrab/llmmm.
This repository must stay private: the source corpora are third-party data that the project does not redistribute.
"""


def units() -> list[tuple[str, list[str]]]:
    """One archive per top-level folder; loose files in a source folder share one archive."""
    found = []
    for source in SOURCES:
        children = [c for c in sorted((ROOT / source).iterdir()) if c != OUT and c != ROOT / BUNDLE]
        if (ROOT / source / "_files").exists():
            sys.exit(f"{source}/_files would collide with the archive of loose files; rename it")
        found += [(f"{source}/{c.name}", [f"{source}/{c.name}"]) for c in children if c.is_dir() and not c.is_symlink()]
        loose = [f"{source}/{c.name}" for c in children if not c.is_dir() or c.is_symlink()]
        if loose:
            found.append((f"{source}/_files", loose))
    for folder in FOLDERS:
        ignored = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--others", "--ignored", "--exclude-standard", "--", folder],
                                 capture_output=True, check=True).stdout.decode().split("\0")
        if any(ignored):
            found.append((folder, sorted(filter(None, ignored))))
    found.append((BUNDLE, [BUNDLE]))
    return found


def unit_of(key: str) -> str:
    """The archive a manifest entry belongs to, so the Hub copy can be read without the local folders."""
    if key == BUNDLE:
        return BUNDLE
    for folder in FOLDERS:
        if key == folder or key.startswith(folder + "/"):
            return folder
    for source in SOURCES:
        if key.startswith(source + "/"):
            head, nested, _ = key[len(source) + 1:].partition("/")
            return f"{source}/{head}" if nested else f"{source}/_files"
    sys.exit(f"manifest entry outside every backed-up folder: {key}")


def by_unit(manifest: dict[str, str]) -> dict[str, dict[str, str]]:
    grouped: dict[str, dict[str, str]] = {}
    for key, value in manifest.items():
        grouped.setdefault(unit_of(key), {})[key] = value
    return grouped


def archive_name(unit: str) -> str:
    return unit.lstrip(".").replace("/", "__") + ".tar.zst"


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


def parse_tsv(text: str) -> dict[str, str]:
    return dict(reversed(line.split("\t", 1)) for line in text.splitlines())


def read_tsv(path: Path) -> dict[str, str]:
    return parse_tsv(path.read_text())


def write_tsv(path: Path, entries: dict[str, str]) -> None:
    path.write_text("".join(f"{value}\t{key}\n" for key, value in sorted(entries.items())))


def parts_of(folder: Path, unit: str) -> list[Path]:
    return sorted(folder.glob(archive_name(unit) + PART))


def pack(_args) -> None:
    UPLOAD.mkdir(parents=True, exist_ok=True)
    recorded = by_unit(read_tsv(UPLOAD / MANIFEST)) if (UPLOAD / MANIFEST).exists() else {}
    part_hashes = read_tsv(OUT / "parts.tsv") if (OUT / "parts.tsv").exists() else {}
    local = units()
    gone = recorded.keys() - {unit for unit, _ in local}
    if gone:
        sys.exit(f"{len(gone)} backed-up folder(s) are not on local disk, e.g. {sorted(gone)[0]}: restore them before packing")
    manifest = {}
    for unit, paths in local:
        files = manifest_of(ROOT, paths)
        manifest.update(files)
        if parts := parts_of(UPLOAD, unit):
            if files != recorded.get(unit, {}):
                sys.exit(f"{unit} changed since it was packed; move {OUT} aside and pack everything again")
            listed = {name for name in part_hashes if name.startswith(archive_name(unit) + ".")}
            if listed != {part.name for part in parts} or any(part_hashes[part.name] != sha256(part) for part in parts):
                sys.exit(f"{unit}: its parts on disk do not match parts.tsv; move {OUT} aside and pack everything again")
            print(f"kept {unit}: already packed and unchanged", flush=True)
            continue
        part_hashes = {name: digest for name, digest in part_hashes.items() if not name.startswith(archive_name(unit) + ".")}
        tar = subprocess.Popen(["tar", "--no-mac-metadata", "-cf", "-", "-C", str(ROOT), *paths], stdout=subprocess.PIPE, env=ENV)
        zstd = subprocess.Popen(["zstd", "-q", "-T0", "-10", "-c"], stdin=tar.stdout, stdout=subprocess.PIPE)
        tar.stdout.close()
        index, block = 0, zstd.stdout.read(CHUNK)
        while block:
            part = UPLOAD / f"{archive_name(unit)}.{index:03d}"
            digest, written = hashlib.sha256(), 0
            with part.open("wb") as handle:
                while block and written < PART_BYTES:
                    handle.write(block)
                    digest.update(block)
                    written += len(block)
                    block = zstd.stdout.read(min(CHUNK, PART_BYTES - written) or CHUNK)
            part_hashes[part.name] = digest.hexdigest()
            index += 1
        for proc in (tar, zstd):
            if proc.wait():
                sys.exit(f"{proc.args[0]} failed for {unit}")
        print(f"packed {unit}: {index} part(s)", flush=True)
    write_tsv(OUT / "parts.tsv", part_hashes)
    write_tsv(UPLOAD / MANIFEST, manifest)
    (UPLOAD / "README.md").write_text(CARD)
    print(f"{len(manifest)} files, {len(part_hashes)} parts in {UPLOAD}")


def decompressed_stream(parts: list[Path]) -> tuple[subprocess.Popen, list[subprocess.Popen]]:
    cat = subprocess.Popen(["cat", *map(str, parts)], stdout=subprocess.PIPE)
    zstd = subprocess.Popen(["zstd", "-q", "-d", "-c"], stdin=cat.stdout, stdout=subprocess.PIPE)
    cat.stdout.close()
    return zstd, [cat, zstd]


def verify(_args) -> None:
    expected, seen = read_tsv(UPLOAD / MANIFEST), {}
    for unit in sorted(by_unit(expected)):
        last, procs = decompressed_stream(parts_of(UPLOAD, unit))
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
            sys.exit(f"decompressing {unit} failed")
        print(f"verified {unit}", flush=True)
    for part, digest in read_tsv(OUT / "parts.tsv").items():
        if sha256(UPLOAD / part) != digest:
            sys.exit(f"part changed on disk: {part}")
    missing, extra = expected.keys() - seen.keys(), seen.keys() - expected.keys()
    wrong = [key for key in expected.keys() & seen.keys() if expected[key] != seen[key]]
    if missing or extra or wrong:
        sys.exit(f"mismatch: {len(missing)} missing, {len(extra)} unexpected, {len(wrong)} differ")
    print(f"all {len(expected)} files decompress to their recorded SHA-256")


def upload(_args) -> None:
    api = HfApi()
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(REPO, repo_type="dataset").private:
        sys.exit(f"refusing to upload: {REPO} is not private")
    parts = read_tsv(OUT / "parts.tsv")
    if missing := [name for name in parts if not (UPLOAD / name).exists()]:
        sys.exit(f"{len(missing)} part(s) in parts.tsv are not on disk, e.g. {missing[0]}: pack again before uploading")
    keep = {*parts, MANIFEST, "README.md", ".gitattributes"}
    stale = [path for path in api.list_repo_files(REPO, repo_type="dataset") if path not in keep]
    api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=UPLOAD, allow_patterns=[*parts, MANIFEST, "README.md"],
                      delete_patterns=stale or None, commit_message="Corpus backup")
    print(f"uploaded to https://huggingface.co/datasets/{REPO}" + (f"; deleted {len(stale)} file(s) no longer listed" if stale else ""))


def check_remote(_args) -> None:
    api = HfApi()
    if not api.repo_info(REPO, repo_type="dataset").private:
        sys.exit(f"{REPO} is not private")
    local = read_tsv(OUT / "parts.tsv")
    local[MANIFEST] = sha256(UPLOAD / MANIFEST)
    remote = {}
    for entry in api.list_repo_tree(REPO, repo_type="dataset", recursive=True, expand=True):
        if entry.path in ("README.md", ".gitattributes") or not hasattr(entry, "size"):
            continue
        if getattr(entry, "lfs", None):
            remote[entry.path] = entry.lfs.sha256
        else:  # small files may be stored as plain git blobs, which carry no SHA-256
            remote[entry.path] = sha256(Path(hf_hub_download(REPO, entry.path, repo_type="dataset")))
    bad = sorted(name for name in local.keys() | remote.keys() if local.get(name) != remote.get(name))
    if bad:
        sys.exit(f"{len(bad)} file(s) differ, are missing or are extra on the Hub: {bad[:5]}")
    print(f"all {len(local)} files on the Hub match the local SHA-256")


def fetch(staging: str, patterns: list[str], attempts: int = 4) -> Path:
    """snapshot_download, retried: a multi-GB download can drop mid-stream, and a retry resumes it."""
    for attempt in range(1, attempts + 1):
        try:
            return Path(snapshot_download(REPO, repo_type="dataset", local_dir=staging, allow_patterns=patterns))
        except Exception as error:
            if attempt == attempts:
                raise
            print(f"download failed ({error}); retrying {attempt}/{attempts - 1}", flush=True)


def restore(args) -> None:
    target = Path(args.directory).resolve()
    with tempfile.TemporaryDirectory(dir=target.parent) as staging:
        download = fetch(staging, [MANIFEST])
        recorded = by_unit(parse_tsv((download / MANIFEST).read_text()))
        unknown = set(args.unit or ()) - recorded.keys()
        if unknown:
            sys.exit(f"not in the backup: {sorted(unknown)}; units are named like raw-data/01-recipenlg")
        chosen = [unit for unit in sorted(recorded) if not args.unit or unit in args.unit]
        fetch(staging, [archive_name(unit) + PART for unit in chosen])
        target.mkdir(parents=True, exist_ok=True)
        for unit in chosen:
            parts = parts_of(download, unit)
            if not parts:
                sys.exit(f"{unit}: no parts on the Hub")
            last, procs = decompressed_stream(parts)
            procs.append(subprocess.Popen(["tar", "-xk", "-f", "-", "-C", str(target)], stdin=last.stdout))
            last.stdout.close()
            if any(proc.wait() for proc in procs):
                sys.exit(f"restoring {unit} failed")
            want = recorded[unit]
            got = manifest_of(target, list(want))
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
