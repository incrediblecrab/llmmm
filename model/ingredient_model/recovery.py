"""Local, standard-library-only snapshots of explicitly declared model assets.

    python -m ingredient_model.recovery [--root ROOT] pack --output snapshot.tar.gz
    python -m ingredient_model.recovery [--root ROOT] restore --bundle snapshot.tar.gz
    python -m ingredient_model.recovery [--root ROOT] verify

``ROOT`` defaults to the checkout containing this file, never the current
directory or IM_DATA/IM_RESULTS. Direct execution of this file also works
without importing the rest of ingredient_model.

Keep workspace.json and the trusted artifacts.lock.json in the checkout; neither
belongs in the archive. The lock binds the exact workspace bytes, archive, run
inventory, and individual files. It is an integrity record, not a signature.
Bundles are local, owner-readable files, not encrypted or uploaded anywhere.

Every directory immediately below a configured run group must contain
manifest.json, metrics.json, and embedding.npy. Additional arrays are collected
only from that run's immediate *.npy files. Incomplete groups/runs are errors.
Inputs should be quiescent while packing. Existing different archives and
destination files are never overwritten: use a new archive name or a clean
checkout. A failed pack cannot invalidate an earlier archive by replacing it.

Restore validates the entire archive and all existing destinations before
publishing files. Individual publications are atomic and do not clobber files;
an interruption during publication can leave a subset of already validated
files, so rerunning restore is safe. Temporary directories are scoped to this
operation and cleaned on ordinary failures. Safe publication requires POSIX
no-follow directory descriptors and a filesystem supporting hard links.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import errno
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tarfile
import tempfile
from typing import BinaryIO, Iterator, Sequence
import zlib


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = "model/workspace.json"
MANIFEST = "model/artifacts.lock.json"
RUNS = "model/results/runs"
REQUIRED_RUN_FILES = ("embedding.npy", "manifest.json", "metrics.json")
CHUNK_SIZE = 1024 * 1024
MAX_JSON_SIZE = 16 * CHUNK_SIZE
DATA_SUFFIXES = {".csv", ".json", ".npy", ".npz", ".parquet"}
SECRET_NAME = re.compile(
    r"(?:^|[._-])(?:secrets?|credentials?|passwords?|tokens?|"
    r"(?:api|private)[_-]?keys?|(?:access|refresh)[_-]?tokens?|"
    r"id_rsa|id_ed25519)(?:[._-]|$)", re.IGNORECASE)


class RecoveryError(ValueError):
    """Invalid, incomplete, unsafe, or conflicting recovery inputs."""


@dataclass(frozen=True)
class Snapshot:
    root: Path
    workspace_size: int
    workspace_sha256: str
    data_files: tuple[str, ...]
    extra_files: tuple[str, ...]
    run_groups: tuple[str, ...]
    default_run: str


@dataclass(frozen=True)
class LockedFile:
    path: str
    size: int
    sha256: str
    group: str | None


@dataclass(frozen=True)
class Verification:
    total: int
    missing: tuple[str, ...]
    mismatched: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing and not self.mismatched


@dataclass(frozen=True)
class Restoration:
    restored: tuple[str, ...]
    skipped: tuple[str, ...]


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(Path(path).expanduser()))


def _relative(value: object, label: str) -> str:
    if (not isinstance(value, str) or not value
            or re.fullmatch(r"[A-Za-z0-9_./-]+", value) is None
            or any(part in ("", ".", "..") or part.endswith(".")
                   or re.match(r"(?i)^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)", part)
                   for part in value.split("/"))):
        raise RecoveryError(f"{label}: expected a canonical relative POSIX path, got {value!r}")
    return value


def _name(value: object, label: str) -> str:
    name = _relative(value, label)
    if "/" in name:
        raise RecoveryError(f"{label}: expected a single path component, got {name!r}")
    return name


def _asset_path(value: object, label: str) -> str:
    path = _relative(value, label)
    for part in path.split("/"):
        if (part.startswith(".") or part.lower() in ("raw-data", "cloud", "venv")
                or SECRET_NAME.search(part)):
            raise RecoveryError(f"{label}: prohibited secret/raw-data path: {path}")
    return path


def _strings(value: object, label: str, *, nonempty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (nonempty and not value):
        raise RecoveryError(f"{label}: expected {'a nonempty' if nonempty else 'a'} list")
    paths = tuple(_asset_path(item, label) for item in value)
    if len(set(paths)) != len(paths):
        raise RecoveryError(f"{label}: duplicate paths")
    return tuple(sorted(paths))


def _check_collisions(paths: Sequence[str]) -> None:
    files = set(paths)
    prefixes: dict[str, str] = {}
    for path in paths:
        parts = path.split("/")
        for length in range(1, len(parts) + 1):
            prefix = "/".join(parts[:length])
            previous = prefixes.setdefault(prefix.casefold(), prefix)
            if previous != prefix:
                raise RecoveryError(f"ambiguous case-alias paths: {previous}, {prefix}")
            if length < len(parts) and prefix in files:
                raise RecoveryError(f"file/directory path collision: {prefix}, {path}")


def _inspect(path: Path, *, directory: bool = False,
             missing: bool = False) -> os.stat_result | None:
    """Check every component without resolving away a symlink."""
    current = Path(path.anchor)
    result = current.lstat()
    for index, part in enumerate(path.parts[1:], start=1):
        current /= part
        try:
            result = current.lstat()
        except FileNotFoundError as exc:
            if missing:
                return None
            raise RecoveryError(f"missing required path: {path}") from exc
        if stat.S_ISLNK(result.st_mode):
            raise RecoveryError(f"symlink paths are not allowed: {current}")
        needs_directory = index < len(path.parts) - 1 or directory
        if needs_directory and not stat.S_ISDIR(result.st_mode):
            raise RecoveryError(f"not a directory: {current}")
        if not needs_directory and not stat.S_ISREG(result.st_mode):
            raise RecoveryError(f"not a regular file: {current}")
    return result


def _stamp(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


@contextmanager
def _directory_fd(path: Path, *, create: bool = False) -> Iterator[int]:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise RecoveryError("recovery requires POSIX no-follow directory operations")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:]:
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, mode=0o700, dir_fd=descriptor)
                child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def _open_regular(path: Path) -> Iterator[BinaryIO]:
    expected = _inspect(path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with _directory_fd(path.parent) as parent, os.fdopen(
            os.open(path.name, flags, dir_fd=parent), "rb") as stream:
        actual = os.fstat(stream.fileno())
        if expected is None or not stat.S_ISREG(actual.st_mode) or _stamp(actual) != _stamp(expected):
            raise RecoveryError(f"file changed while opening: {path}")
        yield stream


class _HashingReader:
    def __init__(self, stream: BinaryIO):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size: int) -> bytes:
        chunk = self.stream.read(size)
        self.digest.update(chunk)
        self.size += len(chunk)
        return chunk


def _digest(stream: BinaryIO, destination: BinaryIO | None = None) -> tuple[int, str]:
    reader = _HashingReader(stream)
    for chunk in iter(lambda: reader.read(CHUNK_SIZE), b""):
        if destination is not None:
            destination.write(chunk)
    return reader.size, reader.digest.hexdigest()


def _fingerprint(path: Path) -> tuple[int, str]:
    with _open_regular(path) as stream:
        before = _stamp(os.fstat(stream.fileno()))
        result = _digest(stream)
        if _stamp(os.fstat(stream.fileno())) != before:
            raise RecoveryError(f"file changed while hashing: {path}")
        return result


def _json_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise RecoveryError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise RecoveryError(f"invalid JSON constant: {value}")


def _read_json(path: Path) -> tuple[dict, int, str]:
    with _open_regular(path) as stream:
        before = _stamp(os.fstat(stream.fileno()))
        if before[2] > MAX_JSON_SIZE:
            raise RecoveryError(f"JSON metadata exceeds {MAX_JSON_SIZE} bytes: {path}")
        payload = stream.read(MAX_JSON_SIZE + 1)
        if len(payload) > MAX_JSON_SIZE or _stamp(os.fstat(stream.fileno())) != before:
            raise RecoveryError(f"JSON metadata changed while reading: {path}")
    try:
        data = json.loads(payload, object_pairs_hook=_json_pairs,
                          parse_constant=_invalid_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise RecoveryError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RecoveryError(f"{path}: expected a JSON object")
    return data, len(payload), hashlib.sha256(payload).hexdigest()


def _version(data: dict, label: str) -> None:
    if type(data.get("version")) is not int or data["version"] != 1:
        raise RecoveryError(f"{label}: unsupported or missing version (expected integer 1)")


def _keys(data: object, expected: set[str], label: str) -> dict:
    if not isinstance(data, dict) or set(data) != expected:
        raise RecoveryError(f"{label}: expected fields {', '.join(sorted(expected))}")
    return data


def _size(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise RecoveryError(f"{label}: expected a nonnegative integer size")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RecoveryError(f"{label}: expected a lowercase SHA-256 digest")
    return value


def load_snapshot(root: Path | str | None = None) -> Snapshot:
    """Read the fixed workspace contract, independent of scientific packages."""
    root = _absolute(ROOT if root is None else root)
    _inspect(root, directory=True)
    data, size, digest = _read_json(root / WORKSPACE)
    _version(data, WORKSPACE)
    snapshot = _keys(data.get("snapshot"),
                     {"manifest", "data_files", "run_groups", "extra_files"},
                     "workspace snapshot")
    if snapshot["manifest"] != MANIFEST:
        raise RecoveryError(f"snapshot.manifest must be {MANIFEST!r}")
    data_files = _strings(snapshot["data_files"], "snapshot.data_files", nonempty=True)
    extra_files = _strings(snapshot["extra_files"], "snapshot.extra_files")
    groups = _strings(snapshot["run_groups"], "snapshot.run_groups", nonempty=True)
    for path in data_files:
        if (not path.startswith(("model/data/", "prior-study/data/"))
                or Path(path).suffix not in DATA_SUFFIXES):
            raise RecoveryError(f"data file is outside the supported data locations/formats: {path}")
    for path in extra_files:
        if (not path.startswith("model/results/") or path.startswith(f"{RUNS}/")
                or Path(path).suffix not in DATA_SUFFIXES):
            raise RecoveryError(f"extra file must be a non-run result artifact: {path}")
    for group in groups:
        _name(group, "run group")
    for key in ("benchmark_sweep", "training_sweep"):
        if _name(data.get(key), key) not in groups:
            raise RecoveryError(f"{key} must be included in snapshot.run_groups")
    default_run = _asset_path(data.get("default_embedding_run"), "default_embedding_run")
    if len(default_run.split("/")) != 2 or default_run.split("/")[0] not in groups:
        raise RecoveryError("default_embedding_run must name a run in a configured group")
    _check_collisions((*data_files, *extra_files, *(f"{RUNS}/{g}" for g in groups)))
    return Snapshot(root, size, digest, data_files, extra_files, groups, default_run)


def _gather(snapshot: Snapshot) -> tuple[dict[str, str | None], dict[str, list[str]]]:
    selected: dict[str, str | None] = {
        path: None for path in (*snapshot.data_files, *snapshot.extra_files)}
    groups: dict[str, list[str]] = {}
    for group in snapshot.run_groups:
        directory = snapshot.root / RUNS / group
        _inspect(directory, directory=True)
        runs = []
        for child in sorted(directory.iterdir()):
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise RecoveryError(f"symlink paths are not allowed: {child}")
            if not stat.S_ISDIR(metadata.st_mode):
                if not stat.S_ISREG(metadata.st_mode):
                    raise RecoveryError(f"not a regular file or run directory: {child}")
                continue
            run = _asset_path(child.name, "run name")
            runs.append(run)
            names = set(REQUIRED_RUN_FILES)
            names.update(path.name for path in child.iterdir() if path.suffix == ".npy")
            for name in sorted(names):
                path = _asset_path(f"{RUNS}/{group}/{run}/{name}", "run asset")
                selected[path] = group
        if not runs:
            raise RecoveryError(f"configured run group contains no runs: {RUNS}/{group}")
        groups[group] = runs
    default_group, default_name = snapshot.default_run.split("/")
    if default_name not in groups[default_group]:
        raise RecoveryError(f"default embedding run is missing: {snapshot.default_run}")
    _check_collisions(tuple(selected))
    for path in sorted(selected):
        _inspect(snapshot.root / path)
    return dict(sorted(selected.items())), groups


def _validate_lock(data: dict, snapshot: Snapshot) -> tuple[dict, tuple[LockedFile, ...]]:
    _version(data, MANIFEST)
    _keys(data, {"version", "workspace", "archive", "run_groups", "files"}, MANIFEST)
    workspace = _keys(data["workspace"], {"path", "size", "sha256"}, "lock workspace")
    if (workspace["path"] != WORKSPACE
            or _size(workspace["size"], "lock workspace") != snapshot.workspace_size
            or _sha256(workspace["sha256"], "lock workspace") != snapshot.workspace_sha256):
        raise RecoveryError(f"{WORKSPACE} differs from the lock; restore the trusted metadata separately")
    archive = _keys(data["archive"], {"filename", "format", "size", "sha256"}, "lock archive")
    _name(archive["filename"], "archive filename")
    if archive["format"] != "tar.gz":
        raise RecoveryError("unsupported archive format (expected tar.gz)")
    if _size(archive["size"], "lock archive") == 0:
        raise RecoveryError("lock archive size must be positive")
    _sha256(archive["sha256"], "lock archive")
    groups = _keys(data["run_groups"], set(snapshot.run_groups), "lock run_groups")
    required = set((*snapshot.data_files, *snapshot.extra_files))
    for group, value in groups.items():
        runs = _strings(value, f"lock group {group}", nonempty=True)
        for run in runs:
            _name(run, "locked run")
            required.update(f"{RUNS}/{group}/{run}/{name}" for name in REQUIRED_RUN_FILES)
    default_group, default_name = snapshot.default_run.split("/")
    if default_name not in groups[default_group]:
        raise RecoveryError(f"default embedding run is absent from the lock: {snapshot.default_run}")
    if not isinstance(data["files"], list) or not data["files"]:
        raise RecoveryError("lock files must be a nonempty list")
    files = []
    fixed = set((*snapshot.data_files, *snapshot.extra_files))
    for item in data["files"]:
        item = _keys(item, {"path", "size", "sha256", "group"}, "locked file")
        path = _asset_path(item["path"], "locked file path")
        group = item["group"]
        if path in fixed:
            if group is not None:
                raise RecoveryError(f"non-run file has a run group: {path}")
        else:
            parts = path.split("/")
            if (len(parts) != 6 or "/".join(parts[:3]) != RUNS
                    or parts[3] not in groups or parts[4] not in groups[parts[3]]
                    or group != parts[3]
                    or (parts[5] not in REQUIRED_RUN_FILES and not parts[5].endswith(".npy"))):
                raise RecoveryError(f"undeclared or invalid locked run asset: {path}")
        files.append(LockedFile(path, _size(item["size"], path),
                                _sha256(item["sha256"], path), group))
    paths = [file.path for file in files]
    if len(set(paths)) != len(paths):
        raise RecoveryError("duplicate locked file paths")
    if paths != sorted(paths):
        raise RecoveryError("locked file paths must be sorted")
    _check_collisions(paths)
    if missing := required - set(paths):
        raise RecoveryError(f"lock is missing required files: {', '.join(sorted(missing))}")
    return archive, tuple(files)


def _load_lock(snapshot: Snapshot) -> tuple[dict, tuple[LockedFile, ...]]:
    data, _, _ = _read_json(snapshot.root / MANIFEST)
    return _validate_lock(data, snapshot)


def _make_parent(path: Path) -> None:
    _inspect(path.parent, directory=True, missing=True)
    with _directory_fd(path.parent, create=True):
        _inspect(path.parent, directory=True)


def _private_file(path: Path) -> BinaryIO:
    with _directory_fd(path.parent) as parent:
        return os.fdopen(os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                                | getattr(os, "O_BINARY", 0), 0o600, dir_fd=parent), "wb")


def _publish(staged: Path, destination: Path, expected: tuple[int, str]) -> bool:
    """Atomically create, never replace, a destination; return whether created."""
    _inspect(destination, missing=True)
    _inspect(staged)
    try:
        with _directory_fd(staged.parent) as source_parent, _directory_fd(destination.parent) as parent:
            os.link(staged.name, destination.name, src_dir_fd=source_parent,
                    dst_dir_fd=parent, follow_symlinks=False)
    except FileExistsError as exc:
        if _fingerprint(destination) != expected:
            raise RecoveryError(f"conflicting existing file (not overwritten): {destination}") from exc
        return False
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        with tempfile.TemporaryDirectory(prefix=".recovery-file-", dir=destination.parent) as temporary:
            local = Path(temporary) / "file"
            with _open_regular(staged) as source, _private_file(local) as target:
                if _digest(source, target) != expected:
                    raise RecoveryError(f"staged file changed: {destination}")
                target.flush()
                os.fsync(target.fileno())
            return _publish(local, destination, expected)
    _inspect(destination)
    return True


def pack(root: Path | str | None = None, *, output: Path | str) -> dict:
    """Create an immutable local tar.gz and atomically update its external lock.

    Return the JSON-compatible lock object. Existing identical archives may be
    reused, but there is deliberately no force-overwrite option.
    """
    snapshot = load_snapshot(root)
    selected, groups = _gather(snapshot)
    output = _absolute(output)
    _name(output.name, "output filename")
    if output in {snapshot.root / path for path in (*selected, WORKSPACE, MANIFEST)}:
        raise RecoveryError(f"archive output overlaps a required input or metadata file: {output}")
    _inspect(output, missing=True)
    manifest = snapshot.root / MANIFEST
    _inspect(manifest, missing=True)
    _make_parent(output)
    with tempfile.TemporaryDirectory(prefix=".recovery-pack-", dir=output.parent) as temporary:
        archive_path = Path(temporary) / "archive.tar.gz"
        files = []
        stamps = {}
        with _private_file(archive_path) as stream:
            with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                    for path, group in selected.items():
                        source_path = snapshot.root / path
                        with _open_regular(source_path) as source:
                            before = _stamp(os.fstat(source.fileno()))
                            member = tarfile.TarInfo(path)
                            member.size = before[2]
                            member.mode = 0o600
                            member.mtime = member.uid = member.gid = 0
                            reader = _HashingReader(source)
                            archive.addfile(member, reader)
                            if (reader.size != member.size or source.read(1)
                                    or _stamp(os.fstat(source.fileno())) != before):
                                raise RecoveryError(f"input changed during packaging: {path}")
                            stamps[path] = before
                            files.append(LockedFile(path, reader.size, reader.digest.hexdigest(), group))
            stream.flush()
            os.fsync(stream.fileno())
        if load_snapshot(snapshot.root) != snapshot or _gather(snapshot) != (selected, groups):
            raise RecoveryError("workspace/run inventory changed during packaging; stop writers and retry")
        for path, before in stamps.items():
            current = _inspect(snapshot.root / path)
            if current is None or _stamp(current) != before:
                raise RecoveryError(f"input changed during packaging: {path}")
        size, digest = _fingerprint(archive_path)
        lock = {
            "version": 1,
            "workspace": {"path": WORKSPACE, "size": snapshot.workspace_size,
                          "sha256": snapshot.workspace_sha256},
            "archive": {"filename": output.name, "format": "tar.gz",
                        "size": size, "sha256": digest},
            "run_groups": groups,
            "files": [asdict(file) for file in files],
        }
        _validate_lock(lock, snapshot)
        payload = (json.dumps(lock, indent=2, sort_keys=True) + "\n").encode("utf-8")
        if len(payload) > MAX_JSON_SIZE:
            raise RecoveryError(f"generated lock JSON exceeds {MAX_JSON_SIZE} bytes")
        with tempfile.TemporaryDirectory(prefix=".recovery-lock-", dir=manifest.parent) as lock_directory:
            staged_lock = Path(lock_directory) / "lock.json"
            with _private_file(staged_lock) as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            _publish(archive_path, output, (size, digest))
            with _open_regular(output) as stream:
                if os.name == "posix":
                    os.fchmod(stream.fileno(), 0o600)
            _inspect(manifest, missing=True)
            with _directory_fd(staged_lock.parent) as source_parent, _directory_fd(manifest.parent) as parent:
                os.replace(staged_lock.name, manifest.name,
                           src_dir_fd=source_parent, dst_dir_fd=parent)
    return lock


class _BoundedTarReader:
    """Prevent tar extended headers from requesting artifact-sized allocations."""

    def __init__(self, stream: BinaryIO):
        self.stream = stream

    def read(self, size: int) -> bytes:
        if size < 0 or size > CHUNK_SIZE:
            raise RecoveryError("archive requires an oversized metadata read")
        return self.stream.read(size)

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        return self.stream.seek(offset, whence)

    def tell(self) -> int:
        return self.stream.tell()


class _BoundedTarInfo(tarfile.TarInfo):
    @classmethod
    def frombuf(cls, buf: bytes, encoding: str, errors: str) -> tarfile.TarInfo:
        member = super().frombuf(buf, encoding, errors)
        extensions = (tarfile.XHDTYPE, tarfile.XGLTYPE,
                      tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK)
        # Newer tarfile versions read extensions in chunks, so a per-read cap
        # alone cannot enforce the declared metadata-size limit.
        if member.type in extensions and member.size > CHUNK_SIZE:
            raise RecoveryError("archive requires an oversized metadata read")
        return member


def _stage_archive(source: BinaryIO, stage: Path, files: tuple[LockedFile, ...]) -> None:
    expected = {file.path: file for file in files}
    seen = set()
    with gzip.GzipFile(fileobj=source, mode="rb") as compressed:
        with tarfile.open(fileobj=_BoundedTarReader(compressed), mode="r:",
                          tarinfo=_BoundedTarInfo) as archive:
            for member in archive:
                path = _relative(member.name, "archive member")
                if path in seen:
                    raise RecoveryError(f"duplicate archive member: {path}")
                if (member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE)
                        or member.linkname or member.sparse is not None
                        or set(member.pax_headers) - {"path", "size"}):
                    raise RecoveryError(f"archive member is not a plain regular file: {path}")
                if path not in expected:
                    raise RecoveryError(f"undeclared archive member: {path}")
                file = expected[path]
                if member.size != file.size:
                    raise RecoveryError(f"archive member size mismatch: {path}")
                destination = stage / path
                _make_parent(destination)
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise RecoveryError(f"cannot read archive member: {path}")
                with extracted, _private_file(destination) as target:
                    actual = _digest(extracted, target)
                    target.flush()
                    os.fsync(target.fileno())
                if actual != (file.size, file.sha256):
                    raise RecoveryError(f"archive member SHA-256/size mismatch: {path}")
                seen.add(path)
            if missing := set(expected) - seen:
                raise RecoveryError(f"archive is missing members: {', '.join(sorted(missing))}")
            # Consume remaining padding and the gzip footer, rather than silently
            # accepting another tar payload or a damaged gzip trailer after EOF.
            padding = 0
            for chunk in iter(lambda: archive.fileobj.read(CHUNK_SIZE), b""):
                padding += len(chunk)
                if chunk.strip(b"\0") or padding > tarfile.RECORDSIZE:
                    raise RecoveryError("unexpected trailing archive data")
            if padding < tarfile.BLOCKSIZE:
                raise RecoveryError("archive is missing its end markers")


def restore(root: Path | str | None = None, *, bundle: Path | str) -> Restoration:
    """Validate a caller-supplied bundle against a trusted lock, then restore.

    Existing exact files are skipped. Any differing file, including tracked
    JSON metadata, is an error; no metadata is silently repaired.
    """
    snapshot = load_snapshot(root)
    archive, files = _load_lock(snapshot)
    bundle = _absolute(bundle)
    with _open_regular(bundle) as source:
        before = _stamp(os.fstat(source.fileno()))
        if before[2] != archive["size"]:
            raise RecoveryError(f"archive size mismatch: {bundle}")
        if _digest(source) != (archive["size"], archive["sha256"]):
            raise RecoveryError(f"archive SHA-256 mismatch: {bundle}")
        if _stamp(os.fstat(source.fileno())) != before:
            raise RecoveryError(f"archive changed while hashing: {bundle}")
        source.seek(0)
        with tempfile.TemporaryDirectory(prefix=".recovery-restore-", dir=snapshot.root) as temporary:
            stage = Path(temporary)
            _stage_archive(source, stage, files)
            if _stamp(os.fstat(source.fileno())) != before:
                raise RecoveryError(f"archive changed while staging: {bundle}")
            if (load_snapshot(snapshot.root) != snapshot
                    or _load_lock(snapshot) != (archive, files)):
                raise RecoveryError("workspace/lock metadata changed during restore")
            pending, skipped, conflicts = [], [], []
            for file in files:
                destination = snapshot.root / file.path
                current = _inspect(destination, missing=True)
                if current is None:
                    pending.append(file)
                elif (current.st_size != file.size
                      or _fingerprint(destination) != (file.size, file.sha256)):
                    conflicts.append(file.path)
                else:
                    skipped.append(file.path)
            if conflicts:
                raise RecoveryError("conflicting existing files (not overwritten): "
                                    + ", ".join(conflicts))
            restored = []
            for file in pending:
                destination = snapshot.root / file.path
                _make_parent(destination)
                if _publish(stage / file.path, destination, (file.size, file.sha256)):
                    restored.append(file.path)
                else:
                    skipped.append(file.path)
    return Restoration(tuple(restored), tuple(sorted(skipped)))


def verify(root: Path | str | None = None) -> Verification:
    """Compare every locked local file; the archive itself is not required."""
    snapshot = load_snapshot(root)
    _, files = _load_lock(snapshot)
    missing, mismatched = [], []
    for file in files:
        path = snapshot.root / file.path
        current = _inspect(path, missing=True)
        if current is None:
            missing.append(file.path)
        elif current.st_size != file.size or _fingerprint(path) != (file.size, file.sha256):
            mismatched.append(file.path)
    return Verification(len(files), tuple(missing), tuple(mismatched))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="checkout root (default: resolved from this module)")
    commands = parser.add_subparsers(dest="command", required=True)
    pack_parser = commands.add_parser("pack", help="create a local immutable tar.gz and lock")
    pack_parser.add_argument("--output", type=Path, required=True)
    restore_parser = commands.add_parser("restore", help="validate and restore missing locked files")
    restore_parser.add_argument("--bundle", type=Path, required=True)
    commands.add_parser("verify", help="check every locked local file")
    args = parser.parse_args(argv)
    try:
        if args.command == "pack":
            lock = pack(args.root, output=args.output)
            print(f"Packed {len(lock['files'])} files into {args.output}; wrote {MANIFEST}.")
        elif args.command == "restore":
            result = restore(args.root, bundle=args.bundle)
            print(f"Restored {len(result.restored)} files; skipped {len(result.skipped)} exact matches.")
        else:
            result = verify(args.root)
            for path in result.missing:
                print(f"missing: {path}")
            for path in result.mismatched:
                print(f"mismatched: {path}")
            if not result.ok:
                return 1
            print(f"Verified {result.total} locked files.")
    except (RecoveryError, OSError, tarfile.TarError, EOFError, zlib.error) as exc:
        print(f"recovery: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
