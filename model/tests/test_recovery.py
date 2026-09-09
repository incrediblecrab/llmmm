"""Recovery uses tiny opaque files, never installed scientific packages or corpora."""
from __future__ import annotations

import copy
import errno
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tarfile

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "ingredient_model" / "recovery.py"
SPEC = importlib.util.spec_from_file_location("standalone_recovery", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
recovery = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = recovery
SPEC.loader.exec_module(recovery)

DATA = (
    "model/data/GENERATION.json",
    "model/data/graphs/ii_graph.npz",
    "prior-study/data/raw/epicure-cooc/vocab.json",
)
EXTRA = "model/results/m6_ranks.npz"
GROUPS = {"benchmark": ("z-run", "a-run"), "current": ("trained-run",)}
ARRAY = "model/results/runs/benchmark/a-run/state__encoder__weight.npy"
MISSING_WEIGHT = "model/results/runs/current/trained-run/embedding.npy"
METADATA = "model/results/runs/current/trained-run/metrics.json"


def write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def write_json(path: Path, payload: dict) -> None:
    write(path, (json.dumps(payload, indent=2) + "\n").encode())


def expected_paths() -> set[str]:
    paths = {*DATA, EXTRA, ARRAY}
    for group, runs in GROUPS.items():
        for run in runs:
            paths.update(f"model/results/runs/{group}/{run}/{name}"
                         for name in ("embedding.npy", "manifest.json", "metrics.json"))
    return paths


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    write_json(root / recovery.WORKSPACE, {
        "version": 1,
        "benchmark_sweep": "benchmark",
        "training_sweep": "current",
        "default_embedding_run": "benchmark/a-run",
        "snapshot": {
            "manifest": recovery.MANIFEST,
            "data_files": list(DATA),
            "run_groups": list(GROUPS),
            "extra_files": [EXTRA],
        },
    })
    for path in expected_paths():
        if path.endswith(".json"):
            write_json(root / path, {"fixture": path})
        else:
            write(root / path, b"\x00\xffopaque array fixture: " + path.encode())
    for path in (
        "raw-data/not-for-export.json",
        "model/.env",
        "model/.venv/not-for-export.npy",
        "model/cloud/config.json",
        "model/data/unlisted.npy",
        "model/results/unlisted.npy",
        "model/results/runs/benchmark/journal.jsonl",
        "model/results/runs/benchmark/sweep.log",
        "model/results/runs/benchmark/a-run/notes.txt",
        "model/results/runs/benchmark/a-run/nested/not-included.npy",
        "model/results/runs/unselected/other/embedding.npy",
    ):
        write(root / path, b"excluded fixture, not a real secret")
    return root


@pytest.fixture
def packed(source, tmp_path):
    bundle = tmp_path / "portable.tar.gz"
    lock = recovery.pack(source, output=bundle)
    return source, bundle, lock


def fresh_checkout(source: Path, tmp_path: Path, *, metadata: bool = True) -> Path:
    root = tmp_path / "fresh"
    for path in (recovery.WORKSPACE, recovery.MANIFEST):
        write(root / path, (source / path).read_bytes())
    if metadata:
        for path in expected_paths():
            if path.endswith(".json"):
                write(root / path, (source / path).read_bytes())
    return root


def tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if not path.is_symlink() and path.is_file()
    }


def no_temporaries(root: Path) -> None:
    assert not list(root.rglob(".recovery-*"))


def archive_members(bundle: Path) -> list[tuple[tarfile.TarInfo, bytes]]:
    with tarfile.open(bundle, "r:gz") as archive:
        members = []
        for member in archive:
            stream = archive.extractfile(member)
            assert stream is not None
            with stream:
                members.append((copy.copy(member), stream.read()))
        return members


def update_archive_lock(source: Path, bundle: Path) -> None:
    path = source / recovery.MANIFEST
    lock = json.loads(path.read_text())
    payload = bundle.read_bytes()
    lock["archive"]["size"] = len(payload)
    lock["archive"]["sha256"] = hashlib.sha256(payload).hexdigest()
    write_json(path, lock)


def rewrite_archive(source: Path, bundle: Path,
                    members: list[tuple[tarfile.TarInfo, bytes]]) -> None:
    with tarfile.open(bundle, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for member, payload in members:
            archive.addfile(member, io.BytesIO(payload) if member.isreg() else None)
    update_archive_lock(source, bundle)


def assert_rejected_without_live_changes(source, bundle, tmp_path, match):
    fresh = fresh_checkout(source, tmp_path)
    before = tree(fresh)
    with pytest.raises(recovery.RecoveryError, match=match):
        recovery.restore(fresh, bundle=bundle)
    assert tree(fresh) == before
    no_temporaries(tmp_path)


def test_round_trip_with_tracked_metadata_and_idempotence(packed, tmp_path):
    source, bundle, lock = packed
    paths = [item["path"] for item in lock["files"]]
    assert paths == sorted(expected_paths())
    assert lock["run_groups"] == {group: sorted(runs) for group, runs in GROUPS.items()}
    assert lock["workspace"]["path"] == recovery.WORKSPACE
    assert lock["workspace"]["sha256"] == hashlib.sha256(
        (source / recovery.WORKSPACE).read_bytes()).hexdigest()
    assert recovery.WORKSPACE not in paths
    assert recovery.MANIFEST not in paths
    for item in lock["files"]:
        payload = (source / item["path"]).read_bytes()
        assert item["size"] == len(payload)
        assert item["sha256"] == hashlib.sha256(payload).hexdigest()
        expected_group = item["path"].split("/")[3] if "/runs/" in item["path"] else None
        assert item["group"] == expected_group
    with tarfile.open(bundle, "r:gz") as archive:
        assert archive.getnames() == paths
        for member in archive:
            assert member.isfile()
            assert member.mtime == member.uid == member.gid == 0
            assert member.uname == member.gname == ""
            assert member.mode == 0o600
    assert lock["archive"]["size"] == bundle.stat().st_size
    assert lock["archive"]["sha256"] == hashlib.sha256(bundle.read_bytes()).hexdigest()

    fresh = fresh_checkout(source, tmp_path)
    tracked = tuple(path for path in paths if path.endswith(".json"))
    before = {path: (fresh / path).stat().st_mtime_ns for path in tracked}
    result = recovery.restore(fresh, bundle=bundle)
    assert result.skipped == tracked
    assert set(result.restored) == expected_paths() - set(tracked)
    assert recovery.verify(fresh) == recovery.Verification(len(paths), (), ())
    for path in paths:
        assert (fresh / path).read_bytes() == (source / path).read_bytes()
    assert {path: (fresh / path).stat().st_mtime_ns for path in tracked} == before

    mtimes = {path: (fresh / path).stat().st_mtime_ns for path in paths}
    again = recovery.restore(fresh, bundle=bundle)
    assert again.restored == ()
    assert again.skipped == tuple(paths)
    assert {path: (fresh / path).stat().st_mtime_ns for path in paths} == mtimes
    no_temporaries(tmp_path)


def test_restores_into_config_and_lock_only_checkout(packed, tmp_path):
    source, bundle, lock = packed
    fresh = fresh_checkout(source, tmp_path, metadata=False)
    result = recovery.restore(fresh, bundle=bundle)
    assert result.skipped == ()
    assert set(result.restored) == expected_paths()
    assert recovery.verify(fresh).ok
    assert json.loads((fresh / recovery.MANIFEST).read_text()) == lock


def test_pack_is_deterministic_and_reuses_identical_archives(packed, tmp_path):
    source, bundle, first_lock = packed
    original = bundle.read_bytes()
    original_mtime = bundle.stat().st_mtime_ns
    assert original[4:8] == b"\0" * 4
    assert original[3] & 8 == 0
    os.utime(source / ARRAY, (1000000, 1000000))
    second_lock = recovery.pack(source, output=bundle)
    assert second_lock == first_lock
    assert bundle.read_bytes() == original
    assert bundle.stat().st_mtime_ns == original_mtime
    elsewhere = tmp_path / "a-different-name.bin"
    recovery.pack(source, output=elsewhere)
    assert elsewhere.read_bytes() == original
    no_temporaries(tmp_path)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_archive_and_restored_files_are_owner_only(packed, tmp_path):
    source, bundle, _ = packed
    assert stat.S_IMODE(bundle.stat().st_mode) == 0o600
    bundle.chmod(0o644)
    recovery.pack(source, output=bundle)
    assert stat.S_IMODE(bundle.stat().st_mode) == 0o600
    fresh = fresh_checkout(source, tmp_path, metadata=False)
    recovery.restore(fresh, bundle=bundle)
    for path in expected_paths():
        assert stat.S_IMODE((fresh / path).stat().st_mode) == 0o600


def test_existing_different_archive_and_previous_lock_are_unchanged(packed, tmp_path):
    source, old_bundle, _ = packed
    lock_before = (source / recovery.MANIFEST).read_bytes()
    old_before = old_bundle.read_bytes()
    different = tmp_path / "existing.bin"
    different.write_bytes(b"do not overwrite this")
    with pytest.raises(recovery.RecoveryError, match="conflicting existing file"):
        recovery.pack(source, output=different)
    assert different.read_bytes() == b"do not overwrite this"
    assert (source / recovery.MANIFEST).read_bytes() == lock_before
    assert old_bundle.read_bytes() == old_before
    no_temporaries(tmp_path)


def test_failed_first_pack_publishes_no_lock(source, tmp_path):
    output = tmp_path / "existing.bin"
    output.write_bytes(b"do not overwrite")
    with pytest.raises(recovery.RecoveryError, match="conflicting existing file"):
        recovery.pack(source, output=output)
    assert not (source / recovery.MANIFEST).exists()
    assert output.read_bytes() == b"do not overwrite"
    no_temporaries(tmp_path)


def test_lock_publication_failure_is_not_reported_as_success(source, tmp_path, monkeypatch, capsys):
    output = tmp_path / "archive.tar.gz"

    def fail_replace(*args, **kwargs):
        raise PermissionError(errno.EACCES, "injected lock publication failure")

    monkeypatch.setattr(recovery.os, "replace", fail_replace)
    assert recovery.main(["--root", str(source), "pack", "--output", str(output)]) == 1
    captured = capsys.readouterr()
    assert "publication failure" in captured.err
    assert "Packed" not in captured.out
    assert not (source / recovery.MANIFEST).exists()
    assert output.is_file()
    no_temporaries(tmp_path)


def test_input_change_during_pack_fails_without_new_archive_or_lock(source, tmp_path, monkeypatch):
    addfile = recovery.tarfile.TarFile.addfile

    def change_input(archive, member, fileobj=None):
        addfile(archive, member, fileobj)
        if member.name == ARRAY:
            write(source / ARRAY, b"changed by an in-progress writer")

    monkeypatch.setattr(recovery.tarfile.TarFile, "addfile", change_input)
    bundle = tmp_path / "unstable.tar.gz"
    with pytest.raises(recovery.RecoveryError, match="changed during packaging"):
        recovery.pack(source, output=bundle)
    assert not bundle.exists()
    assert not (source / recovery.MANIFEST).exists()
    no_temporaries(tmp_path)


def test_new_array_during_pack_is_not_silently_omitted(source, tmp_path, monkeypatch):
    addfile = recovery.tarfile.TarFile.addfile

    def add_array(archive, member, fileobj=None):
        addfile(archive, member, fileobj)
        if member.name == ARRAY:
            write((source / ARRAY).with_name("new_state.npy"), b"another array")

    monkeypatch.setattr(recovery.tarfile.TarFile, "addfile", add_array)
    bundle = tmp_path / "unstable.tar.gz"
    with pytest.raises(recovery.RecoveryError, match="inventory changed"):
        recovery.pack(source, output=bundle)
    assert not bundle.exists()
    assert not (source / recovery.MANIFEST).exists()
    no_temporaries(tmp_path)


def test_pack_does_not_publish_a_lock_too_large_for_its_reader(source, tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "MAX_JSON_SIZE", 2048)
    assert (source / recovery.WORKSPACE).stat().st_size < recovery.MAX_JSON_SIZE
    bundle = tmp_path / "oversized-lock.tar.gz"
    with pytest.raises(recovery.RecoveryError, match="generated lock JSON exceeds"):
        recovery.pack(source, output=bundle)
    assert not bundle.exists()
    assert not (source / recovery.MANIFEST).exists()
    no_temporaries(tmp_path)


@pytest.mark.parametrize("path", [
    DATA[0], DATA[1], EXTRA,
    "model/results/runs/benchmark/a-run/manifest.json",
    "model/results/runs/benchmark/a-run/metrics.json",
    "model/results/runs/benchmark/a-run/embedding.npy",
])
def test_missing_required_pack_inputs_fail(source, tmp_path, path):
    (source / path).unlink()
    bundle = tmp_path / "partial.tar.gz"
    with pytest.raises(recovery.RecoveryError, match="missing required path"):
        recovery.pack(source, output=bundle)
    assert not bundle.exists()
    assert not (source / recovery.MANIFEST).exists()
    no_temporaries(tmp_path)


@pytest.mark.parametrize("state", ["missing", "empty", "incomplete", "missing-default"])
def test_missing_or_incomplete_run_inventory_fails(source, tmp_path, state):
    group = source / "model/results/runs/current"
    if state == "missing":
        group.rename(tmp_path / "saved-group")
    elif state == "empty":
        (group / "trained-run").rename(tmp_path / "saved-run")
    elif state == "incomplete":
        (group / "partial-run").mkdir()
    else:
        (source / "model/results/runs/benchmark/a-run").rename(tmp_path / "saved-run")
    with pytest.raises(recovery.RecoveryError, match="missing|no runs"):
        recovery.pack(source, output=tmp_path / "partial.tar.gz")
    assert not (source / recovery.MANIFEST).exists()


@pytest.mark.parametrize("different_size", [False, True])
def test_verify_reports_all_missing_and_mismatched_paths(packed, capsys, different_size):
    source, _, lock = packed
    path = source / DATA[1]
    original = path.read_bytes()
    path.write_bytes(b"short" if different_size else bytes([original[0] ^ 1]) + original[1:])
    (source / MISSING_WEIGHT).unlink()
    result = recovery.verify(source)
    assert not result.ok
    assert result.total == len(lock["files"])
    assert result.missing == (MISSING_WEIGHT,)
    assert result.mismatched == (DATA[1],)
    assert recovery.main(["--root", str(source), "verify"]) == 1
    output = capsys.readouterr().out
    assert f"missing: {MISSING_WEIGHT}" in output
    assert f"mismatched: {DATA[1]}" in output
    assert "Verified" not in output


def test_verify_does_not_require_the_archive(packed):
    source, bundle, _ = packed
    bundle.unlink()
    assert recovery.verify(source).ok


@pytest.mark.parametrize("damage", ["bytes", "size"])
def test_real_archive_corruption_fails_before_extraction(packed, tmp_path, capsys, damage):
    source, bundle, _ = packed
    fresh = fresh_checkout(source, tmp_path)
    before = tree(fresh)
    payload = bundle.read_bytes()
    if damage == "bytes":
        middle = len(payload) // 2
        payload = payload[:middle] + bytes([payload[middle] ^ 1]) + payload[middle + 1:]
    else:
        payload += b"unexpected byte"
    bundle.write_bytes(payload)
    assert recovery.main(["--root", str(fresh), "restore", "--bundle", str(bundle)]) == 1
    captured = capsys.readouterr()
    assert ("archive SHA-256 mismatch" if damage == "bytes" else "archive size mismatch") in captured.err
    assert "Restored" not in captured.out
    assert tree(fresh) == before
    no_temporaries(tmp_path)


@pytest.mark.parametrize("path", [MISSING_WEIGHT, METADATA])
def test_conflicts_preserve_existing_data_and_prevent_other_restores(packed, tmp_path, path):
    source, bundle, _ = packed
    fresh = fresh_checkout(source, tmp_path)
    write(fresh / path, b"local differing contents must survive")
    before = tree(fresh)
    with pytest.raises(recovery.RecoveryError, match="conflicting existing files.*" + path):
        recovery.restore(fresh, bundle=bundle)
    assert tree(fresh) == before
    assert not (fresh / DATA[1]).exists()
    no_temporaries(tmp_path)


@pytest.mark.parametrize("name", [
    "../escape.npy",
    "/absolute.npy",
    r"..\escape.npy",
    r"C:\escape.npy",
    "C:/escape.npy",
    "C:escape.npy",
    "//server/escape.npy",
    "model//data/escape.npy",
    "model/data/./escape.npy",
    "model/data/../escape.npy",
    "model/data/escape.npy/",
    "model/data/NUL.npy",
])
def test_malicious_archive_paths_are_rejected_after_staging_only(packed, tmp_path, name):
    source, bundle, _ = packed
    members = archive_members(bundle)
    member = tarfile.TarInfo(name)
    member.size = 4
    rewrite_archive(source, bundle, [*members, (member, b"evil")])
    assert_rejected_without_live_changes(source, bundle, tmp_path, "relative POSIX path")
    assert not (tmp_path / "escape.npy").exists()


@pytest.mark.parametrize("kind", [
    tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE,
    tarfile.CHRTYPE, tarfile.BLKTYPE, tarfile.FIFOTYPE,
])
def test_nonregular_archive_members_are_rejected(packed, tmp_path, kind):
    source, bundle, _ = packed
    members = archive_members(bundle)
    removed, _ = members.pop()
    malicious = tarfile.TarInfo(removed.name)
    malicious.type = kind
    if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
        malicious.linkname = "../outside"
    rewrite_archive(source, bundle, [*members, (malicious, b"")])
    assert_rejected_without_live_changes(source, bundle, tmp_path, "plain regular file")


def test_duplicate_archive_members_are_rejected(packed, tmp_path):
    source, bundle, _ = packed
    members = archive_members(bundle)
    rewrite_archive(source, bundle, [*members, members[0]])
    assert_rejected_without_live_changes(source, bundle, tmp_path, "duplicate archive member")


def test_undeclared_archive_members_are_rejected(packed, tmp_path):
    source, bundle, _ = packed
    member = tarfile.TarInfo("model/data/undeclared.npy")
    member.size = 4
    rewrite_archive(source, bundle, [*archive_members(bundle), (member, b"evil")])
    assert_rejected_without_live_changes(source, bundle, tmp_path, "undeclared archive member")


@pytest.mark.parametrize("damage", ["missing", "size", "hash"])
def test_archive_inventory_and_member_bytes_must_match_lock(packed, tmp_path, damage):
    source, bundle, _ = packed
    members = archive_members(bundle)
    member, payload = members.pop()
    if damage == "size":
        member.size += 1
        members.append((member, payload + b"x"))
    elif damage == "hash":
        members.append((member, bytes([payload[0] ^ 1]) + payload[1:]))
    rewrite_archive(source, bundle, members)
    expected = {"missing": "missing members", "size": "size mismatch", "hash": "SHA-256/size mismatch"}
    assert_rejected_without_live_changes(source, bundle, tmp_path, expected[damage])


def test_trailing_tar_payload_is_not_ignored(packed, tmp_path):
    source, bundle, _ = packed
    decoded = gzip.decompress(bundle.read_bytes())
    bundle.write_bytes(gzip.compress(decoded + b"undeclared trailing payload"))
    update_archive_lock(source, bundle)
    assert_rejected_without_live_changes(source, bundle, tmp_path, "trailing archive data")


@pytest.mark.parametrize("kind", [
    tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK,
    tarfile.SOLARIS_XHDTYPE,
])
def test_oversized_tar_extension_is_rejected_without_reading_its_claimed_payload(
        packed, tmp_path, kind):
    source, bundle, _ = packed
    member = tarfile.TarInfo("oversized-pax")
    member.type = kind
    member.size = recovery.CHUNK_SIZE + tarfile.BLOCKSIZE
    header = member.tobuf(format=tarfile.USTAR_FORMAT)
    bundle.write_bytes(gzip.compress(header + b"\0" * tarfile.BLOCKSIZE * 2))
    update_archive_lock(source, bundle)
    assert_rejected_without_live_changes(source, bundle, tmp_path, "oversized metadata read")


def test_pack_and_restore_support_long_array_names_with_pax_headers(source, tmp_path):
    path = f"model/results/runs/benchmark/a-run/state__{'x' * 150}.npy"
    write(source / path, b"long-named state array")
    bundle = tmp_path / "long-name.tar.gz"
    lock = recovery.pack(source, output=bundle)
    assert path in {file["path"] for file in lock["files"]}
    with tarfile.open(bundle, "r:gz") as archive:
        assert archive.getmember(path).pax_headers["path"] == path
    fresh = fresh_checkout(source, tmp_path)
    recovery.restore(fresh, bundle=bundle)
    assert (fresh / path).read_bytes() == b"long-named state array"
    assert recovery.verify(fresh).ok


def test_payloads_are_hashed_and_copied_in_bounded_chunks(source, tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "CHUNK_SIZE", 16 * 1024)
    write(source / ARRAY, b"x" * (3 * recovery.CHUNK_SIZE + 1))
    read = recovery._HashingReader.read
    requested_sizes = []

    def bounded_read(reader, size):
        assert 0 <= size <= recovery.CHUNK_SIZE
        requested_sizes.append(size)
        return read(reader, size)

    monkeypatch.setattr(recovery._HashingReader, "read", bounded_read)
    bundle = tmp_path / "chunked.tar.gz"
    recovery.pack(source, output=bundle)
    fresh = fresh_checkout(source, tmp_path)
    recovery.restore(fresh, bundle=bundle)
    assert recovery.verify(fresh).ok
    assert max(requested_sizes) == recovery.CHUNK_SIZE
    assert (fresh / ARRAY).read_bytes() == (source / ARRAY).read_bytes()


def test_gzip_footer_is_validated_even_when_outer_hash_matches(packed, tmp_path, capsys):
    source, bundle, _ = packed
    payload = bundle.read_bytes()
    bundle.write_bytes(payload[:-8] + bytes([payload[-8] ^ 1]) + payload[-7:])
    update_archive_lock(source, bundle)
    fresh = fresh_checkout(source, tmp_path)
    before = tree(fresh)
    assert recovery.main(["--root", str(fresh), "restore", "--bundle", str(bundle)]) == 1
    captured = capsys.readouterr()
    assert "CRC" in captured.err
    assert "Restored" not in captured.out
    assert tree(fresh) == before
    no_temporaries(tmp_path)


def test_malformed_archive_with_matching_outer_hash_returns_nonzero(packed, tmp_path, capsys):
    source, bundle, _ = packed
    bundle.write_bytes(b"this is not gzip or tar")
    update_archive_lock(source, bundle)
    fresh = fresh_checkout(source, tmp_path)
    before = tree(fresh)
    assert recovery.main(["--root", str(fresh), "restore", "--bundle", str(bundle)]) == 1
    assert "Restored" not in capsys.readouterr().out
    assert tree(fresh) == before
    no_temporaries(tmp_path)


def test_restore_uses_explicit_bundle_bytes_not_a_filename_guess(packed, tmp_path):
    source, bundle, _ = packed
    renamed = tmp_path / "copied-privately.bin"
    bundle.rename(renamed)
    fresh = fresh_checkout(source, tmp_path)
    assert recovery.restore(fresh, bundle=renamed).restored
    assert recovery.verify(fresh).ok


@pytest.mark.parametrize("location", ["input", "input-parent", "group", "output", "config", "root"])
def test_pack_rejects_symlinks_including_parent_directories(source, tmp_path, location):
    bundle = tmp_path / "archive.tar.gz"
    root = source
    if location == "input":
        path = source / ARRAY
    elif location == "input-parent":
        path = source / "model/data/graphs"
    elif location == "group":
        path = source / "model/results/runs/current"
    elif location == "config":
        path = source / recovery.WORKSPACE
    elif location == "output":
        path = bundle
        path.write_bytes(b"existing output target")
    else:
        path = root
    saved = tmp_path / "original"
    path.rename(saved)
    path.symlink_to(saved, target_is_directory=saved.is_dir())
    before = tree(saved) if saved.is_dir() else saved.read_bytes()
    with pytest.raises(recovery.RecoveryError, match="symlink"):
        recovery.pack(root, output=bundle)
    assert (tree(saved) if saved.is_dir() else saved.read_bytes()) == before
    no_temporaries(tmp_path)


@pytest.mark.parametrize("location", ["leaf", "parent", "group-parent", "lock", "bundle"])
def test_restore_rejects_symlink_destinations_and_inputs(packed, tmp_path, location):
    source, bundle, _ = packed
    fresh = fresh_checkout(source, tmp_path)
    outside = tmp_path / "outside"
    if location == "parent":
        path = fresh / "model/data/graphs"
        outside.mkdir()
        write(outside / "marker", b"outside must be unchanged")
    elif location == "group-parent":
        path = fresh / "model/results/runs/current"
        path.rename(outside)
    elif location == "lock":
        path = fresh / recovery.MANIFEST
        path.rename(outside)
    elif location == "bundle":
        path = bundle
        path.rename(outside)
    else:
        path = fresh / MISSING_WEIGHT
        outside.write_bytes(b"outside must be unchanged")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(outside, target_is_directory=outside.is_dir())
    before = tree(fresh)
    outside_before = tree(outside) if outside.is_dir() else outside.read_bytes()
    with pytest.raises(recovery.RecoveryError, match="symlink"):
        recovery.restore(fresh, bundle=bundle)
    assert tree(fresh) == before
    assert (tree(outside) if outside.is_dir() else outside.read_bytes()) == outside_before
    no_temporaries(tmp_path)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX FIFO support")
def test_pack_rejects_special_input_files_without_opening_them(source, tmp_path):
    path = source / ARRAY
    path.unlink()
    os.mkfifo(path)
    with pytest.raises(recovery.RecoveryError, match="not a regular file"):
        recovery.pack(source, output=tmp_path / "archive.tar.gz")
    assert not (source / recovery.MANIFEST).exists()


@pytest.mark.parametrize("path", [
    "raw-data/corpus.npy",
    "model/.env",
    "model/data/.env.json",
    "model/data/.venv/array.npy",
    "model/cloud/config.json",
    "model/data/cloud/config.json",
    "model/data/tokens.json",
    "model/data/access_token.json",
    "model/data/credentials.json",
    "model/data/private-key.json",
    "model/ingredient_model/code.npy",
    "../outside.npy",
    "/absolute.npy",
    r"model\data\array.npy",
    "model/data/*.npz",
])
def test_config_cannot_declare_secret_or_out_of_scope_files(source, tmp_path, path):
    config_path = source / recovery.WORKSPACE
    config = json.loads(config_path.read_text())
    config["snapshot"]["data_files"].append(path)
    write_json(config_path, config)
    with pytest.raises(recovery.RecoveryError):
        recovery.pack(source, output=tmp_path / "archive.tar.gz")
    assert not (source / recovery.MANIFEST).exists()


@pytest.mark.parametrize("mutate", [
    pytest.param(lambda data: data.update(version=True), id="boolean-version"),
    pytest.param(lambda data: data.update(version=2), id="unknown-version"),
    pytest.param(lambda data: data.pop("snapshot"), id="missing-snapshot"),
    pytest.param(lambda data: data["snapshot"].update(data_files="not-a-list"), id="wrong-files-type"),
    pytest.param(lambda data: data["snapshot"].update(data_files=[]), id="empty-data"),
    pytest.param(lambda data: data["snapshot"]["data_files"].append(DATA[0]), id="duplicate-data"),
    pytest.param(lambda data: data["snapshot"].update(manifest=recovery.WORKSPACE), id="metadata-overlap"),
    pytest.param(lambda data: data["snapshot"].update(run_groups=[]), id="empty-groups"),
    pytest.param(lambda data: data["snapshot"].update(run_groups=["benchmark"]), id="missing-sweep"),
    pytest.param(lambda data: data["snapshot"]["run_groups"].append("benchmark"), id="duplicate-group"),
    pytest.param(lambda data: data["snapshot"]["run_groups"].append("../outside"), id="unsafe-group"),
    pytest.param(lambda data: data["snapshot"]["run_groups"].append("Benchmark"), id="case-alias"),
    pytest.param(lambda data: data.update(default_embedding_run="unselected/a-run"), id="wrong-default-group"),
    pytest.param(lambda data: data.update(default_embedding_run="benchmark"), id="missing-default-run"),
    pytest.param(lambda data: data["snapshot"].update(extra_files=[ARRAY]), id="extra-cannot-bypass-runs"),
    pytest.param(lambda data: data["snapshot"].update(unrecognized=[]), id="misspelled-snapshot-field"),
])
def test_invalid_workspace_schema_fails_closed(source, tmp_path, mutate):
    path = source / recovery.WORKSPACE
    config = json.loads(path.read_text())
    mutate(config)
    write_json(path, config)
    with pytest.raises(recovery.RecoveryError):
        recovery.pack(source, output=tmp_path / "archive.tar.gz")
    assert not (source / recovery.MANIFEST).exists()


@pytest.mark.parametrize("payload", [b"{broken", b"[]", b'{"version": 1, "version": 1}', b'{"value": NaN}'])
def test_malformed_json_is_a_descriptive_cli_error(source, capsys, payload):
    (source / recovery.WORKSPACE).write_bytes(payload)
    assert recovery.main(["--root", str(source), "verify"]) == 1
    captured = capsys.readouterr()
    assert recovery.WORKSPACE in captured.err
    assert "JSON" in captured.err
    assert not captured.out


@pytest.mark.parametrize("mutate", [
    pytest.param(lambda lock: lock.update(version=True), id="boolean-version"),
    pytest.param(lambda lock: lock.update(version=2), id="unknown-version"),
    pytest.param(lambda lock: lock.pop("archive"), id="missing-archive"),
    pytest.param(lambda lock: lock.update(files={}), id="wrong-files-type"),
    pytest.param(lambda lock: lock["files"].append(copy.deepcopy(lock["files"][0])), id="duplicate-file"),
    pytest.param(lambda lock: lock["files"].reverse(), id="unordered-files"),
    pytest.param(lambda lock: lock["files"][0].update(size=True), id="boolean-size"),
    pytest.param(lambda lock: lock["files"][0].update(size=-1), id="negative-size"),
    pytest.param(lambda lock: lock["files"][0].update(sha256="not-a-digest"), id="bad-file-digest"),
    pytest.param(lambda lock: lock["files"][0].update(path="../escape.npy"), id="unsafe-file"),
    pytest.param(lambda lock: lock["files"][0].update(group="benchmark"), id="wrong-file-group"),
    pytest.param(lambda lock: lock.update(files=[f for f in lock["files"] if f["path"] != MISSING_WEIGHT]),
                 id="missing-required-run-asset"),
    pytest.param(lambda lock: lock.update(files=[f for f in lock["files"] if f["path"] != DATA[0]]),
                 id="missing-required-data"),
    pytest.param(lambda lock: lock["run_groups"].update(current=[]), id="empty-group"),
    pytest.param(lambda lock: lock["run_groups"].pop("current"), id="missing-group"),
    pytest.param(lambda lock: lock["run_groups"].update(benchmark=["z-run"]), id="missing-default-run"),
    pytest.param(lambda lock: lock["archive"].update(sha256="a" * 63), id="bad-archive-digest"),
    pytest.param(lambda lock: lock["archive"].update(size=0), id="empty-archive"),
    pytest.param(lambda lock: lock["archive"].update(size=True), id="boolean-archive-size"),
    pytest.param(lambda lock: lock["archive"].update(format="zip"), id="unknown-format"),
    pytest.param(lambda lock: lock["archive"].update(filename="../bundle.tar.gz"), id="unsafe-archive-name"),
    pytest.param(lambda lock: lock["workspace"].update(path="model/other.json"), id="wrong-workspace"),
])
def test_invalid_lock_cannot_restore_or_verify(packed, tmp_path, mutate):
    source, bundle, lock = packed
    mutate(lock)
    write_json(source / recovery.MANIFEST, lock)
    fresh = fresh_checkout(source, tmp_path)
    before = tree(fresh)
    with pytest.raises(recovery.RecoveryError):
        recovery.restore(fresh, bundle=bundle)
    with pytest.raises(recovery.RecoveryError):
        recovery.verify(fresh)
    assert tree(fresh) == before
    no_temporaries(tmp_path)


def test_changed_workspace_metadata_is_never_repaired(packed, tmp_path):
    source, bundle, _ = packed
    fresh = fresh_checkout(source, tmp_path)
    path = fresh / recovery.WORKSPACE
    path.write_bytes(path.read_bytes() + b"\n")
    before = tree(fresh)
    with pytest.raises(recovery.RecoveryError, match="differs from the lock"):
        recovery.restore(fresh, bundle=bundle)
    with pytest.raises(recovery.RecoveryError, match="differs from the lock"):
        recovery.verify(fresh)
    assert tree(fresh) == before


@pytest.mark.parametrize("metadata", [recovery.WORKSPACE, recovery.MANIFEST])
def test_metadata_changes_during_staging_prevent_live_publication(
        packed, tmp_path, monkeypatch, metadata):
    source, bundle, _ = packed
    fresh = fresh_checkout(source, tmp_path)
    before = tree(fresh)
    stage_archive = recovery._stage_archive

    def change_metadata(*args):
        stage_archive(*args)
        path = fresh / metadata
        if metadata == recovery.WORKSPACE:
            path.write_bytes(path.read_bytes() + b"\n")
        else:
            lock = json.loads(path.read_text())
            lock["archive"]["filename"] = "new-snapshot.tar.gz"
            write_json(path, lock)
        before[metadata] = path.read_bytes()

    monkeypatch.setattr(recovery, "_stage_archive", change_metadata)
    with pytest.raises(recovery.RecoveryError, match="metadata changed during restore"):
        recovery.restore(fresh, bundle=bundle)
    assert tree(fresh) == before
    no_temporaries(tmp_path)


@pytest.mark.parametrize("missing", [recovery.WORKSPACE, recovery.MANIFEST])
def test_missing_tracked_metadata_returns_nonzero(packed, tmp_path, capsys, missing):
    source, bundle, _ = packed
    fresh = fresh_checkout(source, tmp_path)
    (fresh / missing).unlink()
    before = tree(fresh)
    assert recovery.main(["--root", str(fresh), "restore", "--bundle", str(bundle)]) == 1
    assert missing in capsys.readouterr().err
    assert tree(fresh) == before


def test_output_may_not_replace_an_input_or_tracked_metadata(source):
    for path in (DATA[0], ARRAY, recovery.WORKSPACE, recovery.MANIFEST):
        with pytest.raises(recovery.RecoveryError, match="overlaps"):
            recovery.pack(source, output=source / path)
    assert not (source / recovery.MANIFEST).exists()


def test_cross_device_publication_uses_scoped_temporary_copy(packed, tmp_path, monkeypatch):
    source, bundle, _ = packed
    fresh = fresh_checkout(source, tmp_path)
    link = recovery.os.link
    copied = []
    initial_attempt = True

    def cross_device(staged, destination, **kwargs):
        nonlocal initial_attempt
        if initial_attempt:
            initial_attempt = False
            copied.append(str(destination))
            raise OSError(errno.EXDEV, "simulated different device")
        initial_attempt = True
        return link(staged, destination, **kwargs)

    monkeypatch.setattr(recovery.os, "link", cross_device)
    result = recovery.restore(fresh, bundle=bundle)
    assert len(copied) == len(result.restored)
    assert recovery.verify(fresh).ok
    no_temporaries(tmp_path)


def test_parent_swapped_for_symlink_at_publication_cannot_redirect_writes(
        packed, tmp_path, monkeypatch, capsys):
    source, bundle, _ = packed
    fresh = fresh_checkout(source, tmp_path)
    outside = tmp_path / "outside"
    write(outside / "marker", b"must be untouched")
    outside_before = tree(outside)
    link = recovery.os.link

    def swap_parent(staged, destination, **kwargs):
        if destination == "ii_graph.npz":
            directory = fresh / "model/data/graphs"
            directory.rename(tmp_path / "moved-graphs")
            directory.symlink_to(outside, target_is_directory=True)
        return link(staged, destination, **kwargs)

    monkeypatch.setattr(recovery.os, "link", swap_parent)
    assert recovery.main(["--root", str(fresh), "restore", "--bundle", str(bundle)]) == 1
    captured = capsys.readouterr()
    assert "symlink" in captured.err
    assert "Restored" not in captured.out
    assert tree(outside) == outside_before
    assert not (fresh / MISSING_WEIGHT).exists()
    no_temporaries(tmp_path)


def test_default_root_comes_from_file_not_cwd(packed, tmp_path, monkeypatch):
    source, _, _ = packed
    assert recovery.ROOT == MODULE_PATH.parents[2]
    monkeypatch.setattr(recovery, "ROOT", source)
    elsewhere = tmp_path / "unrelated-working-directory"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert recovery.load_snapshot().root == source
    assert recovery.verify().ok
    assert recovery.main(["verify"]) == 0


def test_standalone_cli_round_trip_without_site_packages(source, tmp_path):
    script = source / "model/ingredient_model/recovery.py"
    script.parent.mkdir()
    shutil.copyfile(MODULE_PATH, script)
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    bundle = tmp_path / "standalone.bin"

    def run(*args):
        return subprocess.run([sys.executable, "-I", "-S", str(script), *args],
                              cwd=cwd, capture_output=True, text=True, check=False)

    packed = run("pack", "--output", str(bundle))
    assert packed.returncode == 0, packed.stderr
    fresh = fresh_checkout(source, tmp_path)
    restored = run("--root", str(fresh), "restore", "--bundle", str(bundle))
    assert restored.returncode == 0, restored.stderr
    verified = run("--root", str(fresh), "verify")
    assert verified.returncode == 0, verified.stderr
    assert "Verified" in verified.stdout

    payload = bundle.read_bytes()
    bundle.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
    corrupt = run("--root", str(fresh), "restore", "--bundle", str(bundle))
    assert corrupt.returncode == 1
    assert "archive SHA-256 mismatch" in corrupt.stderr
    assert "Restored" not in corrupt.stdout


def test_module_cli_itself_needs_only_the_standard_library(packed, tmp_path):
    source, bundle, _ = packed
    package = source / "model/ingredient_model"
    package.mkdir()
    shutil.copyfile(MODULE_PATH, package / "recovery.py")
    (package / "__init__.py").write_text("")
    environment = dict(os.environ, PYTHONPATH=str(source / "model"))
    fresh = fresh_checkout(source, tmp_path)
    result = subprocess.run([
        sys.executable, "-S", "-m", "ingredient_model.recovery",
        "--root", str(fresh), "restore", "--bundle", str(bundle),
    ], cwd=tmp_path, env=environment, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert recovery.verify(fresh).ok
