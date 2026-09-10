from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import stat
from pathlib import Path

import numpy as np
import pytest

from ingredient_model._hashing import file_sha256
from ingredient_model.data import recipe_search_metadata as module
from ingredient_model.data.recipe_search_metadata import (
    RecipeSearchMetadata,
    SearchMetadataError,
    build_search_metadata,
    default_search_metadata_path,
)


@pytest.fixture(autouse=True)
def reset_hash_cache():
    module._HASH_CACHE.clear()
    yield
    module._HASH_CACHE.clear()


def _catalog(root):
    path = root / "catalog.sqlite"
    records = [
        ("en", [0], "One", "Cook.", "", 10, "source_total", 4, "source_servings"),
        ("en", [1, 2], "Unknown", "Mix.", "", None, "derived_only", None, "missing"),
        ("zh", list(range(98)), "Long", "Cook.", "", 30, "source_total", 2, "source_servings"),
        ("en", [3], "Separator", " \x1f\t\r\n", "", 5, "source_total", 4, "source_servings"),
        ("en", [4], "Unicode", "\u00a0\x1f\u2003\u3000", "", 5, "source_total", 4, "source_servings"),
        ("en", [5], "\u00a0", "Cook.", "", 5, "source_total", 4, "source_servings"),
        ("en", [6], "Malformed", "Unparsed instructions.", "unparsed_steps", 5,
         "source_total", 4, "source_servings"),
        ("fr", [7, 8], "Later", " \x1fMix.\x1f\u00a0", "", 60, "source_total", 6, "source_servings"),
        ("en", [9], "Component", "Mix.", "", None, "component_only", 2, "source_servings"),
        ("en", [10], "NUL", "\x00", "", 20, "source_total", 3, "source_servings"),
    ]
    frequencies = [0] * 100
    for _, ids, *_ in records:
        for ingredient in ids:
            frequencies[ingredient] += 1
    metadata = {
        "schema_version": 1, "partial": False, "corpus_sha256": "a" * 64,
        "text_index_sha256": "b" * 64, "n_recipes": len(records),
        "n_slots": sum(frequencies), "vocabulary": [f"v{i}" for i in range(100)],
        "ingredient_frequency": frequencies,
        "coverage": {"by_source": {"fixture": {"n_recipes": len(records)}}},
    }
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE recipes(
            id INTEGER PRIMARY KEY,source TEXT,language TEXT,ingredient_ids BLOB,title TEXT,
            steps TEXT,text_status TEXT,total_minutes REAL,time_status TEXT,
            servings REAL,servings_status TEXT,prep_minutes REAL,cook_minutes REAL,
            derived_total_minutes REAL);
    """)
    connection.executemany("INSERT INTO metadata VALUES (?,?)", [
        (key, json.dumps(value)) for key, value in metadata.items()])
    for index, (language, ids, title, steps, text_status, total, time_status,
                servings, servings_status) in enumerate(records):
        connection.execute("INSERT INTO recipes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            index, "fixture", language, np.asarray(ids, dtype="<u2").tobytes(), title,
            steps, text_status, total, time_status, servings, servings_status, 5, 5, 10))
    connection.commit()
    connection.close()
    return path


def _change(catalog, query, parameters=()):
    connection = sqlite3.connect(catalog)
    connection.execute(query, parameters)
    connection.commit()
    connection.close()


def _manifest(path):
    return json.loads((path / module.MANIFEST_FILE).read_text())


def _write_manifest(path, manifest):
    (path / module.MANIFEST_FILE).write_text(json.dumps(manifest))


def _resign(path, name):
    manifest = _manifest(path)
    manifest["arrays"][name]["sha256"] = file_sha256(path / name)
    manifest["arrays"][name]["bytes"] = (path / name).stat().st_size
    _write_manifest(path, manifest)


def _assert_unpublished(catalog, output):
    assert not output.exists()
    assert not list(output.parent.glob(f".{output.name}.*.building"))
    assert not Path(str(catalog) + "-journal").exists()


def test_complete_immutable_index_and_stable_filtering(tmp_path):
    catalog = _catalog(tmp_path)
    before = file_sha256(catalog)
    report = build_search_metadata(catalog)
    path = default_search_metadata_path(catalog)
    assert path.name == "catalog.sqlite.metadata"
    assert file_sha256(catalog) == before
    assert report["verified"] and report["catalog"]["sha256"] == before
    assert report["catalog"]["n_recipes"] == 10
    assert report["coverage"]["readable"] == 6
    assert str(tmp_path) not in json.dumps(report)
    assert {entry.name for entry in path.iterdir()} == {*module.ARRAY_DTYPES, "manifest.json"}
    assert all(stat.S_IMODE(entry.stat().st_mode) == 0o600 for entry in path.iterdir())
    index = RecipeSearchMetadata.load(catalog)
    try:
        assert index.n_recipes == 10
        assert np.isnan(index.total_minutes[[1, 8]]).all()
        assert np.isnan(index.servings[1])
        assert index.ingredient_counts.tolist() == [1, 2, 98, 1, 1, 1, 1, 2, 1, 1]
        assert index.readable.tolist() == [True, True, True, False, False, False, False, True, True, True]
        assert index.language_names.tolist() == ["en", "zh", "fr"]
        for name in module.ARRAY_DTYPES:
            array = getattr(index, name.removesuffix(".npy"))
            assert isinstance(array, np.memmap) and not array.flags.writeable
        with pytest.raises(ValueError):
            index.total_minutes[0] = 0
        ids = [7, 1, 0, 2, 8, 0, 9, 6, 4, 3]
        assert index.filter_ids(ids) == [7, 1, 0, 2, 8, 0, 9]
        assert index.filter_ids(ids, max_total_minutes=30) == [0, 2, 0, 9]
        assert index.filter_ids(ids, min_servings=4) == [7, 0, 0]
        assert index.filter_ids(ids, language="en", max_ingredients=1) == [0, 8, 0, 9]
        assert index.filter_ids(ids, max_total_minutes=15, min_servings=4, max_ingredients=1) == [0, 0]
        assert index.filter_ids(np.array(ids, dtype=np.uint64), language="xx") == []
        assert index.filter_ids([]) == []
    finally:
        index.close()
    with pytest.raises(SearchMetadataError, match="closed"):
        index.filter_ids([0])
    assert file_sha256(catalog) == before


@pytest.mark.parametrize("ids", [[-1], [10], [True], ["0"], [0.0], [[0]], [2**100],
                                 "0", np.array([0.0]), np.array([[0]]),
                                 np.array([2**64 - 1], dtype=np.uint64)])
def test_invalid_ids_fail_closed(tmp_path, ids):
    catalog = _catalog(tmp_path)
    build_search_metadata(catalog)
    index = RecipeSearchMetadata.load(catalog)
    try:
        with pytest.raises(ValueError):
            index.filter_ids(ids)
    finally:
        index.close()


@pytest.mark.parametrize("constraint", [
    {"max_total_minutes": -1}, {"max_total_minutes": float("inf")},
    {"max_total_minutes": float("nan")}, {"max_total_minutes": True},
    {"min_servings": 0}, {"min_servings": "4"}, {"min_servings": -1},
    {"max_ingredients": 0}, {"max_ingredients": 1.5}, {"max_ingredients": True},
    {"language": ""}, {"language": ["en"]}, {"language": "x" * 33},
])
def test_invalid_constraints_fail_closed_even_for_empty_ids(tmp_path, constraint):
    catalog = _catalog(tmp_path)
    build_search_metadata(catalog)
    index = RecipeSearchMetadata.load(catalog)
    try:
        with pytest.raises(ValueError):
            index.filter_ids([], **constraint)
    finally:
        index.close()


@pytest.mark.parametrize("budget", [0, 0.0, -0.0, np.float64(0)])
def test_zero_time_budget_matches_sql_without_relaxing_stored_totals(tmp_path, budget):
    catalog = _catalog(tmp_path)
    before = file_sha256(catalog)
    build_search_metadata(catalog)
    index = RecipeSearchMetadata.load(catalog)
    try:
        ids = [7, 0, 1, 2, 9, 0]
        with sqlite3.connect(catalog) as connection:
            expected = connection.execute(
                "SELECT id FROM recipes WHERE total_minutes>0 "
                "AND time_status='source_total' AND total_minutes<=?", (float(budget),)).fetchall()
        assert expected == []
        assert index.filter_ids(ids, max_total_minutes=budget) == []
        assert index.filter_ids([], max_total_minutes=budget) == []
        assert index.filter_ids(ids, max_total_minutes=None)
        known = index.total_minutes[~np.isnan(index.total_minutes)]
        assert np.isfinite(known).all() and (known > 0).all()
        with pytest.raises(ValueError, match="positive"):
            index.filter_ids(ids, max_total_minutes=budget, min_servings=0)
    finally:
        index.close()
    assert file_sha256(catalog) == before


@pytest.mark.parametrize(("column", "value"), [
    ("total_minutes", 0), ("total_minutes", -1), ("total_minutes", float("inf")),
    ("time_status", "derived_only"), ("time_status", "unknown"), ("total_minutes", None),
    ("servings", 0), ("servings", -1), ("servings", float("inf")),
    ("servings_status", "missing"), ("language", ""), ("language", "x" * 33),
    ("ingredient_ids", b""), ("ingredient_ids", b"\x00"),
    ("ingredient_ids", b"\x00\x00\x00\x00"), ("ingredient_ids", b"\x64\x00"),
    ("ingredient_ids", b"\x02\x00\x01\x00"),
])
def test_invalid_catalog_records_never_publish(tmp_path, column, value):
    catalog = _catalog(tmp_path)
    _change(catalog, f"UPDATE recipes SET {column}=? WHERE id=0", (value,))
    output = default_search_metadata_path(catalog)
    before = file_sha256(catalog)
    with pytest.raises(SearchMetadataError):
        build_search_metadata(catalog)
    assert file_sha256(catalog) == before
    _assert_unpublished(catalog, output)


@pytest.mark.parametrize("damage", ["partial", "gap", "frequency", "source_count"])
def test_metadata_alignment_and_completeness_are_required(tmp_path, damage):
    catalog = _catalog(tmp_path)
    if damage == "partial":
        _change(catalog, "UPDATE metadata SET value='true' WHERE key='partial'")
    elif damage == "gap":
        _change(catalog, "UPDATE recipes SET id=10 WHERE id=9")
    elif damage == "source_count":
        value = {"by_source": {"fixture": {"n_recipes": 9}}}
        _change(catalog, "UPDATE metadata SET value=? WHERE key='coverage'", (json.dumps(value),))
    else:
        connection = sqlite3.connect(catalog)
        frequency = json.loads(connection.execute(
            "SELECT value FROM metadata WHERE key='ingredient_frequency'").fetchone()[0])
        connection.close()
        frequency[0] -= 1
        frequency[1] += 1
        _change(catalog, "UPDATE metadata SET value=? WHERE key='ingredient_frequency'",
                (json.dumps(frequency),))
    with pytest.raises(SearchMetadataError):
        build_search_metadata(catalog)
    _assert_unpublished(catalog, default_search_metadata_path(catalog))


def test_missing_or_corrupt_index_never_falls_back(tmp_path):
    catalog = _catalog(tmp_path)
    with pytest.raises(FileNotFoundError):
        RecipeSearchMetadata.load(catalog)
    build_search_metadata(catalog)
    path = default_search_metadata_path(catalog)
    (path / "readable.npy").unlink()
    with pytest.raises(SearchMetadataError, match="missing"):
        RecipeSearchMetadata.load(catalog)


@pytest.mark.parametrize("damage", ["partial", "catalog", "path", "descriptor_path", "duplicate", "oversized"])
def test_untrusted_manifest_is_bounded_and_has_no_arbitrary_paths(tmp_path, damage):
    catalog = _catalog(tmp_path)
    build_search_metadata(catalog)
    path = default_search_metadata_path(catalog)
    manifest = _manifest(path)
    if damage == "partial":
        manifest["partial"] = True
    elif damage == "catalog":
        manifest["catalog"]["corpus_sha256"] = "c" * 64
    elif damage == "path":
        manifest["arrays"]["../../outside.npy"] = manifest["arrays"].pop("readable.npy")
    elif damage == "descriptor_path":
        manifest["arrays"]["readable.npy"]["path"] = "../../outside.npy"
    elif damage == "duplicate":
        original = json.dumps(manifest)
        (path / "manifest.json").write_text('{"format_version":1,' + original[1:])
    else:
        (path / "manifest.json").write_text(" " * (64 * 1024 + 1))
    if damage not in {"duplicate", "oversized"}:
        _write_manifest(path, manifest)
    with pytest.raises(SearchMetadataError):
        RecipeSearchMetadata.load(catalog)
    assert not (tmp_path / "outside.npy").exists()


@pytest.mark.parametrize("damage", ["checksum", "shape", "dtype", "object", "infinite",
                                   "zero", "language", "boolean"])
def test_arrays_are_hashed_typed_and_semantically_validated(tmp_path, damage):
    catalog = _catalog(tmp_path)
    build_search_metadata(catalog)
    path = default_search_metadata_path(catalog)
    name = "total_minutes.npy"
    if damage == "checksum":
        array = np.load(path / name, mmap_mode="r+")
        array[0] = 99
        array.flush()
        array._mmap.close()
    elif damage in {"shape", "dtype", "object"}:
        values = (np.zeros(9) if damage == "shape" else
                  np.zeros(10, dtype=np.float32 if damage == "dtype" else object))
        np.save(path / name, values)
        _resign(path, name)
    else:
        if damage == "language":
            name = "language_codes.npy"
            value = 256
        elif damage == "boolean":
            name = "readable.npy"
            value = 2
        else:
            value = float("inf") if damage == "infinite" else 0
        array = np.load(path / name, mmap_mode="r+")
        if damage == "boolean":
            array.view(np.uint8)[0] = value
        else:
            array[0] = value
        array.flush()
        array._mmap.close()
        _resign(path, name)
    with pytest.raises(SearchMetadataError):
        RecipeSearchMetadata.load(catalog)


def test_cache_symlinks_are_rejected(tmp_path):
    catalog = _catalog(tmp_path)
    build_search_metadata(catalog)
    path = default_search_metadata_path(catalog)
    outside = tmp_path / "outside.npy"
    (path / "readable.npy").rename(outside)
    (path / "readable.npy").symlink_to(outside)
    with pytest.raises(SearchMetadataError, match="symlinks"):
        RecipeSearchMetadata.load(catalog)


def test_catalog_hash_cache_reuses_only_unchanged_file_identity(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path)
    build_search_metadata(catalog)
    module._HASH_CACHE.clear()
    calls = []
    original = module.file_sha256

    def tracked(path):
        calls.append(Path(path).resolve())
        return original(path)

    monkeypatch.setattr(module, "file_sha256", tracked)
    first = RecipeSearchMetadata.load(catalog)
    second = RecipeSearchMetadata.load(catalog)
    assert calls.count(catalog.resolve()) == 1
    mode = stat.S_IMODE(catalog.stat().st_mode)
    os.chmod(catalog, mode ^ stat.S_IXUSR)
    with pytest.raises(SearchMetadataError, match="changed"):
        first.filter_ids([0])
    third = RecipeSearchMetadata.load(catalog)
    assert calls.count(catalog.resolve()) == 2
    for index in (first, second, third):
        index.close()
    _change(catalog, "UPDATE recipes SET title='Changed' WHERE id=0")
    with pytest.raises(SearchMetadataError, match="exact catalog"):
        RecipeSearchMetadata.load(catalog)


def test_loaded_index_detects_array_changes(tmp_path):
    catalog = _catalog(tmp_path)
    build_search_metadata(catalog)
    index = RecipeSearchMetadata.load(catalog)
    array = np.load(index.path / "readable.npy", mmap_mode="r+")
    array[0] = False
    array.flush()
    array._mmap.close()
    try:
        with pytest.raises(SearchMetadataError, match="changed"):
            index.filter_ids([0])
    finally:
        index.close()


def test_existing_outputs_and_atomic_publish_race_are_preserved(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path)
    output = default_search_metadata_path(catalog)
    output.mkdir()
    with pytest.raises(FileExistsError):
        build_search_metadata(catalog)
    output.rmdir()
    original = module._publish_directory

    def race(staged, destination):
        destination.mkdir()
        original(staged, destination)

    monkeypatch.setattr(module, "_publish_directory", race)
    with pytest.raises(FileExistsError):
        build_search_metadata(catalog)
    assert output.is_dir() and list(output.iterdir()) == []
    assert not list(tmp_path.glob(f".{output.name}.*.building"))


def test_interrupted_verification_leaves_no_final_index(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path)
    before = file_sha256(catalog)

    def interrupted(*_args, **_kwargs):
        raise RuntimeError("interrupted verification")

    monkeypatch.setattr(RecipeSearchMetadata, "load", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        build_search_metadata(catalog)
    _assert_unpublished(catalog, default_search_metadata_path(catalog))
    assert file_sha256(catalog) == before


def test_sqlite_sidecars_are_not_hidden_by_main_file_hash(tmp_path):
    catalog = _catalog(tmp_path)
    sidecar = Path(str(catalog) + "-wal")
    sidecar.write_bytes(b"not part of main file")
    with pytest.raises(SearchMetadataError, match="sidecar"):
        build_search_metadata(catalog)
    assert not default_search_metadata_path(catalog).exists()


def _cli():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_recipe_catalog.py"
    spec = importlib.util.spec_from_file_location("metadata_build_cli_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_metadata_only_never_rebuilds_catalog(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path)
    before = file_sha256(catalog)
    cli = _cli()

    def forbidden(*_args, **_kwargs):
        pytest.fail("metadata-only mode must not rebuild SQLite")

    monkeypatch.setattr(cli, "build_catalog", forbidden)
    monkeypatch.setattr("sys.argv", ["builder", "--metadata-only", str(catalog)])
    assert cli.main() == 0
    assert default_search_metadata_path(catalog).is_dir()
    assert file_sha256(catalog) == before
    with pytest.raises(SystemExit):
        cli.main()


def test_cli_full_build_includes_metadata_index(tmp_path, monkeypatch):
    cli = _cli()
    output = tmp_path / "fresh.sqlite"
    report_path = tmp_path / "report.json"
    calls = []

    def full_build(path, **_kwargs):
        assert path == output
        calls.append("catalog")
        fixture = _catalog(tmp_path)
        fixture.rename(output)
        return {"catalog_fixture": True}

    real_metadata = cli.build_search_metadata

    def metadata_build(path, destination):
        calls.append("metadata")
        return real_metadata(path, destination)

    monkeypatch.setattr(cli, "build_catalog", full_build)
    monkeypatch.setattr(cli, "build_search_metadata", metadata_build)
    monkeypatch.setattr("sys.argv", ["builder", "--out", str(output), "--report", str(report_path)])
    assert cli.main() == 0
    assert calls == ["catalog", "metadata"]
    report = json.loads(report_path.read_text())
    assert report["search_metadata"]["verified"]
    assert default_search_metadata_path(output).is_dir()
