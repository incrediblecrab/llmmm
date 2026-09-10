from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import os
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from ingredient_model import ingredient_dataset as dataset
from ingredient_model._hashing import file_sha256
from ingredient_model.data.recipe_catalog import _SCHEMA
from ingredient_model.ingredient_catalog import ARRAYS, build_ingredient_catalog, load_ingredient_catalog

REVISION = "0123456789abcdef0123456789abcdef01234567"
VOCABULARY = ["egg", "salt", "milk", "cr\u00e8me fra\u00eeche"]
ROWS = [
    ([0, 1], "fixture-a", "en", 10.25, 2.0, "https://example.test/recipe/0"),
    ([2], "fixture-a", "en", None, None, None),
    ([0, 2, 3], "fixture-b", "fr", 1.5, None, "https://example.test/recipe?id=2"),
    ([0, 1], "fixture-b", "en", None, 4.5, None),
    ([3], "fixture-b", "fr", None, None, "javascript:alert(1)"),
    ([1, 2], "fixture-a", "en", 30.0, None, "http://recipes.example.test/5"),
    ([2], "fixture-b", "en", None, None, "https://example.test/6?access_token=PRIVATE"),
]


@pytest.fixture
def ingredient_index(tmp_path, request):
    rows = getattr(request, "param", ROWS)
    corpus = tmp_path / "canonical.npz"
    flat = np.asarray([value for row in rows for value in row[0]], dtype="<u2")
    offsets = np.asarray([0, *np.cumsum([len(row[0]) for row in rows])], dtype="<u8")
    np.savez(corpus, flat=flat, offsets=offsets)
    catalog = tmp_path / "private.sqlite"
    metadata = {
        "schema_version": 1, "partial": False, "n_recipes": len(rows), "n_slots": len(flat),
        "corpus_sha256": file_sha256(corpus), "text_index_sha256": "a" * 64,
        "vocabulary": VOCABULARY,
        "ingredient_frequency": np.bincount(flat, minlength=len(VOCABULARY)).tolist(),
        "coverage": {},
    }
    with sqlite3.connect(catalog) as connection:
        connection.executescript(_SCHEMA)
        connection.executemany("INSERT INTO metadata VALUES (?,?)",
                               [(key, json.dumps(value)) for key, value in metadata.items()])
        for position, (ingredients, source, language, minutes, servings, url) in enumerate(rows):
            fields = {
                "id": position, "source": source, "language": language,
                "title": "PRIVATE TITLE", "steps": "PRIVATE INSTRUCTIONS",
                "raw_ingredients": "PRIVATE INGREDIENT PROSE",
                "ingredient_ids": np.asarray(ingredients, dtype="<u2").tobytes(),
                "url": url, "total_minutes": minutes,
                "time_status": "source_total" if minutes is not None else "unknown",
                "servings": servings,
                "servings_status": "source_servings" if servings is not None else "unknown",
                "metadata_status": "fixture", "metadata_match_count": 1,
            }
            connection.execute(
                f"INSERT INTO recipes ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})",
                list(fields.values()))
    index = tmp_path / "ingredient-index"
    build_ingredient_catalog(catalog, corpus, index, rows_per_shard=3)
    for filename in ("browser-probe.json", "private-source.txt", "urls/undeclared.json.gz"):
        (index / filename).write_bytes(b"PRIVATE EXTRA INPUT MUST NOT BE EXPORTED")
    return index


@pytest.fixture
def small_limits(monkeypatch):
    monkeypatch.setattr(dataset, "ROWS_PER_SHARD", 4)
    monkeypatch.setattr(dataset, "ROWS_PER_BATCH", 2)
    monkeypatch.setattr(dataset, "SLOTS_PER_BATCH", 3)


def _snapshot(directory):
    return {path.relative_to(directory).as_posix(): file_sha256(path)
            for path in directory.rglob("*") if path.is_file()}


def _metadata(index):
    return json.loads((index / "ingredient-index.json").read_text())


def _write_metadata(index, metadata):
    arrays, shards = metadata["arrays"].values(), metadata["url_shards"]
    metadata["bytes"].update(
        initial_compressed_download=sum(record["bytes"] for record in arrays),
        initial_uncompressed_arrays=sum(record["raw_bytes"] for record in arrays),
        url_shards_compressed=sum(record["bytes"] for record in shards),
        largest_url_shard_compressed=max(record["bytes"] for record in shards),
    )
    (index / "ingredient-index.json").write_text(json.dumps(metadata, allow_nan=False), encoding="utf-8")


def _rewrite_url_shard(index, metadata, position, raw):
    record = metadata["url_shards"][position]
    compressed = gzip.compress(raw, mtime=0)
    (index / record["file"]).write_bytes(compressed)
    record.update(bytes=len(compressed), sha256=hashlib.sha256(compressed).hexdigest(),
                  raw_bytes=len(raw), raw_sha256=hashlib.sha256(raw).hexdigest())
    _write_metadata(index, metadata)


def _assert_unpublished(output):
    assert not os.path.lexists(output)
    assert not list(output.parent.glob(".ingredient-dataset-*"))


def test_complete_bounded_parquet_and_fixed_inventory(ingredient_index, tmp_path, monkeypatch, small_limits):
    before = _snapshot(ingredient_index)
    original = _metadata(ingredient_index)
    decoded = []
    read_shard = dataset._read_url_shard

    def counted_read(directory, record):
        decoded.append(record["file"])
        return read_shard(directory, record)

    monkeypatch.setattr(dataset, "_read_url_shard", counted_read)
    output = tmp_path / "dataset"
    manifest = dataset.build_ingredient_dataset(ingredient_index, output, source_revision=REVISION)
    assert decoded == [record["file"] for record in original["url_shards"]]
    assert _snapshot(ingredient_index) == before
    assert manifest == json.loads((output / "dataset-manifest.json").read_text())
    copied, _ = load_ingredient_catalog(output / "index")
    assert {key: value for key, value in copied.items() if key != "publication"} == {
        key: value for key, value in original.items() if key != "publication"}
    assert original["publication"]["status"] == "local_export_only"
    assert copied["publication"] == {
        "status": "public_ingredient_only_release_staging",
        "scope": "public-ingredient-only-release", "source_revision": REVISION,
        "input_index_sha256": before["ingredient-index.json"],
        "maintainer_confirmed_permission": True, "private_prose_exported": False, "uploaded": False,
    }
    assert manifest["input_index_sha256"] == before["ingredient-index.json"]
    assert manifest["copied_index_sha256"] == file_sha256(output / "index/ingredient-index.json")
    assert manifest["copied_index_sha256"] != manifest["input_index_sha256"]
    assert manifest["source_revision"] == REVISION
    assert manifest["private_prose_exported"] is False
    assert manifest["uploaded"] is False
    parquet_paths = sorted((output / "data").glob("*.parquet"))
    assert [path.name for path in parquet_paths] == [
        "train-00000-of-00002.parquet", "train-00001-of-00002.parquet"]
    tables = []
    for position, path in enumerate(parquet_paths):
        with pq.ParquetFile(path) as parquet:
            assert parquet.metadata.num_rows == (4 if position == 0 else 3)
            schema = parquet.schema_arrow
            assert schema.names == [
                "id", "ingredient_ids", "ingredients", "source", "language",
                "total_minutes", "servings", "source_url"]
            assert schema.field("id").type == pa.uint32()
            assert schema.field("ingredient_ids").type.value_type == pa.uint16()
            assert schema.field("ingredients").type.value_type == pa.string()
            for name in ("source", "language", "source_url"):
                assert schema.field(name).type == pa.string()
            for name in ("total_minutes", "servings"):
                assert schema.field(name).type == pa.float64()
            for field in schema:
                assert field.nullable == (field.name in {"total_minutes", "servings", "source_url"})
            for group_id in range(parquet.metadata.num_row_groups):
                group = parquet.metadata.row_group(group_id)
                assert 1 <= group.num_rows <= 2
                assert group.column(1).num_values <= 3
                assert all(group.column(column).compression == "ZSTD" for column in range(group.num_columns))
            tables.append(parquet.read(use_threads=False))
    table = pa.concat_tables(tables)
    expected = [{
        "id": position, "ingredient_ids": row[0],
        "ingredients": [VOCABULARY[value] for value in row[0]],
        "source": row[1], "language": row[2], "total_minutes": row[3], "servings": row[4],
        "source_url": row[5] if position in (0, 2, 5) else None,
    } for position, row in enumerate(ROWS)]
    assert table.to_pylist() == expected
    assert table.column("ingredient_ids")[0].as_py() == table.column("ingredient_ids")[3].as_py()
    assert manifest["parquet"] == {
        "rows": 7, "ingredient_slots": 12,
        "null_counts": {"id": 0, "ingredient_ids": 0, "ingredients": 0, "source": 0, "language": 0,
                        "total_minutes": 4, "servings": 5, "source_url": 4},
        "split": "train", "compression": "zstd", "shards": 2,
        "max_rows_per_shard": 4, "max_rows_per_batch": 2, "max_ingredient_slots_per_batch": 3,
    }
    assert {item["name"]: item["type"] for item in manifest["output_schema"]} == {
        "id": "uint32", "ingredient_ids": "list<uint16>", "ingredients": "list<string>",
        "source": "string", "language": "string", "total_minutes": "double",
        "servings": "double", "source_url": "string",
    }
    for item in manifest["output_schema"]:
        assert item["nullable"] == table.schema.field(item["name"]).nullable
        if item["name"] in {"ingredients", "ingredient_ids"}:
            assert item["item_nullable"] is False
    expected_files = {"README.md", "index/ingredient-index.json"}
    for record in [*original["arrays"].values(), *original["url_shards"]]:
        name = "index/" + record["file"]
        expected_files.add(name)
        assert (output / name).read_bytes() == (ingredient_index / record["file"]).read_bytes()
    expected_files.update(path.relative_to(output).as_posix() for path in parquet_paths)
    assert set(manifest["files"]) == expected_files
    assert set(_snapshot(output)) == expected_files | {"dataset-manifest.json"}
    for name, record in manifest["files"].items():
        assert record == {"bytes": (output / name).stat().st_size, "sha256": file_sha256(output / name)}
        data = (output / name).read_bytes()
        if name.endswith(".gz"):
            data = gzip.decompress(data)
        assert b"PRIVATE" not in data
    card = (output / "README.md").read_text()
    frontmatter = yaml.safe_load(card.split("---", 2)[1])
    assert frontmatter["license"] == "other"
    assert frontmatter["configs"] == [
        {"config_name": "default", "data_files": [{"split": "train", "path": "data/*.parquet"}]}]
    assert f"https://github.com/incrediblecrab/llmmm/tree/{REVISION}" in card
    assert f"https://github.com/incrediblecrab/llmmm/blob/{REVISION}/raw-data/README.md" in card
    assert "https://huggingface.co/incrediblecrab/llmmm-recipes" in card
    assert "https://huggingface.co/spaces/incrediblecrab/llmmm-recipes-demo" in card
    assert "**3 of 7 records**" in card and "**2 of 7 records**" in card
    assert "**4 records have null URLs**" in card
    assert "## Use" in card and "## Acknowledgements" in card
    assert not list(output.parent.glob(".ingredient-dataset-*"))


@pytest.mark.parametrize("ingredient_index", [
    [([0], "singleton-fixture", "en", None, None, None)],
], indirect=True)
def test_singleton_with_no_numeric_facts_or_links_is_a_complete_split(ingredient_index, tmp_path):
    output = tmp_path / "dataset"
    manifest = dataset.build_ingredient_dataset(ingredient_index, output, source_revision=REVISION)
    assert _metadata(ingredient_index)["coverage"]["url_statuses"] == {"missing": 1}
    with pq.ParquetFile(output / "data/train-00000-of-00001.parquet") as parquet:
        assert parquet.read(use_threads=False).to_pylist() == [{
            "id": 0, "ingredient_ids": [0], "ingredients": ["egg"], "source": "singleton-fixture",
            "language": "en", "total_minutes": None, "servings": None, "source_url": None,
        }]
    assert manifest["parquet"]["rows"] == manifest["parquet"]["ingredient_slots"] == 1
    assert manifest["parquet"]["null_counts"] == {
        "id": 0, "ingredient_ids": 0, "ingredients": 0, "source": 0, "language": 0,
        "total_minutes": 1, "servings": 1, "source_url": 1,
    }


@pytest.mark.parametrize("damage,match", [
    ("compressed_hash", "compressed bytes"),
    ("compressed_size", "compressed bytes"),
    ("raw_hash", "decompressed bytes"),
    ("raw_size", "decompressed bytes"),
    ("first_id", "alignment"),
    ("rows", "alignment"),
    ("urls_type", "alignment"),
    ("fields", "unexpected or missing fields"),
    ("duplicate_fields", "duplicate fields"),
    ("invalid_gzip", "invalid gzip"),
])
def test_url_integrity_and_alignment_fail_atomically(
        ingredient_index, tmp_path, small_limits, damage, match):
    metadata = _metadata(ingredient_index)
    record = metadata["url_shards"][1]
    payload = json.loads(gzip.decompress((ingredient_index / record["file"]).read_bytes()))
    if damage == "compressed_hash":
        record["sha256"] = "0" * 64
    elif damage == "compressed_size":
        record["bytes"] += 1
    elif damage == "raw_hash":
        record["raw_sha256"] = "0" * 64
    elif damage == "raw_size":
        record["raw_bytes"] += 1
    elif damage == "invalid_gzip":
        compressed = b"not gzip"
        (ingredient_index / record["file"]).write_bytes(compressed)
        record.update(bytes=len(compressed), sha256=hashlib.sha256(compressed).hexdigest())
    else:
        if damage == "first_id":
            payload["first_id"] += 1
        elif damage == "rows":
            payload["urls"].pop()
        elif damage == "urls_type":
            payload["urls"] = "PRIVATE"
        elif damage == "fields":
            payload["titles"] = ["PRIVATE"] * record["rows"]
        raw = json.dumps(payload).encode()
        if damage == "duplicate_fields":
            raw = b'{"first_id":3,' + raw[1:]
        _rewrite_url_shard(ingredient_index, metadata, 1, raw)
    _write_metadata(ingredient_index, metadata)
    before = _snapshot(ingredient_index)
    output = tmp_path / "dataset"
    with pytest.raises(ValueError, match=match):
        dataset.build_ingredient_dataset(ingredient_index, output, source_revision=REVISION)
    _assert_unpublished(output)
    assert _snapshot(ingredient_index) == before


@pytest.mark.parametrize("position,value,match", [
    (0, None, "has_source_url flag"),
    (1, "https://example.test/extra", "has_source_url flag"),
    (0, "javascript:alert(1)", "unsafe or not already normalized"),
    (0, "https://example.test/r?api_key=PRIVATE", "unsafe or not already normalized"),
    (0, "https://user:PRIVATE@example.test/r", "unsafe or not already normalized"),
    (0, "https://:PRIVATE@example.test/r", "contains credentials"),
    (0, "https://@example.test/r", "contains credentials"),
    (0, "example.test/r", "unsafe or not already normalized"),
    (0, "", "unsafe or not already normalized"),
    (0, 42, "strings or null"),
])
def test_urls_are_preserved_or_rejected_never_guessed(ingredient_index, tmp_path, position, value, match):
    metadata = _metadata(ingredient_index)
    record = metadata["url_shards"][0]
    payload = json.loads(gzip.decompress((ingredient_index / record["file"]).read_bytes()))
    payload["urls"][position] = value
    _rewrite_url_shard(ingredient_index, metadata, 0, json.dumps(payload).encode())
    output = tmp_path / "dataset"
    with pytest.raises(ValueError, match=match):
        dataset.build_ingredient_dataset(ingredient_index, output, source_revision=REVISION)
    _assert_unpublished(output)


@pytest.mark.parametrize("location", [
    "index", "array", "url_record", "identity", "coverage", "bytes", "semantics", "publication",
    "included_fields", "url_statuses", "source_counts",
])
def test_unexpected_manifest_fields_cannot_be_exported(ingredient_index, tmp_path, location):
    metadata = _metadata(ingredient_index)
    if location == "included_fields":
        metadata["fields_included"].append("title")
    else:
        targets = {
            "index": metadata, "array": metadata["arrays"]["ingredients"],
            "url_record": metadata["url_shards"][0], "identity": metadata["identity"],
            "coverage": metadata["coverage"], "bytes": metadata["bytes"],
            "semantics": metadata["semantics"], "publication": metadata["publication"],
            "url_statuses": metadata["coverage"]["url_statuses"],
            "source_counts": metadata["coverage"]["rows_by_source"],
        }
        targets[location]["PRIVATE_TITLE"] = "PRIVATE"
    _write_metadata(ingredient_index, metadata)
    output = tmp_path / "dataset"
    with pytest.raises(ValueError):
        dataset.build_ingredient_dataset(ingredient_index, output, source_revision=REVISION)
    _assert_unpublished(output)


def test_duplicate_index_fields_are_rejected(ingredient_index, tmp_path):
    path = ingredient_index / "ingredient-index.json"
    path.write_bytes(b'{"schema_version":1,' + path.read_bytes()[1:])
    output = tmp_path / "dataset"
    with pytest.raises(ValueError, match="duplicate fields"):
        dataset.build_ingredient_dataset(ingredient_index, output, source_revision=REVISION)
    _assert_unpublished(output)


@pytest.mark.parametrize("damage", ["array", "frequency", "coverage", "shard_path", "shard_boundary"])
def test_existing_catalog_checks_and_statistics_are_enforced(ingredient_index, tmp_path, damage):
    metadata = _metadata(ingredient_index)
    if damage == "array":
        path = ingredient_index / ARRAYS["ingredients"][0]
        path.write_bytes(path.read_bytes() + b"corrupt")
    elif damage == "frequency":
        metadata["ingredient_frequency"][0] += 1
    elif damage == "coverage":
        metadata["coverage"]["source_servings"] += 1
    elif damage == "shard_path":
        metadata["url_shards"][0]["file"] = "../private.sqlite"
    else:
        metadata["url_shards"][0]["first_id"] += 1
    _write_metadata(ingredient_index, metadata)
    output = tmp_path / "dataset"
    with pytest.raises(ValueError):
        dataset.build_ingredient_dataset(ingredient_index, output, source_revision=REVISION)
    _assert_unpublished(output)


@pytest.mark.parametrize("filename", ["ingredient-index.json", "ingredients.u16.gz", "urls", "urls/0000.json.gz"])
def test_declared_input_symlinks_are_rejected(ingredient_index, tmp_path, filename):
    path = ingredient_index / filename
    target = path.with_name(path.name + ".original")
    path.rename(target)
    path.symlink_to(target, target_is_directory=target.is_dir())
    output = tmp_path / "dataset"
    with pytest.raises(ValueError, match="symlink"):
        dataset.build_ingredient_dataset(ingredient_index, output, source_revision=REVISION)
    _assert_unpublished(output)


@pytest.mark.parametrize("revision", ["", "main", "a" * 39, "a" * 41, "g" * 40, "a" * 40 + "\n", None, 42])
def test_exact_source_revision_is_required(ingredient_index, tmp_path, revision):
    output = tmp_path / "dataset"
    with pytest.raises(ValueError, match="exact 40-character"):
        dataset.build_ingredient_dataset(ingredient_index, output, source_revision=revision)
    _assert_unpublished(output)


@pytest.mark.parametrize("kind", ["directory", "file", "directory_symlink", "file_symlink", "dangling_symlink"])
def test_existing_outputs_are_never_overwritten(ingredient_index, tmp_path, kind):
    output = tmp_path / "dataset"
    target = tmp_path / "target"
    if kind == "directory":
        output.mkdir()
    elif kind == "file":
        output.write_text("keep")
    else:
        if kind == "directory_symlink":
            target.mkdir()
        elif kind == "file_symlink":
            target.write_text("keep")
        output.symlink_to(target, target_is_directory=kind == "directory_symlink")
    with pytest.raises(FileExistsError):
        dataset.build_ingredient_dataset(ingredient_index, output, source_revision=REVISION)
    assert os.path.lexists(output)
    if kind.endswith("symlink"):
        assert output.is_symlink() and output.readlink() == target
    elif kind == "file":
        assert output.read_text() == "keep"
    assert not list(tmp_path.glob(".ingredient-dataset-*"))


def test_output_cannot_mutate_the_input_tree(ingredient_index, tmp_path):
    before = _snapshot(ingredient_index)
    alias = tmp_path / "input-alias"
    alias.symlink_to(ingredient_index, target_is_directory=True)
    for output in (ingredient_index / "release", alias / "release"):
        with pytest.raises(ValueError, match="outside the immutable input"):
            dataset.build_ingredient_dataset(ingredient_index, output, source_revision=REVISION)
        _assert_unpublished(output)
    assert _snapshot(ingredient_index) == before


def test_atomic_publish_does_not_replace_a_competing_directory(ingredient_index, tmp_path, monkeypatch):
    publish = dataset._publish_directory
    output = tmp_path / "dataset"

    def race(staged, destination):
        destination.mkdir()
        publish(staged, destination)

    monkeypatch.setattr(dataset, "_publish_directory", race)
    with pytest.raises(FileExistsError):
        dataset.build_ingredient_dataset(ingredient_index, output, source_revision=REVISION)
    assert output.is_dir() and not list(output.iterdir())
    assert not list(tmp_path.glob(".ingredient-dataset-*"))


@pytest.fixture
def cli(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/build_ingredient_dataset.py"
    spec = importlib.util.spec_from_file_location("ingredient_dataset_cli_fixture", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True, capture_output=True)
    (repository / ".gitignore").write_text("/ignored/\n*.parquet\n/marker-only/README.md\n")
    monkeypatch.setattr(module, "ROOT", repository)
    return module


def test_cli_builds_only_in_a_git_ignored_directory(cli, ingredient_index, monkeypatch, capsys):
    output = cli.ROOT / "ignored/release"
    monkeypatch.setattr("sys.argv", [
        "builder", "--index", str(ingredient_index), "--out", str(output), "--source-revision", REVISION])
    cli.main()
    printed = json.loads(capsys.readouterr().out)
    assert printed == json.loads((output / "dataset-manifest.json").read_text())
    assert printed["uploaded"] is False
    assert printed["parquet"]["rows"] == len(ROWS)
    assert pa.cpu_count() <= 4 and pa.io_thread_count() <= 4


@pytest.mark.parametrize("destination", ["not-ignored/release", "marker-only"])
def test_cli_requires_the_entire_directory_to_be_ignored(cli, ingredient_index, monkeypatch, capsys, destination):
    output = cli.ROOT / destination
    monkeypatch.setattr("sys.argv", [
        "builder", "--index", str(ingredient_index), "--out", str(output), "--source-revision", REVISION])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "entire output directory must be Git-ignored" in capsys.readouterr().err
    _assert_unpublished(output)


def test_cli_refuses_dangling_output_symlinks_before_resolving(cli, ingredient_index, monkeypatch):
    output = cli.ROOT / "ignored"
    target = cli.ROOT / "absent"
    output.symlink_to(target)
    monkeypatch.setattr("sys.argv", [
        "builder", "--index", str(ingredient_index), "--out", str(output), "--source-revision", REVISION])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert output.is_symlink() and not target.exists()


@pytest.mark.parametrize("arguments", [
    [], ["--index", "fixture", "--out", "fixture"],
    ["--out", "fixture", "--source-revision", REVISION],
    ["--index", "fixture", "--source-revision", REVISION],
    ["--index", "fixture", "--out", "fixture", "--source-revision", REVISION, "--upload"],
])
def test_cli_requires_all_inputs_and_has_no_upload_option(cli, monkeypatch, arguments):
    monkeypatch.setattr("sys.argv", ["builder", *arguments])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
