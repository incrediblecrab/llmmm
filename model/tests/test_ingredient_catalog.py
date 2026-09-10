from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3

import numpy as np
import pytest

from ingredient_model.data.recipe_catalog import _SCHEMA
from ingredient_model.ingredient_catalog import (
    ARRAYS, build_ingredient_catalog, load_ingredient_catalog, public_source_url,
)


@pytest.fixture
def canonical_catalog(tmp_path):
    corpus = tmp_path / "canonical.npz"
    flat = np.asarray([0, 1, 0, 2, 1], dtype="<u2")
    offsets = np.asarray([0, 2, 4, 5], dtype="<u8")
    np.savez(corpus, flat=flat, offsets=offsets)
    catalog = tmp_path / "catalog.sqlite"
    metadata = {
        "schema_version": 1, "partial": False, "n_recipes": 3, "n_slots": 5,
        "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "text_index_sha256": "a" * 64, "vocabulary": ["egg", "salt", "milk"],
        "ingredient_frequency": [2, 2, 1], "coverage": {},
    }
    with sqlite3.connect(catalog) as connection:
        connection.executescript(_SCHEMA)
        connection.executemany("INSERT INTO metadata VALUES (?,?)",
                               [(key, json.dumps(value)) for key, value in metadata.items()])
        for position in range(3):
            fields = {
                "id": position, "source": "fixture", "language": "en",
                "title": "PRIVATE TITLE MUST NEVER BE EXPORTED",
                "steps": "PRIVATE INSTRUCTIONS MUST NEVER BE EXPORTED" if position < 2 else "",
                "raw_ingredients": "PRIVATE INGREDIENT PROSE MUST NEVER BE EXPORTED",
                "ingredient_ids": flat[offsets[position]:offsets[position + 1]].tobytes(),
                "url": f"https://example.test/recipe/{position}",
                "total_minutes": 10.25 if position == 0 else None,
                "time_status": "source_total" if position == 0 else "unknown",
                "servings": 2 if position == 0 else None,
                "servings_status": "source_servings" if position == 0 else "unknown",
                "metadata_status": "fixture", "metadata_match_count": 1,
            }
            connection.execute(
                f"INSERT INTO recipes ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})",
                list(fields.values()))
    return catalog, corpus


def test_all_rows_roundtrip_without_copying_prose(canonical_catalog, tmp_path):
    catalog, corpus = canonical_catalog
    output = tmp_path / "ingredient-only"
    report = build_ingredient_catalog(catalog, corpus, output, rows_per_shard=2)
    metadata, arrays = load_ingredient_catalog(output)
    assert report["coverage"]["records"] == 3
    assert report["coverage"]["singletons"] == 1
    assert report["coverage"]["source_total_times"] == 1
    assert report["all_rows_compared_with_canonical_arrays"] is True
    assert report["uploaded"] is False
    assert metadata["publication"]["status"] == "local_export_only"
    assert arrays["ingredients"].tolist() == [0, 1, 0, 2, 1]
    assert arrays["lengths"].tolist() == [2, 2, 1]
    assert arrays["total_minutes"][0] == 10.25
    assert np.isnan(arrays["total_minutes"][1:]).all()
    assert metadata["ingredient_frequency"] == [2, 2, 1]
    assert len(metadata["url_shards"]) == 2
    for path in output.rglob("*"):
        if path.is_file():
            data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
            assert b"PRIVATE" not in data
    first = json.loads(gzip.decompress((output / metadata["url_shards"][0]["file"]).read_bytes()))
    assert first == {"first_id": 0, "urls": ["https://example.test/recipe/0", "https://example.test/recipe/1"]}
    with pytest.raises(FileExistsError):
        build_ingredient_catalog(catalog, corpus, output)


def test_canonical_mismatch_never_publishes_partial_output(canonical_catalog, tmp_path):
    catalog, corpus = canonical_catalog
    with sqlite3.connect(catalog) as connection:
        connection.execute("UPDATE recipes SET ingredient_ids=? WHERE id=1",
                           (np.asarray([0, 1], dtype="<u2").tobytes(),))
    output = tmp_path / "mismatch"
    with pytest.raises(ValueError, match="differ from the canonical"):
        build_ingredient_catalog(catalog, corpus, output)
    assert not output.exists()


@pytest.mark.parametrize("value,status", [
    ("https://example.test/r/1?access_token=secret", "sensitive_query_not_exported"),
    ("https://person:password@example.test/r/1", "invalid_or_oversized"),
    ("javascript:alert(1)", "invalid_or_oversized"),
    ("", "missing"),
    (None, "missing"),
])
def test_invalid_or_sensitive_urls_are_omitted_with_counted_reasons(value, status):
    assert public_source_url(value) == (None, status)


def test_fact_only_urls_are_preserved_without_guessing():
    assert public_source_url("https://example.test/recipe?id=12") == (
        "https://example.test/recipe?id=12", "source_url")


def test_missing_or_invalid_links_do_not_remove_ingredient_records(canonical_catalog, tmp_path):
    catalog, corpus = canonical_catalog
    with sqlite3.connect(catalog) as connection:
        connection.execute("UPDATE recipes SET url=NULL WHERE id=1")
        connection.execute("UPDATE recipes SET url='javascript:alert(1)' WHERE id=2")
    output = tmp_path / "ingredient-only"
    report = build_ingredient_catalog(catalog, corpus, output)
    metadata, arrays = load_ingredient_catalog(output)
    assert metadata["n_recipes"] == 3
    assert arrays["has_source_url"].tolist() == [1, 0, 0]
    assert report["coverage"]["url_statuses"] == {
        "source_url": 1, "missing": 1, "invalid_or_oversized": 1,
    }


def test_array_corruption_is_detected(canonical_catalog, tmp_path):
    catalog, corpus = canonical_catalog
    output = tmp_path / "ingredient-only"
    build_ingredient_catalog(catalog, corpus, output)
    path = output / ARRAYS["ingredients"][0]
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="integrity"):
        load_ingredient_catalog(output)


@pytest.mark.parametrize("patch", [
    {"file": "../private.env"},
    {"first_id": 1},
    {"rows": 2},
])
def test_url_shards_cannot_escape_the_index_or_misalign_records(canonical_catalog, tmp_path, patch):
    catalog, corpus = canonical_catalog
    output = tmp_path / "ingredient-only"
    build_ingredient_catalog(catalog, corpus, output)
    path = output / "ingredient-index.json"
    metadata = json.loads(path.read_text())
    metadata["url_shards"][0].update(patch)
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="shard"):
        load_ingredient_catalog(output)
