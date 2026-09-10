from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import stat
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ingredient_model.config import Paths
from ingredient_model.data import recipe_catalog as catalog
from ingredient_model.data import text as recipe_text
from ingredient_model.data.recipe_catalog import (
    ParsedNumber,
    SourceMetadata,
    build_catalog,
    has_usable_steps,
    load_catalog_metadata,
    parse_duration,
    parse_servings,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(root, baskets=None, sources=None, text_rows=None, vocabulary_size=100):
    baskets = [[0], [1, 2], list(range(98))] if baskets is None else baskets
    sources = ["fixture"] * len(baskets) if sources is None else sources
    paths = Paths(root / "data", root / "results", root / "raw", root / "prior")
    paths.recipes.mkdir(parents=True)
    flat = np.array([item for basket in baskets for item in basket], dtype=np.uint16)
    offsets = np.r_[0, np.cumsum([len(basket) for basket in baskets])].astype(np.int64)
    np.savez(
        paths.recipes / "recipe_ids.npz", flat=flat, offsets=offsets,
        source=np.array(sources), lang=np.array(["en"] * len(baskets)),
        itos=np.array([f"v{index}" for index in range(vocabulary_size)]))
    generation = {
        "sha256": _sha(paths.recipes / "recipe_ids.npz"), "recipes": len(baskets),
        "slots": len(flat), "vocab": vocabulary_size,
    }
    paths.generation_file.write_text(json.dumps(generation), encoding="utf-8")
    if text_rows is None:
        text_rows = [{
            "idx": index, "source": source, "title": f"PRIVATE_TITLE_{index}",
            "url": f"https://fixture.invalid/{index}",
            "raw_ingredients": "\x1f".join(f"v{item}" for item in basket),
            "steps": f"PRIVATE_STEP_{index}\x1fSecond step.",
            "ingredient_quantities": "[]", "quantity_status": "",
            "text_status": "",
        } for index, (source, basket) in enumerate(zip(sources, baskets, strict=True))]
    schema = pa.schema(
        [("idx", pa.int32()), *((name, pa.string()) for name in catalog.TEXT_COLUMNS)],
        metadata={
            "text_schema_version": "2", "corpus_sha256": generation["sha256"],
            "reader_sha256": _sha(Path(recipe_text.__file__)), "partial": "false",
        })
    pq.write_table(
        pa.Table.from_pylist(text_rows, schema=schema), paths.recipes / recipe_text.TEXT_FILE)
    return paths, text_rows


def _record(text, ids, **kwargs):
    return SourceMetadata(
        source=text["source"], text=text, ingredient_ids=tuple(ids), **kwargs)


def _build(paths, **kwargs):
    kwargs.setdefault("metadata_records", ())
    kwargs.setdefault("batch_size", 2)
    return build_catalog(paths.recipes / catalog.CATALOG_FILE, paths=paths, **kwargs)


def _rewrite_text(paths, transform):
    path = paths.recipes / recipe_text.TEXT_FILE
    table = pq.read_table(path)
    pq.write_table(transform(table), path)


def _assert_no_catalog(paths):
    assert not (paths.recipes / catalog.CATALOG_FILE).exists()
    assert not list(paths.recipes.glob("*.building*"))
    assert not list(paths.recipes.glob(".*.building*"))


@pytest.mark.parametrize(("value", "encoding", "expected"), [
    ("PT15M", "iso8601", 15),
    ("PT1H30M", "iso8601", 90),
    ("P1DT2H3M30S", "iso8601", 1563.5),
    ("P2W", "iso8601", 20160),
    ("PT0.5H", "iso8601", 30),
    ("PT0,5H", "iso8601", 30),
    ("PT30S", "iso8601", 0.5),
    (12, "minutes", 12),
    (np.int64(12), "minutes", 12),
    (12.5, "minutes", 12.5),
    (" 12.5 ", "minutes", 12.5),
    ("1e2", "minutes", 100),
    ("1 day 2 hrs 3 mins 30 secs", "en", 1563.5),
    ("1.5 hours", "en", 90),
    ("60 λεπτά", "el", 60),
    ("1 ώρα 20 λεπτά", "el", 80),
    ("1時間30分", "ja", 90),
    ("60 минут", "ru", 60),
    ("1 小時 30 分鐘", "zh", 90),
])
def test_duration_accepts_only_declared_units(value, encoding, expected):
    parsed = parse_duration(value, encoding=encoding)
    assert parsed.status == "provided"
    assert parsed.value == pytest.approx(expected)


@pytest.mark.parametrize("value", [None, "", "  ", "NA", "N/A", "null"])
def test_missing_duration_is_unknown_never_zero(value):
    assert parse_duration(value, encoding="iso8601") == ParsedNumber()
    assert parse_duration(value, encoding="minutes") == ParsedNumber()


@pytest.mark.parametrize(("value", "encoding"), [
    ("P", "iso8601"), ("PT", "iso8601"), ("P1DT", "iso8601"),
    ("P1Y", "iso8601"), ("P1M", "iso8601"), ("P1W2D", "iso8601"),
    ("PT-1M", "iso8601"), ("-PT1M", "iso8601"), ("PT1M2H", "iso8601"),
    ("PT1.5H30M", "iso8601"), ("P1.5DT1H", "iso8601"), ("PT1M junk", "iso8601"),
    ("pt10m", "iso8601"), ("PTNaNM", "iso8601"),
    ("PT" + "9" * 400 + "M", "iso8601"),
    (-1, "minutes"), ("-1", "minutes"), (float("nan"), "minutes"),
    (float("inf"), "minutes"), (-float("inf"), "minutes"), ("NaN", "minutes"),
    ("Infinity", "minutes"), ("1e309", "minutes"), (True, "minutes"),
    ({"minutes": 12}, "minutes"), ([12], "minutes"),
    ("ready in 10 minutes", "en"), ("about 10 mins", "en"),
    ("10-20 minutes", "en"), ("10 minutes plus resting", "en"),
    ("1 hour and 20 minutes", "en"), ("10 mins 1 hr", "en"),
    ("10 mins 20 mins", "en"), ("20", "en"), (20, "en"),
    ("20 minutes", "minutes"), ("2 servings", "minutes"),
    (0, "minutes"), ("PT0S", "iso8601"),
])
def test_invalid_duration_stays_unknown(value, encoding):
    assert parse_duration(value, encoding=encoding) == ParsedNumber(status="invalid")


def test_explicit_zero_is_allowed_for_components_only():
    assert parse_duration("PT0S", encoding="iso8601", allow_zero=True) == ParsedNumber(0, "provided")
    assert parse_duration(0, encoding="minutes", allow_zero=True) == ParsedNumber(0, "provided")
    with pytest.raises(ValueError, match="encoding"):
        parse_duration("20", encoding="guess")


@pytest.mark.parametrize(("value", "encoding", "expected"), [
    (4, "number", 4), ("4.0", "number", 4), ("0.5", "number", 0.5),
    ("4 servings", "en", 4), ("4 people", "en", 4), ("4 μερίδες", "el", 4),
])
def test_exact_servings(value, encoding, expected):
    assert parse_servings(value, encoding=encoding) == ParsedNumber(expected, "provided")


@pytest.mark.parametrize("value", [
    "4-6", "4 to 6", "about 4", "1 loaf", "2 cups", "One", "1/2",
    0, -1, float("nan"), float("inf"), True, {}, [],
])
def test_invalid_servings_are_not_invented_from_yields(value):
    assert parse_servings(value, encoding="en") == ParsedNumber(status="invalid")


def test_missing_servings():
    assert parse_servings(None) == ParsedNumber()
    assert parse_servings("") == ParsedNumber()


@pytest.mark.parametrize(("steps", "expected"), [
    (None, False), ("", False), (" ", False), ("\x1f", False),
    ("\x1f \t\r\n\x1f", False), ("\u00a0\x1f\u00a0", False),
    ("\u2003\u3000\u2028", False), ("\x1c\x1d\x1e\x1f", False),
    (" \x1fMix.\x1f\u00a0", True), ("\x00", True), ("\u200b", True),
])
def test_usable_steps_match_python_stripped_fragments(steps, expected):
    assert has_usable_steps(steps) is expected
    assert has_usable_steps(steps) == any(
        fragment.strip() for fragment in (steps or "").split("\x1f"))


def test_content_coverage_distinguishes_storage_from_usable_fragments(tmp_path):
    steps = ["", "\x1f", " \t\r\n\x1f", "\u00a0\x1f\u00a0", "\u2003\u3000",
             "Cook.\x1f \u00a0", "\x00", " \x1fMix.\x1f"]
    paths, rows = _inputs(tmp_path, baskets=[[index] for index in range(len(steps))])
    for row, value in zip(rows, steps, strict=True):
        row["steps"] = value
    rows[-1]["title"] = "\u00a0 "
    _rewrite_text(paths, lambda table: pa.Table.from_pylist(rows, schema=table.schema))
    report = _build(paths)
    coverage = report["coverage"]["total"]
    assert coverage["n_recipes"] == 8
    assert coverage["with_steps_storage_nonempty"] == 7
    assert coverage["with_steps"] == 3
    assert coverage["steps_whitespace_or_separator_only"] == 4
    assert coverage["with_title_storage_nonempty"] == 8
    assert coverage["with_title"] == 7
    assert coverage["title_whitespace_only"] == 1
    assert coverage["with_title_and_steps_storage_nonempty"] == 7
    assert coverage["with_title_and_steps"] == 2
    assert report["coverage_definitions"]["version"] == 2
    assert report["verification"]["content_coverage_checked"]
    metadata = load_catalog_metadata(paths.recipes / catalog.CATALOG_FILE)
    assert metadata["coverage"] == report["coverage"]
    assert metadata["coverage_definitions"] == report["coverage_definitions"]
    with sqlite3.connect(paths.recipes / catalog.CATALOG_FILE) as connection:
        assert connection.execute("SELECT steps FROM recipes ORDER BY id").fetchall() == [
            (value,) for value in steps]
        assert connection.execute(
            "SELECT COUNT(*) FROM recipes WHERE steps!=''").fetchone() == (7,)
        assert connection.execute(
            "SELECT COUNT(*) FROM recipes WHERE length(steps)>0").fetchone() == (6,)


def test_full_catalog_retains_singletons_long_recipes_and_exact_text(tmp_path):
    paths, rows = _inputs(tmp_path)
    rows[1].update(
        ingredient_quantities='["1", null, "1/2"]',
        quantity_status="count_mismatch",
        steps='c("Cut 1" thick pieces.", "Cook.")',
        text_status="unparsed_steps")
    _rewrite_text(paths, lambda table: pa.Table.from_pylist(rows, schema=table.schema))
    records = [_record(
        rows[1], [1, 2], total=ParsedNumber(25, "provided"), total_field="TotalTime",
        servings=ParsedNumber(4, "provided"), servings_field="RecipeServings")]
    report = _build(paths, metadata_records=records)
    output = paths.recipes / catalog.CATALOG_FILE
    with sqlite3.connect(output) as connection:
        assert connection.execute("SELECT COUNT(*) FROM recipes").fetchone() == (3,)
        assert connection.execute("SELECT COUNT(*) FROM recipe_fts").fetchone() == (3,)
        assert connection.execute("SELECT SUM(length(ingredient_ids)/2) FROM recipes").fetchone() == (101,)
        assert connection.execute(
            "SELECT rowid FROM recipe_fts WHERE recipe_fts MATCH 'i97'").fetchall() == [(2,)]
        assert connection.execute(
            "SELECT rowid FROM recipe_fts WHERE recipe_fts MATCH 'i0'").fetchall() == [(0,), (2,)]
        assert connection.execute(
            "SELECT rowid FROM recipe_fts WHERE recipe_fts MATCH 'i98'").fetchall() == []
        assert connection.execute(
            "SELECT ingredient_tokens FROM recipe_fts WHERE rowid=1").fetchone() == ("i1 i2",)
        assert connection.execute(
            "SELECT ingredient_ids FROM recipes WHERE id=1").fetchone() == (b"\x01\x00\x02\x00",)
        copied = connection.execute(
            "SELECT raw_ingredients,steps,ingredient_quantities,quantity_status,text_status "
            "FROM recipes WHERE id=1").fetchone()
        assert copied == tuple(rows[1][name] for name in catalog.TEXT_COLUMNS[3:])
        assert connection.execute(
            "SELECT total_minutes,servings,time_status,servings_status FROM recipes WHERE id=1"
        ).fetchone() == (25.0, 4.0, "source_total", "source_servings")
    before = _sha(output)
    metadata = load_catalog_metadata(output)
    assert _sha(output) == before
    assert metadata["schema_version"] == 1 and metadata["partial"] is False
    assert metadata["n_recipes"] == 3 and metadata["n_slots"] == 101
    assert len(metadata["vocabulary"]) == 100
    assert metadata["ingredient_frequency"][:4] == [2, 2, 2, 1]
    assert report["min_ingredients"] == 1 and report["max_ingredients"] == 98
    assert report["singletons"] == 1
    assert report["verification"]["fts_tokens"] == 101
    assert report["coverage"]["total"]["total_minutes_known"] == 1
    assert report["coverage"]["total"]["total_minutes_unknown"] == 2
    assert report["coverage"]["total"]["quantity_status"]["count_mismatch"] == 1
    assert "PRIVATE_TITLE" not in json.dumps(report)
    assert "PRIVATE_STEP" not in json.dumps(report)
    assert str(tmp_path) not in json.dumps(report)
    assert str(Path.cwd()) not in json.dumps(report)
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


@pytest.mark.parametrize("fault", ["idx", "source", "rows", "column"])
def test_text_row_identity_misalignment_never_publishes(tmp_path, fault):
    paths, _ = _inputs(tmp_path)

    def damage(table):
        if fault == "rows":
            return table.slice(0, 2)
        if fault == "column":
            return table.drop(["ingredient_quantities"])
        values = [1, 0, 2] if fault == "idx" else ["other", "fixture", "fixture"]
        column = table.schema.get_field_index(fault)
        field = table.schema.field(fault)
        return table.set_column(column, field, pa.array(values, type=field.type))

    _rewrite_text(paths, damage)
    with pytest.raises(RuntimeError, match="misalignment|row count|required v2"):
        _build(paths)
    _assert_no_catalog(paths)


@pytest.mark.parametrize(("key", "value"), [
    (b"corpus_sha256", b"0" * 64), (b"reader_sha256", b"0" * 64),
    (b"text_schema_version", b"1"), (b"partial", b"true"),
])
def test_text_provenance_misalignment_never_publishes(tmp_path, key, value):
    paths, _ = _inputs(tmp_path)
    _rewrite_text(
        paths, lambda table: table.replace_schema_metadata({**table.schema.metadata, key: value}))
    with pytest.raises(RuntimeError, match="misalignment|immutable v2"):
        _build(paths)
    _assert_no_catalog(paths)


def test_explicit_text_pin_detects_content_drift(tmp_path):
    paths, _ = _inputs(tmp_path)
    with pytest.raises(RuntimeError, match="explicit pin"):
        _build(paths, expected_text_sha256="0" * 64)
    _assert_no_catalog(paths)


@pytest.mark.parametrize("fault", ["sha256", "recipes", "slots", "vocab"])
def test_generation_is_a_required_exact_pin(tmp_path, fault):
    paths, _ = _inputs(tmp_path)
    data = json.loads(paths.generation_file.read_text())
    data[fault] = "0" * 64 if fault == "sha256" else data[fault] + 1
    paths.generation_file.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="generation"):
        _build(paths)
    _assert_no_catalog(paths)


@pytest.mark.parametrize("basket", [[2, 1], [1, 1], []])
def test_invalid_corpus_sets_are_not_silently_repaired(tmp_path, basket):
    paths, _ = _inputs(tmp_path, baskets=[[0], basket])
    with pytest.raises(RuntimeError, match="sorted unique|offsets"):
        _build(paths)
    _assert_no_catalog(paths)


def test_exact_text_cannot_hide_metadata_ingredient_misalignment(tmp_path):
    paths, rows = _inputs(tmp_path)
    wrong = _record(
        rows[0], [1], total=ParsedNumber(10, "provided"), total_field="minutes")
    with pytest.raises(RuntimeError, match="metadata ingredient alignment"):
        _build(paths, metadata_records=[wrong])
    _assert_no_catalog(paths)


def test_metadata_source_identity_misalignment_fails(tmp_path):
    paths, rows = _inputs(tmp_path)
    wrong = replace(_record(rows[0], [0]), source="other")
    with pytest.raises(RuntimeError, match="metadata source identity"):
        _build(paths, metadata_records=[wrong])
    _assert_no_catalog(paths)


def test_required_metadata_identity_drift_fails(tmp_path):
    paths, rows = _inputs(tmp_path)
    record = _record(rows[0], [0])
    with pytest.raises(RuntimeError, match="metadata text identity misalignment"):
        _build(paths, metadata_records=[record], require_metadata_match=True)
    _assert_no_catalog(paths)


def test_conflicting_metadata_is_unknown_not_first_match_wins(tmp_path):
    paths, rows = _inputs(tmp_path, baskets=[[0]])
    first = _record(
        rows[0], [0], total=ParsedNumber(10, "provided"), total_field="minutes",
        servings=ParsedNumber(4, "provided"), servings_field="servings")
    second = replace(first, source_row=1, total=ParsedNumber(20, "provided"))
    report = _build(paths, metadata_records=[first, second])
    with sqlite3.connect(paths.recipes / catalog.CATALOG_FILE) as connection:
        assert connection.execute(
            "SELECT total_minutes,time_status,servings,servings_status,"
            "metadata_status,metadata_match_count FROM recipes"
        ).fetchone() == (None, "ambiguous_total", 4, "source_servings", "ambiguous", 2)
    assert report["coverage"]["total"]["total_minutes_ambiguous"] == 1
    assert report["sources"]["fixture"]["ambiguous_text_keys"] == 1


def test_equivalent_duplicate_metadata_is_reliable(tmp_path):
    paths, rows = _inputs(tmp_path, baskets=[[0]])
    first = _record(rows[0], [0], total=ParsedNumber(10, "provided"), total_field="minutes")
    report = _build(paths, metadata_records=[first, replace(first, source_row=1)])
    assert report["coverage"]["total"]["total_minutes_known"] == 1
    assert report["sources"]["fixture"]["duplicate_metadata_records"] == 1


@pytest.mark.parametrize(("total", "expected_total", "status"), [
    (ParsedNumber(), None, "derived_only"),
    (ParsedNumber(status="invalid"), None, "invalid_total"),
    (ParsedNumber(50, "provided"), 50, "source_total"),
    (ParsedNumber(10, "provided"), None, "inconsistent_total"),
])
def test_derived_sums_never_replace_source_totals(tmp_path, total, expected_total, status):
    paths, rows = _inputs(tmp_path, baskets=[[0]])
    record = _record(
        rows[0], [0], total=total, total_field="TotalTime",
        prep=ParsedNumber(10, "provided"), prep_field="PrepTime",
        cook=ParsedNumber(20, "provided"), cook_field="CookTime")
    _build(paths, metadata_records=[record])
    with sqlite3.connect(paths.recipes / catalog.CATALOG_FILE) as connection:
        assert connection.execute(
            "SELECT total_minutes,time_status,prep_minutes,cook_minutes,derived_total_minutes "
            "FROM recipes").fetchone() == (expected_total, status, 10, 20, 30)
        strict = connection.execute(
            "SELECT id FROM recipes WHERE total_minutes<=60 AND time_status='source_total'"
        ).fetchall()
        assert strict == ([(0,)] if expected_total is not None else [])


def test_existing_catalog_is_never_touched(tmp_path):
    output = tmp_path / "existing.sqlite"
    output.write_bytes(b"must stay intact")
    with pytest.raises(FileExistsError, match="not overwritten"):
        build_catalog(output)
    assert output.read_bytes() == b"must stay intact"


def test_read_only_loader_does_not_create_missing_database(tmp_path):
    output = tmp_path / "missing.sqlite"
    with pytest.raises(FileNotFoundError):
        load_catalog_metadata(output)
    assert not output.exists()


def test_loader_rejects_incomplete_catalog(tmp_path):
    paths, _ = _inputs(tmp_path)
    _build(paths)
    output = paths.recipes / catalog.CATALOG_FILE
    with sqlite3.connect(output) as connection:
        connection.execute("UPDATE metadata SET value='true' WHERE key='partial'")
    with pytest.raises(RuntimeError, match="incomplete"):
        load_catalog_metadata(output)


def test_integrity_failure_does_not_publish(tmp_path, monkeypatch):
    paths, _ = _inputs(tmp_path)

    def fail(*_args, **_kwargs):
        raise RuntimeError("simulated integrity failure")

    monkeypatch.setattr(catalog, "_verify_catalog", fail)
    with pytest.raises(RuntimeError, match="simulated integrity"):
        _build(paths)
    _assert_no_catalog(paths)


def test_persisted_content_coverage_is_checked_before_publication(tmp_path, monkeypatch):
    paths, _ = _inputs(tmp_path)
    original = catalog._count_coverage

    def incorrect_coverage(coverage, text, n_ids, metadata):
        original(coverage, text, n_ids, metadata)
        coverage["with_steps"] += 1

    monkeypatch.setattr(catalog, "_count_coverage", incorrect_coverage)
    with pytest.raises(RuntimeError, match="persisted text content coverage"):
        _build(paths)
    _assert_no_catalog(paths)


def test_interrupted_metadata_stream_does_not_publish(tmp_path):
    paths, rows = _inputs(tmp_path)

    def interrupted():
        yield _record(rows[0], [0])
        raise RuntimeError("interrupted source")

    with pytest.raises(RuntimeError, match="interrupted"):
        _build(paths, metadata_records=interrupted(), batch_size=1)
    _assert_no_catalog(paths)


def test_publish_race_preserves_the_other_output(tmp_path, monkeypatch):
    paths, _ = _inputs(tmp_path)
    output = paths.recipes / catalog.CATALOG_FILE

    def race(_source, destination):
        Path(destination).write_bytes(b"other builder")
        raise FileExistsError(destination)

    monkeypatch.setattr(catalog.os, "link", race)
    with pytest.raises(FileExistsError):
        _build(paths)
    assert output.read_bytes() == b"other builder"
    assert not list(paths.recipes.glob(".*.building*"))


def _stub_source_helpers(paths, monkeypatch, key, kind, rel, column, splitter):
    paths.prior_tools.mkdir(parents=True)
    for filename in (
            "corpus.py", "normalize.py", "lexicons.py", "multilingual.py", "build_recipe_cooc.py"):
        (paths.prior_tools / filename).write_text("# fixture\n")
    raw = SimpleNamespace(
        __file__=str(paths.prior_tools / "corpus.py"), BASE=paths.corpus, EXP=paths.corpus,
        EXPANSION=[(key, "en", kind, rel, column, splitter)])
    nm = SimpleNamespace(__file__=str(paths.prior_tools / "normalize.py"))
    normalizer = SimpleNamespace(
        itos=[f"v{i}" for i in range(100)],
        normalize=lambda _lang, items: {
            int(item[1:]) for item in items if item.startswith("v") and item[1:].isdigit()})
    monkeypatch.setattr(recipe_text, "_load_llmmm", lambda: (raw, nm))
    monkeypatch.setattr(recipe_text, "_corpus_normalizer", lambda _: normalizer)
    return raw


def test_foodcom_enrichment_reuses_skip_identity_and_quantity_semantics(tmp_path, monkeypatch):
    key = "foodcom-522k"
    source_rows = [
        {"Name": "skip", "RecipeIngredientParts": "", "TotalTime": "PT1M"},
        {"Name": "one", "RecipeIngredientParts": 'c("v0")', "TotalTime": "PT10M",
         "RecipeServings": "4", "RecipeIngredientQuantities": 'c("2")'},
        {"Name": "unmapped", "RecipeIngredientParts": 'c("other")', "TotalTime": "PT5M"},
        {"Name": "two", "RecipeIngredientParts": 'c("v1", "v2")',
         "TotalTime": "", "PrepTime": "PT0S", "CookTime": "PT20M",
         "RecipeIngredientQuantities": 'c("1")',
         "RecipeInstructions": 'c("Cut 1" thick pieces.", "Cook.")'},
    ]
    rows = []
    for index, source in enumerate((source_rows[1], source_rows[3])):
        fields = recipe_text._foodcom_fields(
            source["RecipeIngredientParts"], source.get("RecipeIngredientQuantities"),
            source.get("RecipeInstructions"))
        rows.append({
            "idx": index, "source": key, "title": source["Name"], "url": "",
            "raw_ingredients": fields.pop("raw"), **fields,
        })
    paths, _ = _inputs(tmp_path, baskets=[[0], [1, 2]], sources=[key, key], text_rows=rows)
    rel = "10-foodcom-canonical/recipes.csv"
    filename = paths.corpus / rel
    filename.parent.mkdir(parents=True)
    columns = [
        "Name", "RecipeIngredientParts", "RecipeIngredientQuantities",
        "RecipeInstructions", "TotalTime", "PrepTime", "CookTime", "RecipeServings",
    ]
    with filename.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(source_rows)
    _stub_source_helpers(
        paths, monkeypatch, key, "csv", rel, "RecipeIngredientParts",
        lambda value: recipe_text._r_sequence(value))
    report = _build(paths, metadata_records=None)
    with sqlite3.connect(paths.recipes / catalog.CATALOG_FILE) as connection:
        assert connection.execute(
            "SELECT id,total_minutes,servings,time_status FROM recipes ORDER BY id"
        ).fetchall() == [(0, 10.0, 4.0, "source_total"), (1, None, None, "derived_only")]
        assert connection.execute(
            "SELECT ingredient_quantities,quantity_status,text_status,derived_total_minutes "
            "FROM recipes WHERE id=1"
        ).fetchone() == ('["1"]', "count_mismatch", "unparsed_steps", 20.0)
    assert report["sources"][key]["reader_rows"] == 3
    assert report["sources"][key]["empty_normalized"] == 1
    assert report["sources"][key]["matched_text_keys"] == 2
    assert report["sources"][key]["inputs"][0]["sha256"] == _sha(filename)


def test_foodcom_raw_numeric_minutes_are_not_row_zipped(tmp_path, monkeypatch):
    key = "foodcom-raw-231k"
    rows = [
        {"idx": 0, "source": key, "title": "first", "url": "",
         "raw_ingredients": "['v0']", "steps": "", "ingredient_quantities": "[]",
         "quantity_status": "", "text_status": ""},
        {"idx": 1, "source": key, "title": "last", "url": "",
         "raw_ingredients": "['v1']", "steps": "", "ingredient_quantities": "[]",
         "quantity_status": "", "text_status": ""},
    ]
    # _join decodes literal ingredient lists into the v2 separator representation.
    rows[0]["raw_ingredients"] = "v0"
    rows[1]["raw_ingredients"] = "v1"
    paths, _ = _inputs(tmp_path, baskets=[[0], [1]], sources=[key, key], text_rows=rows)
    rel = "11-foodcom-raw/food_recipes.parquet"
    filename = paths.corpus / rel
    filename.parent.mkdir(parents=True)
    pq.write_table(pa.table({
        "name": ["skip", "first", "not_in_corpus", "last"],
        "ingredients": [None, "['v0']", "['other']", "['v1']"],
        "steps": ["", "", "", ""], "minutes": [1, 12, 30, -1],
    }), filename)
    _stub_source_helpers(
        paths, monkeypatch, key, "parquet", rel, "ingredients",
        lambda value: [] if value is None else json.loads(value.replace("'", '"')))
    report = _build(paths, metadata_records=None)
    with sqlite3.connect(paths.recipes / catalog.CATALOG_FILE) as connection:
        assert connection.execute(
            "SELECT id,total_minutes,time_status,metadata_source_row FROM recipes ORDER BY id"
        ).fetchall() == [(0, 12.0, "source_total", 0), (1, None, "invalid_total", 2)]
    assert report["coverage"]["by_source"][key]["total_minutes_invalid"] == 1
    assert report["sources"][key]["reader_rows"] == 3


def test_source_reader_identity_drift_fails_before_publication(tmp_path, monkeypatch):
    key = "foodcom-raw-231k"
    paths, _ = _inputs(tmp_path, baskets=[[0]], sources=[key])
    rel = "11-foodcom-raw/food_recipes.parquet"
    filename = paths.corpus / rel
    filename.parent.mkdir(parents=True)
    pq.write_table(pa.table({
        "name": ["real"], "ingredients": ["['v0']"], "minutes": [20], "steps": [""],
    }), filename)
    _stub_source_helpers(
        paths, monkeypatch, key, "parquet", rel, "ingredients",
        lambda value: ["v0"])
    monkeypatch.setattr(recipe_text, "_meta_stream", lambda *_: iter([
        {**recipe_text._blank(), "title": "wrong", "raw": "v0"}]))
    with pytest.raises(RuntimeError, match="reader identity misalignment"):
        _build(paths, metadata_records=None)
    _assert_no_catalog(paths)


def test_jsonl_optional_metadata_fields_remain_missing(tmp_path, monkeypatch):
    key = "halal-2k"
    rows = [{
        "idx": 0, "source": key, "title": "plain", "url": "",
        "raw_ingredients": "v0", "steps": "", "ingredient_quantities": "[]",
        "quantity_status": "", "text_status": "",
    }]
    paths, _ = _inputs(tmp_path, baskets=[[0]], sources=[key], text_rows=rows)
    rel = "25-halal/data/recipes.jsonl"
    filename = paths.corpus / rel
    filename.parent.mkdir(parents=True)
    filename.write_text(json.dumps({"title": "plain", "ingredients": ["v0"]}) + "\n")
    _stub_source_helpers(
        paths, monkeypatch, key, "jsonl", rel, "ingredients",
        lambda value: value or [])
    report = _build(paths, metadata_records=None)
    assert report["coverage"]["total"]["total_minutes_missing"] == 1
    assert report["coverage"]["total"]["servings_missing"] == 1
    assert report["sources"][key]["matched_text_keys"] == 1
