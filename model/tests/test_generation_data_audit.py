"""Coverage must not turn empty or quantity-only fields into training-ready recipes."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_generation_data.py"
SPEC = importlib.util.spec_from_file_location("audit_generation_data", PATH)
assert SPEC is not None and SPEC.loader is not None
audit_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)


def sample():
    return pa.record_batch({
        "idx": [0, 1, 2, 3],
        "source": ["a", "a", "b", "b"],
        "title": ["Soup", "Soup", "", "Dish"],
        "raw_ingredients": ['c("1/2", "2", NA, None)', "2 onions", None, "\u8c46\u8150"],
        "steps": ["Cook.", "Cook.", " ", "\u716e"],
    })


def test_quantity_only_and_missing_fields_are_separate():
    got = audit_module.summarize_batch(sample())
    assert got["a"]["complete_text"] == 2
    assert got["a"]["complete_with_ingredient_letters"] == 1
    assert got["b"]["complete_text"] == 1
    assert got["b"]["ingredient_letters"] == 1


def test_streaming_audit_checks_row_identity_and_records_digest(tmp_path):
    path = tmp_path / "text.parquet"
    pq.write_table(pa.Table.from_batches([sample()]), path)
    generation = {"generation": "fixture", "recipes": 4, "sha256": "a" * 64}
    result = audit_module.audit(path, generation, batch_size=2)
    assert result["totals"]["rows"] == 4
    assert result["totals"]["complete_text"] == 3
    assert len(result["text_index"]["sha256"]) == 64
    broken = sample().set_column(0, "idx", pa.array([0, 1, 2, 5]))
    pq.write_table(pa.Table.from_batches([broken]), path)
    with pytest.raises(ValueError, match="identity drift"):
        audit_module.audit(path, generation)


def test_quantity_column_risk_is_recorded_even_when_some_rows_have_letters(tmp_path):
    batch = sample().set_column(
        1, "source", pa.array(["foodcom-522k", "foodcom-522k", "b", "b"]))
    path = tmp_path / "text.parquet"
    pq.write_table(pa.Table.from_batches([batch]), path)
    result = audit_module.audit(
        path, {"generation": "fixture", "recipes": 4, "sha256": "a" * 64})
    issue, = result["source_field_issues"]
    assert issue["rows_in_source"] == 2
    assert issue["rows_failing_ingredient_letter_screen"] == 1


def test_audit_cli_cannot_replace_its_input(tmp_path):
    source = tmp_path / "index.json"
    source.write_text("original bytes")
    result = subprocess.run(
        [sys.executable, str(PATH), "--input", str(source), "--out", str(source)],
        capture_output=True, text=True)
    assert result.returncode != 0
    assert "must not overwrite" in result.stderr
    assert source.read_text() == "original bytes"


def test_repaired_names_do_not_imply_recovered_quantity_units(tmp_path):
    batch = sample().append_column(
        "ingredient_quantities", pa.array(['["1",null]', "[]", "[]", "[]"]))
    batch = batch.append_column(
        "quantity_status", pa.array(["values_without_units", "", "", ""]))
    table = pa.Table.from_batches([batch]).replace_schema_metadata({"text_schema_version": "2"})
    path = tmp_path / "text.parquet"
    pq.write_table(table, path)
    result = audit_module.audit(
        path, {"generation": "fixture", "recipes": 4, "sha256": "a" * 64})
    assert result["totals"]["unresolved_quantity_units"] == 1
    assert result["source_field_issues"][0]["rows_with_quantity_values_without_units"] == 1
