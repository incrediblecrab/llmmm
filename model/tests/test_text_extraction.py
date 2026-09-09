from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from ingredient_model.data.recipes import RecipeCorpus
from ingredient_model.data.text import _foodcom_fields, _r_sequence, build_index


def test_r_missing_values_do_not_shift_quantities():
    assert _r_sequence('c("1", NA, "1/2")') == ["1", None, "1/2"]
    result = _foodcom_fields(
        'c("flour", "salt", "milk")', 'c("1", NA, "1/2")',
        'c("Mix the ingredients.", "Cook, then serve.")')
    assert result["raw"] == "flour\x1fsalt\x1fmilk"
    assert json.loads(result["ingredient_quantities"]) == ["1", None, "1/2"]
    assert result["quantity_status"] == "values_without_units"
    assert result["steps"] == "Mix the ingredients.\x1fCook, then serve."


def test_r_quoted_commas_and_escapes_are_preserved():
    assert _r_sequence(r'c("milk, cold", "stir \"gently\"")') == [
        "milk, cold", 'stir "gently"']
    assert _r_sequence('"flour"') == ["flour"]
    assert _r_sequence("NA") == []
    assert _r_sequence('"Cut into 1 1/2" pieces."') == ['Cut into 1 1/2" pieces.']


@pytest.mark.parametrize("value", [
    "c(system('x'))", 'c("unterminated"', "c(x + y)", 'c("flour" "salt")',
])
def test_r_expressions_are_rejected_not_executed(value):
    with pytest.raises(ValueError):
        _r_sequence(value)


def test_foodcom_mismatched_quantities_are_not_zipped_silently():
    result = _foodcom_fields('c("flour", "salt")', 'c("1")', "Cook.")
    assert result["raw"] == "flour\x1fsalt"
    assert json.loads(result["ingredient_quantities"]) == ["1"]
    assert result["quantity_status"] == "count_mismatch"


def test_malformed_instruction_arrays_are_retained_and_flagged():
    original = 'c("Cut 1" thick pieces.", "Cook.")'
    result = _foodcom_fields('c("flour", "salt")', 'c("1", NA)', original)
    assert result["steps"] == original
    assert result["text_status"] == "unparsed_steps"


def test_index_build_refuses_to_truncate_an_existing_artifact(tmp_path):
    path = tmp_path / "index.parquet"
    path.write_bytes(b"existing index")
    with pytest.raises(FileExistsError, match="not overwritten"):
        build_index(path)
    assert path.read_bytes() == b"existing index"


def test_alignment_failure_does_not_publish_a_partial_index(tmp_path, monkeypatch):
    from ingredient_model import config
    from ingredient_model.data import recipes, text

    corpus = RecipeCorpus(
        flat=np.array([0, 1, 2, 1, 2, 3]),
        offsets=np.array([0, 3, 6]),
        source=np.array(["fixture", "fixture"]),
        lang=np.array(["en", "en"]), itos=["0", "1", "2", "3"])
    raw = SimpleNamespace(iter_all=lambda _: iter([
        ("fixture", "en", ["0", "1", "2"]),
        ("fixture", "en", ["0", "1", "3"]),
    ]))
    normalizer = SimpleNamespace(
        itos=corpus.itos, normalize=lambda _language, items: [int(item) for item in items])
    monkeypatch.setattr(recipes, "load_recipes", lambda: corpus)
    monkeypatch.setattr(text, "_load_llmmm", lambda: (raw, None))
    monkeypatch.setattr(text, "_corpus_normalizer", lambda _: normalizer)
    monkeypatch.setattr(text, "_meta_stream", lambda *_: iter([text._blank(), text._blank()]))
    monkeypatch.setattr(config, "corpus_generation", lambda: {"sha256": "a" * 64})
    path = tmp_path / "index.parquet"
    with pytest.raises(RuntimeError, match="alignment lost"):
        build_index(path, chunk=1)
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []
