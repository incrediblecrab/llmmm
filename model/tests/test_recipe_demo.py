from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")

from ingredient_model.recipe_demo import (
    add_demo_links, browser_policy, example_pantries, load_sample, make_catalog, verify_browser,
)
from ingredient_model.recipe_ranker import RecipeRankingPolicy

ROOT = Path(__file__).resolve().parents[2]
VOCABULARY = ["egg", "milk", "salt", "pepper"]


def sample_row(name="example", ingredients=None, minutes=10, servings=2):
    return {
        "id": name, "title": "Synthetic fixture, not a cooking recipe", "language": "en",
        "canonical_ingredients": ingredients or ["egg", "salt"],
        "raw_ingredients": ["Synthetic fixture ingredient"],
        "instructions": ["Synthetic fixture instruction"],
        "total_minutes": minutes, "servings": servings,
        "time_evidence": None if minutes is None else "Time: 10 minutes",
        "servings_evidence": None if servings is None else "Serves: 2",
        "source_url": "https://en.wikibooks.org/w/index.php?title=Cookbook:Test&oldid=123",
        "source_title": "Cookbook:Test", "source_revision_id": 123,
        "retrieved_at": "2026-09-10T00:00:00+00:00", "license": "cc-by-sa-4.0",
        "attribution": "Synthetic test fixture",
        "attribution_url": "https://en.wikibooks.org/w/index.php?title=Cookbook:Test&action=history",
        "changes": "Synthetic test fixture, never published as a real recipe",
        "unmapped_ingredients": [],
    }


def write_rows(tmp_path, rows):
    path = tmp_path / "recipes.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_sample_is_separate_complete_catalog_with_full_public_vocabulary(tmp_path):
    rows = [sample_row(), sample_row("second", ["egg", "milk"], None, None)]
    loaded = load_sample(write_rows(tmp_path, rows), VOCABULARY)
    catalog = make_catalog(loaded, VOCABULARY)
    assert catalog["n_recipes"] == 2
    assert catalog["vocabulary"] == VOCABULARY
    assert catalog["ingredient_frequency"] == [2, 1, 1, 0]
    assert catalog["statistics_scope"] == "public-sample"
    assert loaded[1]["total_minutes"] is None
    assert loaded[1]["servings"] is None
    assert example_pantries(catalog)[0]["available_ingredients"] == ["egg", "milk", "salt"]


@pytest.mark.parametrize("patch", [
    {"id": "../private"},
    {"canonical_ingredients": ["unrecognized"]},
    {"canonical_ingredients": ["egg", "egg"]},
    {"license": "mit"},
    {"raw_ingredients": []},
    {"instructions": [" "]},
    {"attribution": ""},
    {"unmapped_ingredients": None},
    {"total_minutes": 0},
    {"total_minutes": float("nan")},
    {"total_minutes": True},
    {"total_minutes": "10"},
    {"servings": -1},
    {"time_evidence": None},
    {"servings_evidence": ""},
    {"source_revision_id": 124},
    {"source_url": "https://en.wikibooks.org/wiki/Cookbook:Test"},
    {"attribution_url": "javascript:alert(1)"},
])
def test_sample_rejects_unverifiable_or_malformed_data(tmp_path, patch):
    with pytest.raises(ValueError):
        load_sample(write_rows(tmp_path, [sample_row() | patch]), VOCABULARY)


def test_sample_rejects_duplicate_rows_and_implicit_unknown_metadata(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        load_sample(write_rows(tmp_path, [sample_row(), sample_row()]), VOCABULARY)
    row = sample_row()
    del row["total_minutes"]
    with pytest.raises(ValueError, match="explicit null"):
        load_sample(write_rows(tmp_path, [row]), VOCABULARY)


def test_sample_rejects_duplicate_json_fields_and_nonfinite_extra_fields(tmp_path):
    path = tmp_path / "recipes.jsonl"
    path.write_text('{"id":"first","id":"second"}\n')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_sample(path, VOCABULARY)
    with pytest.raises(ValueError, match="non-finite"):
        load_sample(write_rows(tmp_path, [sample_row() | {"extra": float("inf")}]), VOCABULARY)


@pytest.mark.parametrize("time_enabled", [True, False])
@pytest.mark.skipif(shutil.which("node") is None, reason="browser parity requires the optional Node runtime")
def test_browser_features_scores_and_searches_match_python(time_enabled):
    rows = [
        sample_row(),
        sample_row("second", ["egg", "milk"], None, 4),
        sample_row("third", ["milk", "salt", "pepper"], 30, None),
    ]
    catalog = make_catalog(rows, VOCABULARY)
    policy = RecipeRankingPolicy(hidden_dim=32, seed=73, time_features_enabled=time_enabled)
    exported = browser_policy(policy, {"purpose": "synthetic test fixture"})
    assert exported["learned_parameter_count"] == 705
    result = verify_browser(ROOT, catalog, policy, exported, cases=8)
    assert result["feature_rows"] == 64
    assert result["python_finder_searches_matched"] == 42
    assert result["private_corpus_used"] is False
    assert result["quality_benchmark"] is False


def test_model_card_demo_section_is_idempotent_and_keeps_scope_caveats():
    original = "# Model\n\n## Finding recipes\n\nExisting model evidence.\n"
    updated = add_demo_links(original)
    assert add_demo_links(updated) == updated
    assert "Existing model evidence." in updated
    assert updated.count("## Try the public demo") == 1
    assert "not the full training catalog or a new quality benchmark" in updated
    assert "weights retain their existing terms" in updated
    with pytest.raises(ValueError, match="malformed"):
        add_demo_links(original + "<!-- PUBLIC-DEMO:START -->")
