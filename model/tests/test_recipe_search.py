from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from types import ModuleType

import numpy as np
import pytest

from ingredient_model.recipe_search import RecipeFinder, RecipeQuery


class FixturePolicy:
    def score(self, features):
        return np.arange(len(features), dtype=np.float32)


@pytest.fixture(autouse=True)
def isolated_ranking_features(monkeypatch):
    """Constraint tests must hold regardless of the learned feature/scoring implementation."""
    module = ModuleType("ingredient_model.recipe_ranker")
    module.candidate_features = lambda available, candidates, **kwargs: np.zeros(
        (len(candidates), 1), dtype=np.float32)
    module.heuristic_scores = lambda features: features[:, 0]
    monkeypatch.setitem(sys.modules, module.__name__, module)


@pytest.fixture
def catalog(tmp_path):
    path = tmp_path / "recipes.sqlite"
    vocabulary = ["chicken", "rice", "broccoli", "peanut", "salt", "pepper"]
    records = [
        (0, [0, 1, 2], 20, 2, "Complete recipe"),
        (1, [0, 1], 60, 4, "Slow recipe"),
        (2, [0, 2], None, 2, "Unknown time"),
        (3, [0, 1, 3], 25, 2, "Contains peanut"),
        (4, [0, 1], 30.25, 2, "Over the time limit"),
        (5, [0, 1, 2, 4, 5], 20, 2, "Needs extra ingredients"),
        (6, [0, 1], 20, None, "Unknown servings"),
        (7, [0, 1], 15, 2, "Malformed source URL"),
        (8, [0, 1], 20, 2, "Missing instructions"),
    ]
    frequency = np.bincount(
        np.concatenate([row[1] for row in records]), minlength=len(vocabulary)).tolist()
    metadata = {
        "schema_version": 1, "partial": False, "n_recipes": len(records),
        "n_slots": sum(len(row[1]) for row in records), "vocabulary": vocabulary,
        "ingredient_frequency": frequency, "corpus_sha256": "a" * 64,
        "text_index_sha256": "b" * 64,
        "coverage": {},
    }
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE recipes(
            id INTEGER PRIMARY KEY, source TEXT, language TEXT, title TEXT, url TEXT,
            ingredient_ids BLOB, raw_ingredients TEXT, steps TEXT,
            ingredient_quantities TEXT, quantity_status TEXT, text_status TEXT,
            total_minutes REAL, servings REAL, time_status TEXT, servings_status TEXT);
        CREATE VIRTUAL TABLE recipe_fts USING fts5(ingredient_tokens);
    """)
    connection.executemany("INSERT INTO metadata VALUES (?, ?)",
                           [(key, json.dumps(value)) for key, value in metadata.items()])
    for recipe_id, ids, minutes, servings, title in records:
        connection.execute(
            "INSERT INTO recipes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (recipe_id, "fixture", "en", title,
             "javascript:alert(1)" if recipe_id == 7 else f"https://example.test/{recipe_id}",
             np.asarray(ids, dtype="<u2").tobytes(),
             "\x1f".join(vocabulary[value] for value in ids),
             "" if recipe_id == 8 else "Fixture instruction.",
             '["1", "2", "3"]' if recipe_id == 7 else "[]",
             "count_mismatch" if recipe_id == 7 else "", "",
             minutes, servings, "unknown" if minutes is None else "source_total",
             "unknown" if servings is None else "source_servings"))
        connection.execute(
            "INSERT INTO recipe_fts(rowid, ingredient_tokens) VALUES (?, ?)",
            (recipe_id, " ".join(f"i{value}" for value in ids)))
    connection.commit()
    connection.close()
    return path


def test_hard_constraints_hold_regardless_of_policy_preferences(catalog):
    finder = RecipeFinder(catalog, policy=FixturePolicy())
    result = finder.search(RecipeQuery(
        ["chicken", "rice", "broccoli"], must_use=["chicken"], exclude=["peanut"],
        max_total_minutes=30, max_missing=0, min_servings=2))
    assert {recipe.recipe_id for recipe in result.recipes} == {0, 7}
    assert result.retrieval_truncated is False
    for recipe in result.recipes:
        assert recipe.total_minutes is not None and recipe.total_minutes <= 30
        assert recipe.servings is not None and recipe.servings >= 2
        assert "chicken" in recipe.ingredients
        assert "peanut" not in recipe.ingredients
        assert not recipe.missing_ingredients
        assert recipe.steps
    malformed = next(recipe for recipe in result.recipes if recipe.recipe_id == 7)
    assert malformed.source_url is None
    assert "source_url_unavailable" in malformed.warnings
    assert malformed.ingredient_quantities == ("1", "2", "3")
    assert "count_mismatch" in malformed.warnings


def test_unknown_time_is_not_silently_treated_as_zero(catalog):
    finder = RecipeFinder(catalog, policy=FixturePolicy())
    unrestricted = finder.search(RecipeQuery(["chicken", "broccoli"], max_missing=1))
    unknown = next(recipe for recipe in unrestricted.recipes if recipe.recipe_id == 2)
    assert unknown.total_minutes is None
    assert "total_time_unknown" in unknown.warnings
    restricted = finder.search(RecipeQuery(
        ["chicken", "broccoli"], max_total_minutes=30, max_missing=1))
    assert all(recipe.recipe_id != 2 for recipe in restricted.recipes)
    assert all(recipe.recipe_id != 4 for recipe in restricted.recipes)


def test_derived_totals_cannot_masquerade_as_reported_cooking_times(catalog):
    with sqlite3.connect(catalog) as connection:
        connection.execute("UPDATE recipes SET time_status='derived_sum' WHERE id=0")
    finder = RecipeFinder(catalog, policy=FixturePolicy())
    with pytest.raises(ValueError, match="source-total provenance"):
        finder.search(RecipeQuery(["chicken", "rice", "broccoli"], max_total_minutes=30))


def test_zero_reported_servings_are_not_valid_metadata(catalog):
    with sqlite3.connect(catalog) as connection:
        connection.execute("UPDATE recipes SET servings=0 WHERE id=0")
    finder = RecipeFinder(catalog, policy=FixturePolicy())
    with pytest.raises(ValueError, match="servings must be positive"):
        finder.search(RecipeQuery(["chicken", "rice", "broccoli"]))


def test_derived_servings_cannot_masquerade_as_source_reports(catalog):
    with sqlite3.connect(catalog) as connection:
        connection.execute("UPDATE recipes SET servings_status='scaled' WHERE id=0")
    finder = RecipeFinder(catalog, policy=FixturePolicy())
    with pytest.raises(ValueError, match="source-serving provenance"):
        finder.search(RecipeQuery(["chicken", "rice", "broccoli"], min_servings=2))


@pytest.mark.parametrize("overrides", [
    {"available_ingredients": []},
    {"available_ingredients": "chicken"},
    {"available_ingredients": ["invented ingredient"]},
    {"must_use": ["peanut"], "exclude": ["peanut"]},
    {"available_ingredients": ["chicken"], "exclude": ["chicken"]},
    {"max_total_minutes": float("nan")},
    {"max_total_minutes": -1},
    {"max_total_minutes": True},
    {"max_missing": -1},
    {"max_missing": True},
    {"min_servings": 0},
    {"top_k": 0},
    {"top_k": True},
    {"language": ""},
])
def test_invalid_constraints_fail_explicitly(catalog, overrides):
    finder = RecipeFinder(catalog, policy=FixturePolicy())
    with pytest.raises(ValueError):
        finder.search(RecipeQuery(**{"available_ingredients": ["chicken", "rice"], **overrides}))


def test_truncation_is_visible_even_when_no_feasible_result_is_found(catalog):
    finder = RecipeFinder(catalog, policy=FixturePolicy(), max_candidates=1, scan_limit=1)
    result = finder.search(RecipeQuery(["chicken"], max_missing=0, top_k=1))
    assert not result.recipes
    assert result.retrieval_truncated is True
    assert result.warnings


def test_shortlist_limit_is_not_reported_as_exhaustive_search(catalog):
    finder = RecipeFinder(catalog, policy=FixturePolicy(), max_candidates=1, scan_limit=10)
    result = finder.search(RecipeQuery(["chicken", "rice"], top_k=1))
    assert len(result.recipes) == 1
    assert result.feasible_candidates == 1
    assert result.retrieval_truncated is True


def test_required_ingredients_do_not_remove_other_pantry_terms_from_retrieval(catalog):
    finder = RecipeFinder(catalog, policy=FixturePolicy(), max_candidates=1, scan_limit=20)
    result = finder.search(RecipeQuery(
        ["chicken", "rice", "broccoli"], must_use=["chicken"], top_k=1))
    assert "broccoli" in result.recipes[0].ingredients


@pytest.mark.parametrize("steps", ["\x1f\t\n\x1f", "\u3000\u00a0", " \x1f \u2003 "])
def test_whitespace_only_instructions_are_not_usable_recipes(catalog, steps):
    with sqlite3.connect(catalog) as connection:
        connection.execute("UPDATE recipes SET steps=? WHERE id=8", (steps,))
    result = RecipeFinder(catalog, policy=FixturePolicy()).search(
        RecipeQuery(["chicken", "rice"]))
    assert all(recipe.recipe_id != 8 for recipe in result.recipes)


def test_instruction_cleanup_preserves_separate_raw_quantity_positions(catalog):
    with sqlite3.connect(catalog) as connection:
        connection.execute(
            "UPDATE recipes SET steps=?,raw_ingredients=?,ingredient_quantities=? WHERE id=7",
            ("  First. \x1f\x1f Second. ", "chicken\x1f\x1frice", '["1", null, "2"]'))
    result = RecipeFinder(catalog, policy=FixturePolicy()).search(
        RecipeQuery(["chicken", "rice"]))
    recipe = next(recipe for recipe in result.recipes if recipe.recipe_id == 7)
    assert recipe.steps == ("First.", "Second.")
    assert recipe.raw_ingredients == ("chicken", "", "rice")
    assert recipe.ingredient_quantities == ("1", None, "2")


def test_explicit_or_required_metadata_index_cannot_fall_back(catalog, tmp_path):
    with pytest.raises(FileNotFoundError, match="metadata index"):
        RecipeFinder(catalog, policy=FixturePolicy(), metadata_index_path=tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="metadata index"):
        RecipeFinder(catalog, policy=FixturePolicy(), require_metadata_index=True)


def test_metadata_acceleration_preserves_results_and_shortlist(catalog):
    from ingredient_model.data.recipe_search_metadata import build_search_metadata

    reference = RecipeFinder(catalog, policy=FixturePolicy())
    build_search_metadata(catalog)
    accelerated = RecipeFinder(catalog, policy=FixturePolicy(), require_metadata_index=True)
    for budget in (None, 0, 20, 30, 30.25):
        query = RecipeQuery(
            ["chicken", "rice", "broccoli"], must_use=["chicken"], exclude=["peanut"],
            max_total_minutes=budget, min_servings=2, max_missing=0)
        before = reference.search(query, include_candidate_ids=True)
        after = accelerated.search(query, include_candidate_ids=True)
        assert before.recipes == after.recipes
        assert before.candidate_recipe_ids == after.candidate_recipe_ids
        assert before.candidates_scanned == after.candidates_scanned
        assert before.retrieval_truncated == after.retrieval_truncated
        assert before.metadata_index_used is False
        assert after.metadata_index_used is True


def test_canonical_vector_filter_preserves_results_and_scanned_candidates(catalog, tmp_path):
    from ingredient_model._hashing import file_sha256
    from ingredient_model.data.recipe_search_metadata import build_search_metadata

    with sqlite3.connect(catalog) as connection:
        rows = [np.frombuffer(row[0], dtype="<u2") for row in connection.execute(
            "SELECT ingredient_ids FROM recipes ORDER BY id")]
        path = tmp_path / "canonical.npz"
        np.savez(path, flat=np.concatenate(rows),
                 offsets=np.asarray([0, *np.cumsum([len(row) for row in rows])], dtype=np.int64))
        connection.execute("UPDATE metadata SET value=? WHERE key='corpus_sha256'",
                           (json.dumps(file_sha256(path)),))
    reference = RecipeFinder(catalog, policy=FixturePolicy())
    build_search_metadata(catalog)
    accelerated = RecipeFinder(catalog, policy=FixturePolicy(), corpus_path=path,
                               require_corpus_index=True, require_metadata_index=True)
    for missing in (0, 1, 3, None):
        query = RecipeQuery(["chicken", "rice", "broccoli"], max_missing=missing)
        before = reference.search(query, include_candidate_ids=True)
        after = accelerated.search(query, include_candidate_ids=True)
        assert before.recipes == after.recipes
        assert before.candidate_recipe_ids == after.candidate_recipe_ids
        assert before.candidates_scanned == after.candidates_scanned
        assert after.ingredient_index_used is True
    assert reference._ingredient_index is None


def test_nonfinite_policy_output_cannot_produce_recommendations(catalog):
    class InvalidPolicy:
        def score(self, features):
            return np.full(len(features), np.nan)

    finder = RecipeFinder(catalog, policy=InvalidPolicy())
    with pytest.raises(ValueError, match="invalid scores"):
        finder.search(RecipeQuery(["chicken", "rice"]))


def test_a_heuristic_fallback_must_be_explicit(catalog):
    with pytest.raises(ValueError, match="requires a trained policy"):
        RecipeFinder(catalog)
    finder = RecipeFinder(catalog, ranking="heuristic")
    assert finder.search(RecipeQuery(["chicken", "rice"])).ranking == "heuristic"


def test_search_does_not_modify_the_catalog(catalog):
    before = hashlib.sha256(catalog.read_bytes()).hexdigest()
    finder = RecipeFinder(catalog, policy=FixturePolicy())
    first = finder.search(RecipeQuery(["chicken", "rice"]))
    second = finder.search(RecipeQuery(["chicken", "rice"]))
    assert first.recipes == second.recipes
    assert hashlib.sha256(catalog.read_bytes()).hexdigest() == before


def test_open_finder_rejects_a_changed_catalog(catalog):
    finder = RecipeFinder(catalog, policy=FixturePolicy())
    with sqlite3.connect(catalog) as connection:
        connection.execute("UPDATE recipes SET total_minutes=100 WHERE id=0")
    with pytest.raises(ValueError, match="changed after initialization"):
        finder.search(RecipeQuery(["chicken", "rice"]))


def test_partial_catalog_requires_an_explicit_opt_in(catalog):
    with sqlite3.connect(catalog) as connection:
        connection.execute("UPDATE metadata SET value='true' WHERE key='partial'")
    with pytest.raises(ValueError, match="partial catalog"):
        RecipeFinder(catalog, policy=FixturePolicy())
    finder = RecipeFinder(catalog, policy=FixturePolicy(), allow_partial=True)
    result = finder.search(RecipeQuery(["chicken", "rice"]))
    assert "partial_catalog" in " ".join(result.warnings)


def test_cli_returns_results_and_explicit_constraint_errors(catalog, capsys):
    from ingredient_model import cli

    command = ["find", "--catalog", str(catalog), "--heuristic",
               "--ingredients", "chicken", "rice", "--max-total-minutes", "30"]
    assert cli.main(command) == 0
    response = json.loads(capsys.readouterr().out)
    assert response["ranking"] == "heuristic"
    assert all(recipe["total_minutes"] <= 30 for recipe in response["recipes"])
    assert cli.main(command + ["--max-missing", "-1"]) == 2
    assert "max_missing" in capsys.readouterr().err


def test_hub_bundle_is_bound_to_verified_policy_bytes_and_catalog(catalog, tmp_path, monkeypatch):
    pytest.importorskip("huggingface_hub")
    bundle = tmp_path / "bundle"
    policy_dir = bundle / "recipe_policy"
    policy_dir.mkdir(parents=True)
    policy_file = policy_dir / "policy.safetensors"
    policy_file.write_bytes(b"fixture-policy")
    vocabulary = ["chicken", "rice", "broccoli", "peanut", "salt", "pepper"]
    configuration = {
        "schema_version": 1, "deployed_ranker": "learned",
        "search_defaults": {"max_candidates": 12, "scan_limit": 40, "timeout_seconds": 1.5},
        "corpus_sha256": "a" * 64,
        "vocabulary_sha256": hashlib.sha256(json.dumps(
            vocabulary, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
        "policy_files": {
            "policy.safetensors": hashlib.sha256(policy_file.read_bytes()).hexdigest(),
        },
    }
    config = bundle / "recipe_search_config.json"
    config.write_text(json.dumps(configuration))
    monkeypatch.setattr(RecipeFinder, "from_directory", classmethod(
        lambda cls, path, *, catalog_path, **options:
            cls(catalog_path, policy=FixturePolicy(), **options)))
    finder = RecipeFinder.from_pretrained(str(bundle), catalog_path=catalog)
    assert finder.vocabulary == tuple(vocabulary)
    assert (finder.max_candidates, finder.scan_limit, finder.timeout_seconds) == (12, 40, 1.5)
    overridden = RecipeFinder.from_pretrained(
        str(bundle), catalog_path=catalog, max_candidates=15)
    assert overridden.max_candidates == 15
    assert overridden.scan_limit == 40
    configuration["corpus_sha256"] = "b" * 64
    config.write_text(json.dumps(configuration))
    with pytest.raises(ValueError, match="different corpus/vocabulary"):
        RecipeFinder.from_pretrained(str(bundle), catalog_path=catalog)
    policy_file.write_bytes(b"changed-policy")
    with pytest.raises(ValueError, match="checksum"):
        RecipeFinder.from_pretrained(str(bundle), catalog_path=catalog)


def test_hub_bundle_cannot_request_files_outside_its_policy_directory(catalog, tmp_path):
    pytest.importorskip("huggingface_hub")
    bundle = tmp_path / "invalid-bundle"
    bundle.mkdir()
    (bundle / "recipe_search_config.json").write_text(json.dumps({
        "schema_version": 1, "policy_files": {"../outside.json": "a" * 64},
    }))
    with pytest.raises(ValueError, match="safe file inventory"):
        RecipeFinder.from_pretrained(str(bundle), catalog_path=catalog)
