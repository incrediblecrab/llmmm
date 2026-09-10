from __future__ import annotations

import importlib
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def evaluation(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("evaluate_recipe_search")


def observation(rank, *, timeout=False, violations=0):
    return {
        "rank": rank, "latency_ms": 5000 if timeout else 20,
        "shortlist_hit": rank != float("inf"), "empty": rank == float("inf"),
        "truncated": timeout, "timeout": timeout, "constraint_violations": violations,
    }


def test_timeouts_and_misses_remain_in_the_recovery_denominator(evaluation):
    result = evaluation._summary([
        observation(1), observation(3), observation(float("inf"), timeout=True),
        observation(float("inf"), violations=1)])
    assert result["queries"] == 4
    assert result["source_set_recall_at_1"] == 0.25
    assert result["source_set_recall_at_5"] == 0.5
    assert result["mrr_at_10"] == pytest.approx(1 / 3)
    assert result["source_set_in_shortlist"] == 0.5
    assert result["timeouts"] == 1
    assert result["constraint_violations"] == 1


def test_paired_intervals_do_not_turn_a_tie_into_improvement(evaluation):
    tied = evaluation._paired_interval([observation(1)] * 20, [observation(1)] * 20, 42)
    assert tied == {"difference_in_recall_at_5": 0.0, "ci95": [0.0, 0.0]}
    wins = evaluation._paired_interval(
        [observation(1)] * 20, [observation(float("inf"))] * 20, 42)
    assert wins == {"difference_in_recall_at_5": 1.0, "ci95": [1.0, 1.0]}


def test_operational_gate_rejects_empty_successes_and_deadline_overruns(evaluation):
    good = evaluation._summary([observation(1)] * 38 + [observation(float("inf"))] * 2)
    assert evaluation._operational_pass(good, 5)
    too_many_misses = evaluation._summary(
        [observation(1)] * 37 + [observation(float("inf"))] * 3)
    assert not evaluation._operational_pass(too_many_misses, 5)
    good["latency_ms"]["maximum"] = 5000.01
    assert not evaluation._operational_pass(good, 5)


@pytest.mark.parametrize("validation_only", [False, True])
def test_selection_precedes_test_generation_and_debug_mode_leaves_test_untouched(
        evaluation, monkeypatch, tmp_path, validation_only):
    np.savez(tmp_path / "recipe_ids.npz", flat=np.asarray([0, 1], dtype=np.uint16),
             offsets=np.asarray([0, 2], dtype=np.int64))
    (tmp_path / "recipe_search.sqlite").write_bytes(b"opaque fixture catalog")
    corpus_hash = evaluation.file_sha256(tmp_path / "recipe_ids.npz")
    events = []

    class Finder:
        def __init__(self, *args, **kwargs):
            self.metadata = {"corpus_sha256": corpus_hash}
            self.vocabulary = ("chicken", "rice")
            self.n_recipes = 1
            self._metadata_index = object()

    def cases(finder, count, seed, partition):
        events.append(f"generate_{partition}")
        return [(0, evaluation.RecipeQuery(["chicken", "rice"]))] * count

    def evaluate(finders, generated, flat, offsets, *, phase):
        events.append(f"evaluate_{phase}")
        return {"heuristic": [observation(1)] * len(generated)}

    original_select = evaluation._select_policy

    def select(*args, **kwargs):
        events.append("select")
        return original_select(*args, **kwargs)

    monkeypatch.setattr(evaluation, "RecipeFinder", Finder)
    monkeypatch.setattr(evaluation, "PATHS", SimpleNamespace(recipes=tmp_path))
    monkeypatch.setattr(evaluation, "_query_cases", cases)
    monkeypatch.setattr(evaluation, "_evaluate_cases", evaluate)
    monkeypatch.setattr(evaluation, "_select_policy", select)
    output = tmp_path / "evaluation.json"
    arguments = ["evaluate_recipe_search.py", "--queries", "10", "--out", str(output)]
    if validation_only:
        arguments.append("--validation-only")
    monkeypatch.setattr(sys, "argv", arguments)
    assert evaluation.main() == 0
    expected = ["generate_validation", "evaluate_validation", "select"]
    if not validation_only:
        expected.extend(["generate_test", "evaluate_test"])
    assert events == expected
    report = json.loads(output.read_text())
    assert report["selection_frozen_before_test_scoring"] is True
    assert report["test_scored"] is not validation_only
    assert report["release_evaluation"] is False


def test_live_queries_use_the_same_held_out_pantry_hashes_as_training(evaluation, tmp_path):
    path = tmp_path / "catalog.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE recipes(id INTEGER PRIMARY KEY, ingredient_ids BLOB, "
            "total_minutes REAL, servings REAL, language TEXT, title TEXT, steps TEXT, "
            "text_status TEXT)")
        rng = np.random.default_rng(21)
        for recipe_id in range(64):
            ids = np.sort(rng.choice(40, 8, replace=False)).astype("<u2")
            connection.execute("INSERT INTO recipes VALUES (?,?,?,?,?,?,?,?)", (
                recipe_id, ids.tobytes(), 20.5, 2, "en", "Fixture recipe",
                "Fixture instruction.", ""))
    finder = SimpleNamespace(
        catalog_path=path, n_recipes=64,
        vocabulary=tuple(f"ingredient_{value}" for value in range(40)))
    pantries = set()
    for partition in ("validation", "test"):
        cases = evaluation._query_cases(finder, 10, 42, partition)
        assert evaluation._case_fingerprint(cases) == evaluation._case_fingerprint(
            evaluation._query_cases(finder, 10, 42, partition))
        assert len(cases) == 10
        assert len({recipe_id for recipe_id, _ in cases}) == 10
        assert sum(query.max_total_minutes is not None for _, query in cases) == 5
        for _, query in cases:
            ids = np.asarray([finder.vocabulary.index(name)
                              for name in query.available_ingredients], dtype=np.uint16)
            signature = evaluation.pantry_signature(ids)
            assert evaluation.query_partition(signature) == partition
            assert signature not in pantries
            pantries.add(signature)
    with pytest.raises(ValueError, match="held-out"):
        evaluation._query_cases(finder, 10, 42, "train")
