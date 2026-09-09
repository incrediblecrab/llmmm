"""Native seed evidence must come from recorded trials, not the report's metric choice."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / "scripts" / "ranking_stability.py"
SPEC = importlib.util.spec_from_file_location("ranking_stability", PATH)
assert SPEC is not None and SPEC.loader is not None
stability = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stability)


def trial(root, model, seed, score):
    directory = root / f"{model}-{seed}"
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps({"model": model, "seed": seed}))
    (directory / "metrics.json").write_text(json.dumps({
        "M6_recall_at_10": 0.1,
        "M6_native_recall_at_10": score,
        "M6_popularity_recall_at_10": 0.3,
    }))


def test_native_count_uses_each_seed_instead_of_subtracting_a_constant(tmp_path):
    trial(tmp_path, "ease", 0, 0.6)
    trial(tmp_path, "ease", 1, 0.6)
    trial(tmp_path, "masked-set", 0, 0.7)
    trial(tmp_path, "masked-set", 1, 0.25)
    embeddings = {"ease": {0: 0.1, 1: 0.1},
                  "masked-set": {0: 0.1, 1: 0.1},
                  "svd": {0: 0.4, 1: 0.4}}
    result = stability.native_seed_summary(
        tmp_path, [{"model": "ease"}, {"model": "masked-set"}],
        embeddings, {0: 0.3, 1: 0.3}, list(embeddings), [0, 1])
    assert result["complete"]
    assert result["served_n_below_popularity_by_seed"] == {"0": 0, "1": 1}


def test_missing_native_seed_does_not_fall_back_to_embedding_or_seed_zero(tmp_path):
    trial(tmp_path, "ease", 0, 0.6)
    result = stability.native_seed_summary(
        tmp_path, [{"model": "ease"}], {"ease": {0: 0.1, 1: 0.1}},
        {0: 0.3, 1: 0.3}, ["ease"], [0, 1])
    assert not result["complete"]
    assert result["models"][0]["missing_seeds"] == [1]
    assert result["served_n_below_popularity_by_seed"] == {"0": 0, "1": None}
