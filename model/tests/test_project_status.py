"""Current documentation must be bound to the right artifact, not a matching number."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "project_status.py"
SPEC = importlib.util.spec_from_file_location("project_status", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
status = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = status
SPEC.loader.exec_module(status)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def add_result(root, sweep, model, seed=0):
    directory = root / "model" / "results" / "runs" / sweep / f"{model}-recipe-holdout-s{seed}"
    write_json(directory / "manifest.json", {
        "model": model, "seed": seed, "duration_s": 1.234,
        "params": {"split": "recipe-holdout", "epochs": 3,
                   "max_recipes": 600000, "max_len": 32},
        "environment": {"corpus_generation": "v2"},
    })
    metrics = {
        "M6_n": 10, "M6_recall_at_10": 0.4,
        "M6_centred_recall_at_10": 0.45, "M6_popularity_recall_at_10": 0.3,
    }
    if model in ("ease", "masked-set"):
        metrics["M6_native_recall_at_10"] = 0.6
    write_json(directory / "metrics.json", metrics)
    return directory / "metrics.json"


@pytest.fixture
def checkout(tmp_path):
    model = tmp_path / "model"
    write_json(model / "workspace.json", {
        "version": 1, "benchmark_sweep": "benchmark", "training_sweep": "current",
        "training_experiment": "experiments/current.yaml",
        "default_embedding_run": "benchmark/svd-ppmi-recipe-holdout-s0",
    })
    generation = {
        "generation": "v2", "recipes": 20, "slots": 80, "vocab": 8,
        "sha256": "ab" * 32,
    }
    write_json(model / "data" / "GENERATION.json", generation)
    write_json(model / "results" / "corpus_stats.json", {
        "generation": generation, "n_sources": 1,
        "per_source": [{"source": "fixture", "kept": 20}],
    })
    for name, models, seeds in (
            ("benchmark", ["ease", "svd-ppmi"], [0]),
            ("current", ["ease", "masked-set"], [42])):
        write_json(model / "experiments" / f"{name}.yaml", {
            "name": name, "models": models, "seeds": seeds,
            "splits": ["recipe-holdout"],
        })
    add_result(tmp_path, "benchmark", "ease")
    add_result(tmp_path, "benchmark", "svd-ppmi")
    (tmp_path / "README.md").write_text(
        f"# Fixture\n\n{status.START}\npending\n{status.END}\n")
    return tmp_path


def test_status_works_from_metadata_without_weights(checkout):
    assert status.main(["--root", str(checkout), "--write"]) == 0
    assert status.main(["--root", str(checkout), "--check"]) == 0
    text = (checkout / "README.md").read_text()
    assert "20 recipes" in text
    assert "0/2 runs scored" in text
    assert not list(checkout.rglob("*.npy"))


def test_wrong_attribution_fails_even_when_the_number_exists_elsewhere(checkout):
    assert status.main(["--root", str(checkout), "--write"]) == 0
    path = checkout / "README.md"
    text = path.read_text()
    assert "| 0.6000 |" in text
    path.write_text(text.replace("| 0.6000 |", "| 0.4000 |", 1))
    assert status.main(["--root", str(checkout), "--check"]) == 1


def test_lost_native_metrics_cannot_be_published(checkout, capsys):
    path = checkout / "model/results/runs/benchmark/ease-recipe-holdout-s0/metrics.json"
    metrics = json.loads(path.read_text())
    del metrics["M6_native_recall_at_10"]
    write_json(path, metrics)
    assert status.main(["--root", str(checkout)]) == 1
    assert "missing native predictor score" in capsys.readouterr().err


def test_current_training_is_counted_only_after_scoring(checkout):
    add_result(checkout, "current", "ease", seed=42)
    text = status.render_status(checkout)
    assert "1/2 runs scored" in text
    add_result(checkout, "current", "masked-set", seed=42)
    text = status.render_status(checkout)
    assert "2/2 runs scored" in text
    assert "sampling cap **600,000 recipes**" in text


def test_mismatched_completion_draw_is_rejected(checkout):
    path = add_result(checkout, "current", "ease", seed=42)
    metrics = json.loads(path.read_text())
    metrics["M6_n"] = 5
    write_json(path, metrics)
    with pytest.raises(ValueError, match="completion draw"):
        status.render_status(checkout)


def test_missing_benchmark_record_is_not_a_smaller_successful_benchmark(checkout):
    path = checkout / "model/results/runs/benchmark/ease-recipe-holdout-s0/metrics.json"
    path.unlink()
    assert status.main(["--root", str(checkout)]) == 1


def test_duplicate_markers_are_rejected():
    with pytest.raises(ValueError, match="exactly one"):
        status.replace_block(status.START + status.START + status.END, "new")
