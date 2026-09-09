from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from ingredient_model import artifacts, cli, experiments
from ingredient_model.data.splits import LeakageError, check_training_protocol, get_split
from ingredient_model.eval.report import collect, render_one
from ingredient_model.spec import ModelSpec, TrainResult


def test_full_training_requires_no_evaluation():
    for requires in (("recipes",), ("ii_graph_train",)):
        with pytest.raises(LeakageError, match="no held-out"):
            check_training_protocol(get_split("full"), requires)
        assert check_training_protocol(
            get_split("full"), requires, no_eval=True) is None
    with pytest.raises(LeakageError):
        check_training_protocol(
            get_split("edge-holdout"), ("recipes",), no_eval=True)


def test_production_sweep_never_loads_evaluation_data(tmp_path, monkeypatch):
    calls = []

    def train(context):
        calls.append(context)
        assert context.corpus == "recipe_ids.npz"
        assert context.split == "full"
        return TrainResult(np.eye(3, dtype=np.float32))

    def forbidden(*args, **kwargs):
        pytest.fail("a train-only run must not invoke evaluation")

    model = ModelSpec(name="tiny", family="test", requires=("recipes",), train=train)
    monkeypatch.setattr(experiments, "discover", lambda: None)
    monkeypatch.setattr(experiments, "get", lambda _: model)
    monkeypatch.setattr(experiments, "corpus_generation", lambda: {"generation": "v2"})
    monkeypatch.setattr(experiments, "PATHS", SimpleNamespace(runs=tmp_path / "runs"))
    monkeypatch.setattr(experiments, "held_out_recipes", forbidden)
    monkeypatch.setattr(experiments, "build_context", forbidden)
    monkeypatch.setattr(experiments, "evaluate", forbidden)
    monkeypatch.setattr(artifacts, "_environment", lambda: {})
    declaration = tmp_path / "production.json"
    declaration.write_text(json.dumps({
        "name": "production", "models": ["tiny"], "splits": ["full"],
        "no_eval": True,
    }))
    assert experiments.run_experiment(declaration) == 0
    assert len(calls) == 1
    run, = artifacts.iter_runs(tmp_path / "runs")
    metrics = artifacts.load_metrics(run)
    assert metrics == {"split": "full", "evaluation_status": "not_run"}
    assert artifacts.Manifest.load(run).params["no_eval"] is True
    assert not collect(tmp_path / "runs")
    assert "Evaluation was not run" in render_one("production", metrics)
    assert experiments.run_experiment(declaration) == 0
    assert len(calls) == 1
    (run / artifacts.METRICS).write_text('{"M6_native_recall_at_10": 1.0}')
    with pytest.raises(ValueError, match="invalid training-only completion"):
        experiments.run_experiment(declaration)
    assert len(calls) == 1


@pytest.mark.parametrize("array_path", [False, True])
def test_production_checkpoint_cannot_be_rescored_on_its_training_rows(
        tmp_path, monkeypatch, array_path):
    monkeypatch.setattr(artifacts, "_environment", lambda: {})
    model = ModelSpec(name="tiny", family="test", train=lambda _: None)
    run = artifacts.save_run(
        "production", model, TrainResult(np.eye(3)), graph="ii_graph.npz",
        seed=1, params={"split": "full", "no_eval": True},
        duration_s=0, out_dir=tmp_path / "run")
    artifacts.save_metrics(run, artifacts.unevaluated_metrics("full"))
    before = (run / "metrics.json").read_bytes()
    target = run / "embedding.npy" if array_path else run
    args = cli.build_parser().parse_args([
        "eval", str(target), "--split", "recipe-holdout",
    ])
    with pytest.raises(ValueError, match="trained on the full corpus"):
        cli.cmd_eval(args)
    assert (run / "metrics.json").read_bytes() == before


def test_cli_full_training_saves_an_explicit_unscored_completion(tmp_path, monkeypatch):
    def train(context):
        assert context.corpus == "recipe_ids.npz"
        assert context.split == "full"
        return TrainResult(np.eye(3, dtype=np.float32))

    def forbidden(*args, **kwargs):
        pytest.fail("a train-only run must not invoke evaluation")

    model = ModelSpec(name="tiny", family="test", requires=("recipes",), train=train)
    monkeypatch.setattr(cli, "get", lambda _: model)
    monkeypatch.setattr(cli, "check_available", lambda _: [])
    monkeypatch.setattr(cli, "held_out_recipes", forbidden)
    monkeypatch.setattr(cli, "build_context", forbidden)
    monkeypatch.setattr(cli, "evaluate", forbidden)
    monkeypatch.setattr(artifacts, "_environment", lambda: {})
    run = tmp_path / "direct"
    args = cli.build_parser().parse_args([
        "train", "tiny", "--split", "full", "--no-eval", "--out", str(run),
    ])
    assert cli.cmd_train(args) == 0
    assert artifacts.load_metrics(run) == {
        "split": "full", "evaluation_status": "not_run",
    }
    assert artifacts.Manifest.load(run).params["no_eval"] is True


def test_no_length_cap_is_representable_in_cli_parameters():
    assert cli._parse_set(["max_len=null", "min_len=1"]) == {
        "max_len": None, "min_len": 1,
    }
