"""Workspace declarations must select real inputs, not directory-order defaults."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ingredient_model import artifacts, experiments, workspace
from ingredient_model.reasoning import reasoner
from ingredient_model.spec import ModelSpec, TrainResult


def test_workspace_requires_declared_relative_paths(tmp_path):
    config = {
        "version": 1,
        "benchmark_sweep": "benchmark",
        "training_sweep": "current",
        "training_experiment": "experiments/current.yaml",
        "default_embedding_run": "benchmark/svd",
    }
    path = tmp_path / "workspace.json"
    path.write_text(json.dumps(config))
    assert workspace.load_workspace(tmp_path).default_embedding_run == "benchmark/svd"
    config["training_experiment"] = "../outside.yaml"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="relative workspace path"):
        workspace.load_workspace(tmp_path)


def test_training_checks_corpus_bytes_not_just_generation_label(tmp_path):
    recipes = tmp_path / "recipes"
    recipes.mkdir()
    corpus = recipes / "recipe_ids.npz"
    corpus.write_bytes(b"declared corpus")
    marker = {
        "generation": "v2",
        "corpus": "recipe_ids.npz",
        "sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
    }
    (tmp_path / "GENERATION.json").write_text(json.dumps(marker))
    assert workspace.verify_training_corpus(tmp_path, "v2") == marker
    with pytest.raises(ValueError, match="requires generation"):
        workspace.verify_training_corpus(tmp_path, "v1")
    corpus.write_bytes(b"different corpus")
    with pytest.raises(ValueError, match="checksum differs"):
        workspace.verify_training_corpus(tmp_path, "v2")


def test_reasoner_uses_the_declared_embedding_and_explicit_statistics_mode(
        monkeypatch, tmp_path):
    selected = []
    monkeypatch.setattr(
        workspace, "load_workspace",
        lambda: SimpleNamespace(default_embedding_run="benchmark/svd"))

    def resolve(ref):
        selected.append(ref)
        return tmp_path

    monkeypatch.setattr(reasoner, "resolve_run", resolve)
    monkeypatch.setattr(
        reasoner.Manifest, "load",
        lambda path: SimpleNamespace(run_id="declared-svd", params={}))
    monkeypatch.setattr(reasoner, "load_embedding", lambda path: np.eye(2))
    graph = SimpleNamespace(
        itos=["tomato", "basil"], src=np.array([0]), dst=np.array([1]),
        npmi=np.array([0.2]), count=np.array([5]))
    monkeypatch.setattr(reasoner, "load_ii_graph", lambda name: graph)
    monkeypatch.setattr(reasoner, "load_chem_graph", lambda: None)

    assert reasoner.Reasoner.load().run_id == "declared-svd"
    assert selected == ["benchmark/svd"]
    control = reasoner.Reasoner.load(statistics_only=True)
    assert control.W is None
    assert selected == ["benchmark/svd"]
    with pytest.raises(ValueError, match="cannot be combined"):
        reasoner.Reasoner.load(tmp_path, statistics_only=True)


def test_sweeps_record_resolved_parameters(monkeypatch, tmp_path):
    model = ModelSpec(
        name="tiny", family="test", defaults={"epochs": 3, "reg": 500},
        train=lambda ctx: TrainResult(embedding=np.eye(2)))
    monkeypatch.setattr(experiments, "discover", lambda: None)
    monkeypatch.setattr(experiments, "get", lambda name: model)
    monkeypatch.setattr(experiments, "corpus_generation", lambda: {"generation": "v2"})
    monkeypatch.setattr(experiments, "PATHS", SimpleNamespace(runs=tmp_path / "runs"))
    monkeypatch.setattr(experiments, "held_out_recipes", lambda split: None)
    monkeypatch.setattr(experiments, "build_context", lambda split: None)
    monkeypatch.setattr(experiments, "evaluate", lambda *args, **kw: {"M6_n": 0})
    monkeypatch.setattr(experiments, "render_one", lambda *args: "")
    monkeypatch.setattr(experiments, "leaderboard", lambda **kw: "")
    monkeypatch.setattr(artifacts, "_environment", lambda: {})
    declaration = tmp_path / "experiment.json"
    declaration.write_text(json.dumps({
        "name": "test", "models": [{"name": "tiny", "params": {"reg": 700}}],
    }))
    assert experiments.run_experiment(declaration) == 0
    manifests = list((tmp_path / "runs").rglob("manifest.json"))
    assert len(manifests) == 1
    saved = json.loads(manifests[0].read_text())
    assert saved["params"] == {"epochs": 3, "reg": 700, "split": "recipe-holdout"}
