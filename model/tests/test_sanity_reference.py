"""Core sanity checks must not depend on an unshipped legacy run."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts" / "sanity_check.py"
SPEC = importlib.util.spec_from_file_location("sanity_check", PATH)
assert SPEC is not None and SPEC.loader is not None
sanity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sanity)


def record(path):
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps({
        "run_id": path.name, "model": "ease", "family": "recipe_basket",
        "graph": "ii_graph_rh_train.npz", "seed": 0,
        "params": {"split": "recipe-holdout"}, "created": "fixture",
        "duration_s": 1, "shape": [2, 2],
    }))


def test_sanity_selects_the_declared_benchmark_not_legacy_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(sanity, "PATHS", SimpleNamespace(runs=tmp_path))
    monkeypatch.setattr(
        sanity, "load_workspace", lambda: SimpleNamespace(benchmark_sweep="published"))
    record(tmp_path / "ease-rh")
    with pytest.raises(ValueError, match="found 0"):
        sanity.benchmark_run("ease")
    current = tmp_path / "published" / "ease-current"
    record(current)
    assert sanity.benchmark_run("ease") == current
    record(tmp_path / "published" / "ease-another")
    with pytest.raises(ValueError, match="found 2"):
        sanity.benchmark_run("ease")
