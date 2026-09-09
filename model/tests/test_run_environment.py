"""Run provenance records the promoted corpus digest without rehashing data."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from ingredient_model import artifacts, config
from ingredient_model.spec import ModelSpec, TrainResult


@pytest.fixture
def generation_file(tmp_path, monkeypatch):
    path = tmp_path / "data" / "GENERATION.json"
    path.parent.mkdir()
    paths = SimpleNamespace(generation_file=path)
    monkeypatch.setattr(config, "PATHS", paths)
    monkeypatch.setattr(artifacts, "PATHS", paths)
    monkeypatch.setattr(artifacts, "subprocess", SimpleNamespace(
        DEVNULL=artifacts.subprocess.DEVNULL,
        check_output=lambda *args, **kwargs: "test-git\n"))
    return path


@pytest.mark.parametrize("digest", ["12" * 32, "ab" * 32])
def test_saved_manifest_records_canonical_corpus_digest(
        tmp_path, generation_file, digest):
    generation_file.write_text(json.dumps({
        "generation": "v2", "corpus": "recipe_ids.npz", "sha256": digest}))
    result = TrainResult(np.eye(3, dtype=np.float32))

    def no_training(ctx):
        pytest.fail("recording provenance must not train")

    run = artifacts.save_run(
        "fixture", ModelSpec("svd-ppmi", "matrix", no_training), result,
        graph="test.npz", seed=42, params={}, duration_s=0,
        out_dir=tmp_path / "run")

    environment = artifacts.Manifest.load(run).environment
    assert environment["corpus_generation"] == "v2"
    assert environment["corpus_sha256"] == digest
    assert not (generation_file.parent / "recipes" / "recipe_ids.npz").exists()


@pytest.mark.parametrize("generation", [None, "v1"])
def test_legacy_or_unpromoted_corpus_does_not_invent_a_digest(
        generation_file, generation):
    if generation is not None:
        generation_file.write_text(json.dumps({"generation": generation}))
    environment = artifacts._environment()
    assert environment["corpus_generation"] == (generation or "unknown")
    assert "corpus_sha256" not in environment


@pytest.mark.parametrize("digest", [None, "", 123, "abc", "g" * 64])
def test_invalid_declared_digest_is_not_silently_dropped(generation_file, digest):
    generation_file.write_text(json.dumps({
        "generation": "v2", "corpus": "recipe_ids.npz", "sha256": digest}))
    with pytest.raises(
            ValueError, match="GENERATION.json: invalid canonical corpus SHA-256"):
        artifacts._environment()
