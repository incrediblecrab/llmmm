from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
hub = pytest.importorskip("huggingface_hub")

from ingredient_model import ingredient_demo as demo


@pytest.fixture
def model_binding(tmp_path, monkeypatch):
    configuration = {
        "schema_version": 1, "repo_id": demo.MODEL_REPOSITORY, "selected_policy": "supervised",
        "corpus_sha256": "a" * 64, "evaluated_catalog_sha256": "b" * 64,
    }
    path = tmp_path / "recipe_search_config.json"
    path.write_text(json.dumps(configuration))
    record = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    receipt = tmp_path / "model/results/huggingface_recipe_search_release.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"files": {"recipe_search_config.json": record}}))
    source = {"revision": "c" * 40, "files": {}}
    monkeypatch.setattr(demo, "load_public_policy",
                        lambda repository, **kwargs: (["egg", "milk"], object(), source))

    def download(repository, filename, **kwargs):
        assert repository == demo.MODEL_REPOSITORY
        assert filename == "recipe_search_config.json"
        assert kwargs["revision"] == source["revision"]
        assert kwargs["token"] is False
        return str(path)

    monkeypatch.setattr(hub, "hf_hub_download", download)
    metadata = {
        "vocabulary": ["egg", "milk"],
        "identity": {"corpus_sha256": "a" * 64, "catalog_sha256": "b" * 64},
    }
    return tmp_path, path, metadata


def test_full_browser_policy_is_bound_to_the_published_search_inputs(model_binding):
    root, _, metadata = model_binding
    vocabulary, _, source = demo.load_ingredient_policy(root, metadata)
    assert vocabulary == metadata["vocabulary"]
    assert source["corpus_sha256"] == metadata["identity"]["corpus_sha256"]
    assert "recipe_search_config.json" in source["files"]


@pytest.mark.parametrize("field", ["vocabulary", "corpus_sha256", "catalog_sha256"])
def test_changed_vocabulary_or_input_identity_is_rejected(model_binding, field):
    root, _, metadata = model_binding
    if field == "vocabulary":
        metadata[field] = ["egg", "salt"]
    else:
        metadata["identity"][field] = "d" * 64
    with pytest.raises(ValueError, match="differs from the released"):
        demo.load_ingredient_policy(root, metadata)


def test_configuration_corruption_is_not_treated_as_a_different_dataset(model_binding):
    root, path, metadata = model_binding
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="integrity"):
        demo.load_ingredient_policy(root, metadata)


def test_independent_reference_applies_constraints_before_shortlisting(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    reference = importlib.import_module("verify_ingredient_catalog").reference_search
    metadata = {
        "vocabulary": ["egg", "milk", "salt"], "n_recipes": 3,
        "ingredient_frequency": [2, 1, 1], "language_names": ["en"],
    }
    arrays = {
        "ingredients": np.array([0, 1, 0, 2], dtype="<u2"),
        "lengths": np.array([2, 1, 1], dtype="<u2"),
        "total_minutes": np.array([5, np.nan, 10]),
        "servings": np.array([2, np.nan, np.nan]),
        "has_source_url": np.array([1, 0, 1], dtype="u1"),
    }
    policy = demo.RecipeRankingPolicy()
    query = {"available_ingredients": ["egg"], "max_missing": 2}
    result = reference(metadata, arrays, query, policy, max_candidates=1)
    assert result["feasible_count"] == 2
    assert result["candidates_scored"] == 1
    assert result["learned"]["ids"] == result["heuristic"]["ids"] == [1]
    result = reference(metadata, arrays, {**query, "require_source_url": True}, policy)
    assert result["feasible_count"] == 1
    assert result["learned"]["ids"] == result["heuristic"]["ids"] == [0]
    result = reference(metadata, arrays, {**query, "max_total_minutes": 0}, policy)
    assert result["feasible_count"] == result["candidates_scored"] == 0
    assert result["learned"]["ids"] == result["heuristic"]["ids"] == []


@pytest.mark.parametrize("arguments", [
    {"source_revision": "main"},
    {"dataset_revision": "main"},
    {"dataset_revision": "a" * 40},
])
def test_public_preview_requires_immutable_source_and_dataset_pins(tmp_path, arguments):
    with pytest.raises(ValueError, match="commit|pinned source"):
        demo.build_ingredient_demo(tmp_path, tmp_path / "index", tmp_path / "out", **arguments)
    assert not (tmp_path / "out").exists()


def test_model_card_update_only_replaces_the_existing_demo_section():
    card = "Before\n<!-- PUBLIC-DEMO:START -->\nOld sample description.\n<!-- PUBLIC-DEMO:END -->\nAfter"
    updated = demo.add_ingredient_demo_links(card)
    assert updated.startswith("Before\n")
    assert updated.endswith("\nAfter")
    assert "4,653,430" in updated
    assert demo.INGREDIENT_DATASET_REPOSITORY in updated
    assert demo.DATASET_REPOSITORY in updated
    assert "not measure this browser retrieval" in updated
    assert demo.add_ingredient_demo_links(updated) == updated
    with pytest.raises(ValueError, match="single demo-link section"):
        demo.add_ingredient_demo_links("No replacement boundary")
