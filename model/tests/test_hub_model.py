from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("huggingface_hub")
pytest.importorskip("safetensors")

import torch
from safetensors.torch import load_file, save_file

from ingredient_model.hub import IngredientPredictor
from models.set_transformer.train import _make_scorer


def model():
    return IngredientPredictor(
        ["a", "b", "c", "d", "e"],
        dict(d_model=8, n_heads=2, n_layers=1, ff_mult=2, dropout=0.0, tie_output=True)).eval()


def test_hub_export_reloads_the_complete_predictor(tmp_path):
    original = model()
    contexts = np.array([[0, 1], [2, 3]], dtype=np.int64)
    expected = _make_scorer(original.network, 5, "cpu")(contexts)
    original.save_pretrained(tmp_path)
    restored = IngredientPredictor.from_pretrained(tmp_path, local_files_only=True)
    with torch.inference_mode():
        actual = restored(torch.from_numpy(contexts)).numpy()
    np.testing.assert_array_equal(actual, expected)
    assert restored.vocabulary == original.vocabulary
    assert not (tmp_path / "pytorch_model.bin").exists()


def test_inference_excludes_context_and_rejects_unknowns():
    predictor = model()
    result = predictor.recommend(["a", "b"], top_k=3)
    assert len(result) == 3
    assert {item["ingredient"] for item in result} == {"c", "d", "e"}
    with pytest.raises(ValueError, match="unknown"):
        predictor.recommend(["a", "not-in-vocabulary"])
    with pytest.raises(ValueError, match="distinct"):
        predictor.recommend(["a", "a"])
    with pytest.raises(ValueError, match="unique"):
        predictor(torch.tensor([[0, 0]]))


def test_missing_native_weights_cannot_load_a_random_partial_model(tmp_path):
    predictor = model()
    predictor.save_pretrained(tmp_path)
    path = tmp_path / "model.safetensors"
    state = load_file(path)
    state.pop("network.bias")
    save_file(state, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        IngredientPredictor.from_pretrained(tmp_path, local_files_only=True)
