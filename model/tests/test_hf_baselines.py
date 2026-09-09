from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from ingredient_model.config import REPO
from ingredient_model.hf_baselines import (
    align_embeddings, ensure_model, epicure_embeddings, group_scores, load_pins,
    mean_pool, verify_file,
)


def test_pins_include_all_requested_tasks():
    pins = load_pins()
    assert len(pins) == 4
    assert {spec["role"] for spec in pins.values()} == {
        "ingredient_embeddings", "food_text_encoder", "recipe_generator",
    }


@pytest.mark.parametrize("algorithm", ["sha256", "git_blob_sha1"])
def test_asset_identity_is_checked_not_just_shape(tmp_path, algorithm):
    data = b"actual model asset"
    digest = (hashlib.sha256(data).hexdigest() if algorithm == "sha256" else
              hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest())
    pin = {"bytes": len(data), algorithm: digest}
    path = tmp_path / "asset"
    path.write_bytes(data)
    verify_file(path, pin)
    path.write_bytes(b"X" * len(data))
    with pytest.raises(ValueError, match="digest"):
        verify_file(path, pin)


@pytest.mark.parametrize("name", ["../model.bin", "/model.bin", "a//b", "a/./b"])
def test_lock_rejects_nonlocal_asset_paths(tmp_path, name):
    pin = {"bytes": 1, "sha256": "a" * 64}
    document = {"version": 1, "models": {
        "owner/model": {"revision": "a" * 40, "files": {name: pin}},
    }}
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="unsafe"):
        load_pins(path)


def test_alignment_uses_ingredient_identity_not_position():
    matrix = np.array([[1, 0], [0, 1]], dtype=np.float32)
    got = align_embeddings(matrix, ["b", "a"], ["a", "b"])
    np.testing.assert_array_equal(got, matrix[::-1])
    with pytest.raises(ValueError, match="missing"):
        align_embeddings(matrix, ["a", "b"], ["a", "c"])
    with pytest.raises(ValueError, match="unique"):
        align_embeddings(matrix, ["a", "a"], ["a", "b"])


def test_mean_pool_ignores_padding_but_keeps_real_special_tokens():
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[[1., 3.], [3., 5.], [999., 999.]]])
    mask = torch.tensor([[1, 1, 0]])
    np.testing.assert_array_equal(mean_pool(hidden, mask).numpy(), [[2, 4]])
    with pytest.raises(ValueError, match="no attended"):
        mean_pool(hidden, torch.zeros_like(mask))


def test_group_scores_keep_the_denominators_visible():
    scores = group_scores(np.array([1., 11., 2.]), np.array(["en", "en", "zh"]))
    assert scores["en"]["n"] == 2
    assert scores["en"]["recall_at_10"] == 0.5
    assert scores["zh"]["n"] == 1
    assert scores["zh"]["recall_at_10"] == 1.0
    with pytest.raises(ValueError, match="align"):
        group_scores(np.array([1., 2.]), np.array(["en"]))


def test_cache_rejects_corruption_instead_of_silently_replacing_it(tmp_path, monkeypatch):
    import ingredient_model.hf_baselines as module

    data = b"verified"
    spec = {"revision": "a" * 40, "files": {
        "asset": {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()},
    }}
    monkeypatch.setattr(module, "load_pins", lambda _path: {"owner/model": spec})
    source = tmp_path / "source"
    source.mkdir()
    (source / "asset").write_bytes(data)
    cache = tmp_path / "cache"
    with pytest.raises(FileNotFoundError, match="--download"):
        ensure_model("owner/model", cache=cache)
    folder = ensure_model("owner/model", cache=cache, local_source=source)
    assert (folder / "asset").read_bytes() == data
    (folder / "asset").write_bytes(b"corrupt!")
    with pytest.raises(ValueError, match="digest"):
        ensure_model("owner/model", cache=cache, local_source=source)
    assert (folder / "asset").read_bytes() == b"corrupt!"


def test_epicure_index_mapping_is_not_dict_insertion_order(tmp_path):
    tensors = pytest.importorskip("safetensors.numpy")
    matrix = np.array([[1, 0], [0, 1]], dtype=np.float32)
    tensors.save_file({"embeddings": matrix}, tmp_path / "embeddings.safetensors")
    path = tmp_path / "itos.json"
    path.write_text(json.dumps({"1": "b", "0": "a"}))
    np.testing.assert_array_equal(epicure_embeddings(tmp_path, ["b", "a"]), matrix[::-1])
    path.write_text(json.dumps({"0": "a", "2": "b"}))
    with pytest.raises(ValueError, match="contiguous"):
        epicure_embeddings(tmp_path, ["a", "b"])


@pytest.mark.parametrize("script", ["compare_hf_baselines.py", "audit_generation_data.py"])
def test_report_cli_rejects_conflicting_output_extensions(tmp_path, script):
    output = tmp_path / "report.csv"
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / script), "--out", str(output)],
        capture_output=True, text=True)
    assert result.returncode != 0
    assert "--out must end in .json" in result.stderr
    assert not output.exists()
