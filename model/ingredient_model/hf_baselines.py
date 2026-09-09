"""Local, pinned upstream assets; no remote inference or repository code."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np

from .config import PATHS, REPO

LOCK = REPO / "hf_baselines.lock.json"


def load_pins(path: Path = LOCK) -> dict:
    document = json.loads(path.read_text())
    if document.get("version") != 1 or not isinstance(document.get("models"), dict):
        raise ValueError(f"{path}: unsupported upstream model lock")
    for model_id, spec in document["models"].items():
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", model_id):
            raise ValueError(f"invalid model identity: {model_id!r}")
        if not re.fullmatch(r"[0-9a-f]{40}", spec["revision"]):
            raise ValueError(f"{model_id}: a full immutable revision is required")
        if not spec["files"]:
            raise ValueError(f"{model_id}: no pinned assets")
        for name, pin in spec["files"].items():
            if (not re.fullmatch(r"[\w./-]+", name)
                    or name.startswith("/") or any(
                        part in ("", ".", "..") for part in name.split("/"))):
                raise ValueError(f"{model_id}: unsafe asset name {name!r}")
            if type(pin.get("bytes")) is not int or pin["bytes"] < 0:
                raise ValueError(f"{model_id}/{name}: invalid asset size")
            digests = [key for key in ("sha256", "git_blob_sha1") if key in pin]
            if len(digests) != 1:
                raise ValueError(f"{model_id}/{name}: exactly one digest is required")
            key = digests[0]
            width = 64 if key == "sha256" else 40
            if not re.fullmatch(rf"[0-9a-f]{{{width}}}", pin[key]):
                raise ValueError(f"{model_id}/{name}: invalid digest")
    return document["models"]


def verify_file(path: Path, pin: dict) -> None:
    size = path.stat().st_size
    if size != pin["bytes"]:
        raise ValueError(f"{path}: asset size differs from the upstream lock")
    key = "sha256" if "sha256" in pin else "git_blob_sha1"
    digest = hashlib.sha256() if key == "sha256" else hashlib.sha1()
    if key == "git_blob_sha1":
        digest.update(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != pin[key]:
        raise ValueError(f"{path}: asset digest differs from the upstream lock")


def ensure_model(model_id: str, *, download: bool = False,
                 cache: Path = PATHS.hf_cache, local_source: Path | None = None,
                 pins_file: Path = LOCK) -> Path:
    spec = load_pins(pins_file)[model_id]
    folder = cache / model_id.replace("/", "--") / spec["revision"]
    for name, pin in spec["files"].items():
        destination = folder / name
        if destination.exists():
            verify_file(destination, pin)
            continue
        source = None if local_source is None else local_source / name
        if (source is None or not source.exists()) and not download:
            raise FileNotFoundError(
                f"{destination}: missing pinned baseline; rerun with --download")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temp:
            staged = Path(temp.name)
        try:
            if source is not None and source.exists():
                verify_file(source, pin)
                shutil.copyfile(source, staged)
            else:
                url = f"https://huggingface.co/{model_id}/resolve/{spec['revision']}/{name}"
                print(f"Downloading pinned {model_id}/{name}", flush=True)
                subprocess.run(
                    ["curl", "--disable", "-4", "--proto", "=https", "--location",
                     "--connect-timeout", "10", "--max-time", "600", "--fail",
                     "--silent", "--show-error", "--output", str(staged), url],
                    check=True, timeout=610)
            verify_file(staged, pin)
            try:
                os.link(staged, destination)
            except FileExistsError:
                verify_file(destination, pin)
        finally:
            staged.unlink()
    return folder


def align_embeddings(matrix: np.ndarray, names: list[str],
                     vocabulary: list[str]) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] != len(names):
        raise ValueError("upstream embedding shape and vocabulary disagree")
    if len(set(names)) != len(names) or len(set(vocabulary)) != len(vocabulary):
        raise ValueError("ingredient identities must be unique")
    missing = set(vocabulary) - set(names)
    if missing:
        raise ValueError(f"upstream vocabulary is missing {len(missing)} canonical ingredients")
    if not np.isfinite(matrix).all():
        raise ValueError("upstream embedding contains non-finite values")
    index = {name: i for i, name in enumerate(names)}
    return matrix[[index[name] for name in vocabulary]].astype(np.float32)


def mean_pool(hidden, attention_mask):
    """Match single-example mean pooling while excluding batch padding."""
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    counts = mask.sum(dim=1)
    if (counts == 0).any():
        raise ValueError("cannot pool a token sequence with no attended tokens")
    return (hidden * mask).sum(dim=1) / counts


def epicure_embeddings(folder: Path, vocabulary: list[str]) -> np.ndarray:
    from safetensors.numpy import load_file

    identities = json.loads((folder / "itos.json").read_text())
    if isinstance(identities, dict):
        if set(identities) != {str(i) for i in range(len(identities))}:
            raise ValueError("upstream itos must have contiguous integer-string keys")
        names = [identities[str(i)] for i in range(len(identities))]
    elif isinstance(identities, list):
        names = identities
    else:
        raise ValueError("upstream itos must be an ordered list or index mapping")
    if not all(isinstance(name, str) for name in names):
        raise ValueError("upstream ingredient identities must be strings")
    matrix = load_file(folder / "embeddings.safetensors")["embeddings"]
    return align_embeddings(matrix, names, vocabulary)


def recipebert_embeddings(folder: Path, vocabulary: list[str],
                          batch_size: int = 64) -> np.ndarray:
    import torch
    from transformers import AutoTokenizer, BertModel

    from models.text_embedding.train import _readable

    if batch_size <= 0 or not vocabulary:
        raise ValueError("encoding requires a positive batch size and nonempty vocabulary")
    tokenizer = AutoTokenizer.from_pretrained(
        folder, local_files_only=True, trust_remote_code=False)
    model, loading = BertModel.from_pretrained(
        folder, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, add_pooling_layer=False, output_loading_info=True)
    if (loading["missing_keys"] or loading["mismatched_keys"]
            or loading.get("error_msgs")
            or any(not name.startswith("cls.predictions.")
                   for name in loading["unexpected_keys"])):
        raise ValueError(f"RecipeBERT encoder did not load completely: {loading}")
    model.eval()
    vectors = []
    texts = [_readable(name) for name in vocabulary]
    with torch.inference_mode():
        for start in range(0, len(vocabulary), batch_size):
            tokens = tokenizer(texts[start:start + batch_size],
                               padding=True, truncation=False, return_tensors="pt")
            if tokens["input_ids"].shape[1] > model.config.max_position_embeddings:
                raise ValueError("ingredient names exceed RecipeBERT's context limit")
            hidden = model(**tokens).last_hidden_state
            vectors.append(mean_pool(hidden, tokens["attention_mask"]).cpu().numpy())
    return align_embeddings(np.concatenate(vectors), vocabulary, vocabulary)


def group_scores(ranks: np.ndarray, labels: np.ndarray) -> dict:
    if ranks.ndim != 1 or labels.shape != ranks.shape or not len(ranks):
        raise ValueError("group labels and nonempty rank vectors must align")
    return {
        str(label): {
            "n": int((labels == label).sum()),
            "recall_at_10": float((ranks[labels == label] <= 10).mean()),
            "mrr": float((1 / ranks[labels == label]).mean()),
        }
        for label in np.unique(labels)
    }
