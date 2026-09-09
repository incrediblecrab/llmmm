"""Evidence checks for an all-record, training-only ingredient checkpoint."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .artifacts import (Manifest, load_embedding, load_metrics, load_native_scorer,
                        unevaluated_metrics)
from .workspace import verify_training_corpus


def verify_full_training(run_dir: Path, data_root: Path, *,
                         expected_recipes: int, generation: str = "v2") -> dict:
    manifest = Manifest.load(run_dir)
    params, metadata = manifest.params, manifest.metadata
    if (manifest.model != "masked-set" or params.get("split") != "full"
            or params.get("no_eval") is not True):
        raise ValueError("all-record verification requires a full, training-only checkpoint")
    if load_metrics(run_dir) != unevaluated_metrics("full"):
        raise ValueError("missing or invalid training-only completion record")

    marker = verify_training_corpus(data_root, generation)
    if (manifest.environment.get("corpus_sha256") != marker["sha256"]
            or manifest.environment.get("corpus_generation") != generation):
        raise ValueError("checkpoint and verified corpus identities differ")
    with np.load(data_root / "recipes" / "recipe_ids.npz", allow_pickle=False) as stored:
        offsets, flat = stored["offsets"], stored["flat"]
        sizes = np.diff(offsets)
        if (offsets.ndim != 1 or offsets[0] != 0 or not len(sizes)
                or np.any(sizes <= 0) or int(offsets[-1]) != flat.size):
            raise ValueError("invalid canonical recipe offsets or ingredient-slot count")
        recipes, slots = len(sizes), int(flat.size)
        minimum, maximum = int(sizes.min()), int(sizes.max())
    if (recipes != expected_recipes or marker["recipes"] != recipes
            or marker["slots"] != slots):
        raise ValueError("measured corpus counts differ from the required population")
    for key, value in {
        "max_recipes": 0, "min_len": 1, "max_len": None,
        "expected_recipes": expected_recipes,
    }.items():
        if (key not in params or key not in metadata
                or params[key] != value or metadata[key] != value):
            raise ValueError(f"{key}: the checkpoint did not declare unfiltered all-record training")

    epochs, batch_size = params["epochs"], params["batch_size"]
    if type(epochs) is not int or epochs < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive integers")
    if metadata.get("epochs") != epochs or metadata.get("batch_size") != batch_size:
        raise ValueError("recorded epochs or batch size differ from the training parameters")
    required = {
        "n_recipes": recipes,
        "n_eligible_recipes": recipes,
        "n_examples_seen": recipes * epochs,
        "n_ingredient_slots_seen": slots * epochs,
        "n_optimizer_steps": ((recipes + batch_size - 1) // batch_size) * epochs,
        "observed_min_len": minimum,
        "observed_max_len": maximum,
    }
    for key, value in required.items():
        if type(metadata.get(key)) is not int or metadata[key] != value:
            raise ValueError(f"{key}: recorded {metadata.get(key)!r}, expected {value}")
    coverage = metadata.get("epoch_coverage")
    if not isinstance(coverage, list) or len(coverage) != epochs:
        raise ValueError("one coverage record is required for every completed epoch")
    for epoch, record in enumerate(coverage, 1):
        expected = {
            "epoch": epoch, "unique_recipes": recipes, "examples_seen": recipes,
            "ingredient_slots_seen": slots, "every_eligible_recipe_once": True,
        }
        if record != expected or record["every_eligible_recipe_once"] is not True:
            raise ValueError(f"epoch {epoch}: every record and ingredient slot must be used once")
    losses = metadata.get("loss_history")
    if not isinstance(losses, list) or len(losses) != epochs or not np.isfinite(losses).all():
        raise ValueError("every completed epoch must have a finite recorded training loss")
    weights = load_embedding(run_dir)
    if (weights.ndim != 2 or weights.shape != manifest.shape
            or weights.shape[0] != marker["vocab"]
            or not np.isfinite(weights).all()):
        raise ValueError("saved embedding does not match the verified vocabulary and manifest")
    if load_native_scorer(run_dir, marker["vocab"]) is None:
        raise ValueError("the complete trained predictor could not be restored")
    files = [run_dir / "manifest.json", run_dir / "metrics.json", run_dir / "embedding.npy",
             *sorted(run_dir.glob("state__*.npy"))]
    hashes = {}
    for path in files:
        with path.open("rb") as stream:
            hashes[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "status": "verified_all_record_training",
        "run_id": manifest.run_id,
        "corpus_generation": generation,
        "corpus_sha256": marker["sha256"],
        "recipes_per_epoch": recipes,
        "ingredient_slots_per_epoch": slots,
        "epochs": epochs,
        "example_presentations": recipes * epochs,
        "ingredient_slot_presentations": slots * epochs,
        "optimizer_steps": required["n_optimizer_steps"],
        "minimum_recipe_length": minimum,
        "maximum_recipe_length": maximum,
        "epoch_coverage": coverage,
        "complete_predictor_restored": True,
        "evaluation_status": "not_run",
        "training_code_sha256": metadata["training_code_sha256"],
        "artifact_sha256": hashes,
    }
