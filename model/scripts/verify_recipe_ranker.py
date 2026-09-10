"""Verify saved all-record ranking coverage and write public, metadata-only evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ingredient_model._hashing import file_sha256
from ingredient_model.config import PATHS
from ingredient_model.data.recipe_catalog import load_catalog_metadata
from export_recipe_search import training_code_references, verify_training


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=PATHS.recipes / "recipe_search.sqlite")
    parser.add_argument("--corpus", type=Path, default=PATHS.recipes / "recipe_ids.npz")
    parser.add_argument("--out", type=Path, default=PATHS.results / "recipe_ranker_training.json")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"{args.out}: verification records are immutable")
    metadata = load_catalog_metadata(args.catalog)
    if file_sha256(args.corpus) != metadata["corpus_sha256"]:
        raise ValueError("canonical corpus bytes differ from the catalog/training identity")
    with np.load(args.corpus, allow_pickle=False) as corpus:
        offsets, flat = corpus["offsets"], corpus["flat"]
        if (len(offsets) != metadata["n_recipes"] + 1
                or len(flat) != metadata["n_slots"] or offsets[0] != 0
                or offsets[-1] != len(flat) or (np.diff(offsets) <= 0).any()):
            raise ValueError("canonical row and ingredient-slot accounting is invalid")
    report = verify_training(args.run, metadata)
    report["training_code_references"] = training_code_references(
        report["run_configuration"]["code_sha256"])
    report["independent_coverage_verification"] = {
        "canonical_corpus_sha256_checked": True,
        "canonical_row_and_slot_counts_checked": True,
        "all_saved_per_row_counts_exactly_one_per_stage_epoch": True,
        "optimizer_steps_and_reinforce_action_counts_checked": True,
        "held_out_pantry_buckets_absent_from_training_counters": True,
        "checkpoint_reload_checked_by_this_command": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Verified full ranking coverage; metadata-only evidence: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
