"""Export and re-evaluate the full native model, never just its token vectors."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ingredient_model.artifacts import load_native_scorer
from ingredient_model.config import PATHS, REPO, SEED
from ingredient_model.data.graphs import load_ii_graph
from ingredient_model.data.splits import held_out_recipes
from ingredient_model.eval.completion import completion_ranks
from ingredient_model.hub import IngredientPredictor
from ingredient_model.workspace import verify_training_corpus
from m6_intervals import summarise_ranks

DEFAULT_CANDIDATE = "train-v2-full-20260909/masked-set-full-recipe-holdout-s42"
DEFAULT_REFERENCE = "train-v2-20260909/masked-set-recipe-holdout-s42"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=PATHS.results / "native_full_release.json")
    args = parser.parse_args()
    if args.report.suffix != ".json" or args.out.resolve() in args.report.resolve().parents:
        parser.error("--report must be a .json file outside the export directory")
    if args.out.exists():
        raise FileExistsError(f"{args.out}: exports are immutable; choose a new path")
    generation = verify_training_corpus(PATHS.data, "v2")
    candidate = PATHS.runs / args.candidate
    reference = PATHS.runs / args.reference
    manifest = json.loads((candidate / "manifest.json").read_text())
    if (manifest["model"] != "masked-set"
            or manifest["environment"]["corpus_sha256"] != generation["sha256"]
            or manifest["params"]["split"] != "recipe-holdout"):
        raise ValueError("the candidate is not the expected corpus-pinned native model")
    corpus = held_out_recipes("recipe-holdout")
    if corpus is None:
        raise ValueError("the shared diagnostic corpus is unavailable")
    graph = load_ii_graph("ii_graph.npz")
    load_native_scorer(candidate, corpus.n_vocab)
    architecture = {key: manifest["metadata"][key] for key in (
        "d_model", "n_heads", "n_layers", "ff_mult", "dropout", "tie_output")}
    model = IngredientPredictor(corpus.itos, architecture)
    state = {
        path.stem[len("state__"):].replace("__", "."):
            torch.from_numpy(np.load(path, allow_pickle=False))
        for path in candidate.glob("state__*.npy")
    }
    model.network.load_state_dict(state, strict=True)
    model.eval().save_pretrained(args.out)
    reloaded = IngredientPredictor.from_pretrained(args.out, local_files_only=True)

    def scorer(contexts):
        with torch.inference_mode():
            return np.concatenate([
                reloaded(torch.from_numpy(contexts[start:start + 4096])).numpy()
                for start in range(0, len(contexts), 4096)])

    matrix = np.load(candidate / "embedding.npy", allow_pickle=False)
    got = completion_ranks(matrix, corpus, seed=SEED, unigram=graph.unigram,
                           scorer=scorer, include_instances=True)
    old = completion_ranks(
        np.load(reference / "embedding.npy", allow_pickle=False), corpus, seed=SEED,
        scorer=load_native_scorer(reference, corpus.n_vocab), include_instances=True)
    old_metrics = json.loads((reference / "metrics.json").read_text())
    if float((old["native"] <= 10).mean()) != old_metrics["M6_native_recall_at_10"]:
        raise ValueError("the reference predictor no longer reproduces its recorded score")
    for key in ("recipe_row", "target", "recipe_size"):
        if not np.array_equal(got[key], old[key]):
            raise ValueError("candidate and reference were evaluated on different cases")
    stored = json.loads((candidate / "metrics.json").read_text())
    recall = float((got["native"] <= 10).mean())
    mrr = float((1 / got["native"]).mean())
    if (recall != stored["M6_native_recall_at_10"]
            or mrr != stored["M6_native_mrr"] or len(got["native"]) != 20_000):
        raise ValueError("the exported full predictor does not reproduce its recorded metrics")
    paired = summarise_ranks(got["native"], old["native"], 2000,
                             np.random.default_rng(20260909))
    report = {
        "status": "verified_local_export_not_published",
        "publication_clearance": "pending_data_rights_review",
        "candidate": args.candidate,
        "reference": args.reference,
        "corpus_sha256": generation["sha256"],
        "n_completion": 20_000,
        "sampling_seed": SEED,
        "recall_at_10": recall,
        "recall_at_10_ci95": paired["recall_at_10_ci95"],
        "mrr": mrr,
        "previous_recall_at_10": float((old["native"] <= 10).mean()),
        "paired_improvement": paired["lift_over_popularity"],
        "paired_improvement_ci95": paired["lift_over_popularity_ci95"],
        "training": {key: manifest["metadata"][key] for key in (
            "n_recipes", "n_eligible_recipes", "n_examples_seen",
            "n_optimizer_steps", "epochs", "max_recipes", "max_len")},
        "n_parameters": sum(parameter.numel() for parameter in reloaded.parameters()),
        "model_file": {"bytes": (args.out / "model.safetensors").stat().st_size,
                       "sha256": digest(args.out / "model.safetensors")},
        "config_sha256": digest(args.out / "config.json"),
        "limitations": [
            "This is ingredient completion, not recipe generation or a foundation language model.",
            "The held-out row split is not duplicate-family-disjoint or externally decontaminated.",
            "Intervals are paired over instances, not training seeds or duplicate clusters.",
            "Public redistribution permissions remain unresolved; no permissive weights license is asserted.",
            "Scores are not calibrated probabilities or guarantees about taste, allergies or food safety.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out / "README.md").write_text(f"""---
library_name: pytorch
tags:
- ingredient-completion
- set-transformer
- pytorch_model_hub_mixin
---

# llmmm Ingredients - full-data candidate

A {report['n_parameters']:,}-parameter ingredient-set transformer trained from
random initialization. This package contains the complete conditional predictor,
not merely its much weaker token embeddings. It does not write recipes.

**Status: local research candidate. Public redistribution rights are still
under review; this card does not grant a weights license.**

## Measured result

On the same {report['n_completion']:,} held-out-row completion instances,
native recall@10 is **{recall:.3%}**, versus **{report['previous_recall_at_10']:.3%}**
for the earlier sampling-capped model. The paired improvement is
**{100 * report['paired_improvement']:.3f} percentage points**
(95% instance-bootstrap interval:
{100 * report['paired_improvement_ci95'][0]:.3f} to
{100 * report['paired_improvement_ci95'][1]:.3f} percentage points).

The training partition contains {report['training']['n_recipes']:,} recipes;
{report['training']['n_eligible_recipes']:,} pass the training length filter.
The trainer recorded {report['training']['n_examples_seen']:,} examples across
{report['training']['epochs']} epochs. Larger source-corpus totals are not the
number of examples this model saw.

These are diagnostics, not decontaminated generalization or recipe-generation
claims. Recipe-family overlap and external pretraining overlap are not excluded.
The prior and current runs, exact corpus identity, package hash and uncertainty
are recorded in `evaluation.json`.

## Local usage

Install the project's `torch` and `hf` extras from its reviewed source checkout:

```bash
cd model
python -m pip install -e '.[torch,hf]'
```

```python
from ingredient_model.hub import IngredientPredictor

model = IngredientPredictor.from_pretrained(
    "/path/to/this/folder", local_files_only=True
)
print(model.recommend(["tomato", "basil"], top_k=10))
```

Use canonical ingredient names from `model.vocabulary`; spaces are accepted in
place of underscores. Unknown names are rejected rather than silently omitted.
Recommendations exclude ingredients already supplied. Scores are unnormalized
logits, not calibrated probabilities. `forward()` exposes the underlying logits.

## Data and limitations

The corpus combines source groups with noncommercial or unresolved terms.
No source recipes, titles or cooking instructions are included in this package.
The model is not a substitute for culinary, allergy or food-safety review.
See the source repository's data provenance and licensing investigation before
any redistribution or commercial use.
""")
    print(f"Verified full predictor: {recall:.3%}; paired improvement "
          f"{100 * report['paired_improvement']:.3f} percentage points")
    print(f"Local export: {args.out}; no upload performed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
