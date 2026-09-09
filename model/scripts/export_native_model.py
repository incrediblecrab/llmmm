"""Export complete native predictors with checkpoint-appropriate evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from ingredient_model.artifacts import load_native_scorer
from ingredient_model.config import PATHS, REPO, SEED
from ingredient_model.data.graphs import load_ii_graph
from ingredient_model.data.recipes import load_recipes
from ingredient_model.data.splits import held_out_recipes
from ingredient_model.eval.completion import completion_ranks
from ingredient_model.hub import IngredientPredictor
from ingredient_model.production import verify_full_training
from ingredient_model.workspace import verify_training_corpus
from m6_intervals import summarise_ranks

DEFAULT_CANDIDATE = "train-v2-full-20260909/masked-set-full-recipe-holdout-s42"
DEFAULT_REFERENCE = "train-v2-20260909/masked-set-recipe-holdout-s42"
DEFAULT_PRODUCTION = "production-v2-all-20260909/llmmm-recipes-full-s42"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def export_predictor(candidate: Path, vocabulary: list[str], manifest: dict, out: Path):
    native = load_native_scorer(candidate, len(vocabulary))
    if native is None:
        raise ValueError("the candidate has no complete native predictor")
    architecture = {key: manifest["metadata"][key] for key in (
        "d_model", "n_heads", "n_layers", "ff_mult", "dropout", "tie_output")}
    model = IngredientPredictor(vocabulary, architecture)
    state = {
        path.stem[len("state__"):].replace("__", "."):
            torch.from_numpy(np.load(path, allow_pickle=False))
        for path in candidate.glob("state__*.npy")
    }
    model.network.load_state_dict(state, strict=True)
    model.eval().save_pretrained(out)
    reloaded = IngredientPredictor.from_pretrained(out, local_files_only=True)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(
            reloaded.state_dict()[name], tensor, rtol=0, atol=0)
    return reloaded, native


def export_production(args, generation: dict, candidate: Path, manifest: dict) -> int:
    verification = verify_full_training(
        candidate, PATHS.data, expected_recipes=generation["recipes"])
    if digest(REPO / "models/set_transformer/train.py") != verification["training_code_sha256"]:
        raise ValueError("the checkout's trainer differs from the recorded training code")
    vocabulary = load_recipes().itos
    model, native = export_predictor(candidate, vocabulary, manifest, args.out)
    sizes = sorted({min(size, len(vocabulary) - 1) for size in (2, 3, 8, 16, 32, 97)})
    rng = np.random.default_rng(SEED)
    contexts_checked = 0
    for size in sizes:
        contexts = np.array([
            rng.choice(len(vocabulary), size, replace=False) for _ in range(32)
        ], dtype=np.int64)
        expected = native(contexts)
        with torch.inference_mode():
            actual = model(torch.from_numpy(contexts)).numpy()
        np.testing.assert_array_equal(actual, expected)
        contexts_checked += len(contexts)
    source_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    report = {
        "status": "verified_local_export_not_published",
        "repo_id": args.repo_id,
        "tag": args.tag,
        "candidate": args.candidate,
        "source_code_revision": source_revision,
        "training": verification,
        "n_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model_file": {"bytes": (args.out / "model.safetensors").stat().st_size,
                       "sha256": digest(args.out / "model.safetensors")},
        "config_sha256": digest(args.out / "config.json"),
        "reload_parity": {
            "context_count": contexts_checked, "context_sizes": sizes,
            "all_state_tensors_identical": True, "max_absolute_logit_error": 0.0,
            "contexts_are_synthetic_not_quality_evaluation": True,
        },
        "evaluation_status": "not_run",
    }
    policy = {
        "version": 1, "release_tier": "private_educational_all_record_model",
        "public_release_cleared": False, "weights_license": None,
        "scope": "Complete ingredient predictor; recipe-text generation is unsupported.",
        "intended_use": "Noncommercial research and education.",
        "rights_note": "No permissive weights license is granted. Dataset license labels "
                      "alone do not establish model-weight redistribution rights.",
        "source_code_repository": "https://github.com/incrediblecrab/llmmm",
        "source_code_revision": source_revision,
        "source_data_included": False,
    }
    for filename, document in {
        "training_verification.json": report,
        "release_policy.json": policy,
    }.items():
        (args.out / filename).write_text(json.dumps(document, indent=2) + "\n")
    shutil.copyfile(candidate / "manifest.json", args.out / "training_manifest.json")
    (args.out / "README.md").write_text(f"""---
library_name: pytorch
tags:
- ingredient-completion
- set-transformer
- pytorch_model_hub_mixin
---

# llmmm-recipes

A {report['n_parameters']:,}-parameter ingredient-completion model trained from
scratch on **all {verification['recipes_per_epoch']:,} canonical recipe records**.
Each record is a set of normalized ingredient names. The model predicts missing
ingredients; recipe-text generation is unsupported.

## Training evidence

| Measure | Verified count |
|---|---:|
| Recipe records used in each epoch | {verification['recipes_per_epoch']:,} |
| Completed epochs | {verification['epochs']} |
| Recipe presentations across all epochs | {verification['example_presentations']:,} |
| Ingredient slots processed in each epoch | {verification['ingredient_slots_per_epoch']:,} |
| Optimizer steps | {verification['optimizer_steps']:,} |
| Recipe length, in canonical ingredients | {verification['minimum_recipe_length']} to {verification['maximum_recipe_length']} |

There was no sampling cap or length exclusion. Single-ingredient records,
two-ingredient records and long records were included. During each epoch, the
trainer checked that every record was visited exactly once and that all
ingredient slots were processed. Record counts do not imply unique recipe
content: different records can describe the same recipe.

The complete saved predictor was restored. Exported weights were byte-identical
at the tensor level, and reloaded logits matched the native predictor exactly
on {contexts_checked} synthetic contexts. This checks serialization and inference,
not prediction quality. Corpus and artifact hashes, per-epoch coverage and reload
evidence are in [training_verification.json](training_verification.json).
Settings and losses are in [training_manifest.json](training_manifest.json).

## Evaluation status

**This checkpoint has no held-out quality score.** Training includes the rows
previously held out for evaluation. The older `v0.2.0-preview` remains available
as a separate evaluated checkpoint; its scores do not apply to these weights.

## Usage

Install the inference implementation from its pinned source revision:

```bash
python -m pip install "ingredient-model[torch,hf] @ git+https://github.com/incrediblecrab/llmmm.git@{source_revision}#subdirectory=model"
hf auth login
```

```python
from ingredient_model.hub import IngredientPredictor

model = IngredientPredictor.from_pretrained(
    "{args.repo_id}",
    revision="{args.tag}",
    token=True,
)
print(model.recommend(["tomato", "basil"], top_k=10))
```

Inference does not require the training corpus. Supply at least two distinct
ingredient names from `model.vocabulary`; spaces can replace underscores.
Unknown names raise an error, and recommendations exclude supplied ingredients.
Scores are unnormalized logits, not calibrated probabilities.

## Intended use and limits

For noncommercial research and education. No permissive weights license is
granted. This package contains no source recipes, titles or cooking instructions.
Predictions have not been validated for taste, allergies or food safety.
Review the [data provenance](https://github.com/incrediblecrab/llmmm/blob/{source_revision}/raw-data/README.md)
and applicable terms before redistribution or commercial use.
""")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified all-record export: {verification['recipes_per_epoch']:,} records "
          f"per epoch; {contexts_checked} exact reload comparisons; no held-out score")
    print(f"Local export: {args.out}; no upload performed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--repo-id", default="incrediblecrab/llmmm-recipes")
    parser.add_argument("--tag", default="v0.3.0-all-recipes")
    args = parser.parse_args()
    args.candidate = args.candidate or (
        DEFAULT_PRODUCTION if args.production else DEFAULT_CANDIDATE)
    args.report = args.report or PATHS.results / (
        "all_record_release.json" if args.production else "native_full_release.json")
    if args.report.suffix != ".json" or args.out.resolve() in args.report.resolve().parents:
        parser.error("--report must be a .json file outside the export directory")
    if args.out.exists():
        raise FileExistsError(f"{args.out}: exports are immutable; choose a new path")
    generation = verify_training_corpus(PATHS.data, "v2")
    candidate = PATHS.runs / args.candidate
    reference = PATHS.runs / args.reference
    manifest = json.loads((candidate / "manifest.json").read_text())
    if args.production:
        return export_production(args, generation, candidate, manifest)
    if (manifest["model"] != "masked-set"
            or manifest["environment"]["corpus_sha256"] != generation["sha256"]
            or manifest["params"]["split"] != "recipe-holdout"):
        raise ValueError("the candidate is not the expected corpus-pinned native model")
    corpus = held_out_recipes("recipe-holdout")
    if corpus is None:
        raise ValueError("the shared diagnostic corpus is unavailable")
    graph = load_ii_graph("ii_graph.npz")
    reloaded, _ = export_predictor(candidate, corpus.itos, manifest, args.out)

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

# llmmm-recipes

A {report['n_parameters']:,}-parameter set transformer that predicts missing
ingredients. It was trained from random initialization. The package contains
the complete predictor. It does not generate recipes.

This is a local research candidate. Public-release permissions remain unresolved,
and this card grants no weights license.

## Measured result

The hidden ingredient appeared among the top ten predictions in {recall:.1%}
of {report['n_completion']:,} test cases, compared with
{report['previous_recall_at_10']:.1%} for the earlier capped run.
The paired gain was {100 * report['paired_improvement']:.1f} percentage points,
with a 95% bootstrap interval of
{100 * report['paired_improvement_ci95'][0]:.1f} to
{100 * report['paired_improvement_ci95'][1]:.1f} percentage points.

| Training measure | Count |
|---|---:|
| Recipes in the training partition | {report['training']['n_recipes']:,} |
| Recipes used after the length filter | {report['training']['n_eligible_recipes']:,} |
| Epochs | {report['training']['epochs']} |
| Examples processed across all epochs | {report['training']['n_examples_seen']:,} |

The split holds out recipe rows but does not exclude duplicate recipe families
or account for external pretraining overlap. The bootstrap resamples test cases;
it does not measure variation across training runs. Exact scores, run identities,
and package hashes are in [evaluation.json](evaluation.json). Training settings
are in [training_manifest.json](training_manifest.json).

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

Use ingredient names from `model.vocabulary`. Spaces can replace underscores.
Unknown names raise an error. Recommendations exclude ingredients already
supplied. Scores are unnormalized logits, not calibrated probabilities.
`forward()` returns the logits.

## Data and limitations

The training corpus includes sources with noncommercial or unresolved terms.
This package contains no source recipes, titles or cooking instructions.
Predictions have not been validated for taste, allergies or food safety.
Review the source repository's data provenance and release permissions before
redistribution or commercial use.
""")
    print(f"Verified full predictor: {recall:.3%}; paired improvement "
          f"{100 * report['paired_improvement']:.3f} percentage points")
    print(f"Local export: {args.out}; no upload performed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
