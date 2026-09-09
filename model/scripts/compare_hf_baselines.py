"""Matched completion diagnostics, not unseen-data or recipe-generation claims."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from ingredient_model.artifacts import iter_runs, load_native_scorer
from ingredient_model.config import PATHS, REPO, SEED
from ingredient_model.data.graphs import GRAPH_FULL, load_ii_graph
from ingredient_model.data.splits import held_out_recipes
from ingredient_model.eval.completion import completion_ranks
from ingredient_model.hf_baselines import (
    LOCK, ensure_model, epicure_embeddings, group_scores, load_pins,
    recipebert_embeddings,
)
from ingredient_model.workspace import load_workspace, verify_training_corpus
from m6_intervals import summarise_ranks

N_TEST = 20_000
BOOTSTRAP_SEED = 20260909
REFERENCE = "llmmm/masked-set/native"


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def summarize(store: dict[str, np.ndarray], corpus, train_frequency: np.ndarray,
              n_boot: int) -> list[dict]:
    reference = store[REFERENCE]
    targets = store["target"]
    source = corpus.source[store["recipe_row"]]
    language = corpus.lang[store["recipe_row"]]
    head = np.argsort(-train_frequency, kind="stable")[:20]
    frequency_group = np.where(np.isin(targets, head), "top_20", "outside_top_20")
    target_n = np.bincount(targets, minlength=corpus.n_vocab)
    rows = []
    identities = {"recipe_row", "target", "recipe_size"}
    for name, ranks in sorted(store.items()):
        if name in identities:
            continue
        if len(ranks) != N_TEST or not np.isfinite(ranks).all() or (ranks < 1).any():
            raise ValueError(f"{name}: invalid or incomplete completion ranks")
        summary = summarise_ranks(ranks, store["popularity"], n_boot,
                                  np.random.default_rng(BOOTSTRAP_SEED))
        paired = summarise_ranks(reference, ranks, n_boot,
                                 np.random.default_rng(BOOTSTRAP_SEED))
        target_hits = np.bincount(targets, weights=(ranks <= 10),
                                  minlength=corpus.n_vocab)
        rows.append({
            "model": name,
            "n": int(len(ranks)),
            **summary,
            "masked_set_native_minus_this": paired["lift_over_popularity"],
            "masked_set_native_minus_this_ci95": paired["lift_over_popularity_ci95"],
            "macro_recall_at_10_observed_targets": float(
                (target_hits[target_n > 0] / target_n[target_n > 0]).mean()),
            "n_observed_targets": int((target_n > 0).sum()),
            "by_train_frequency": group_scores(ranks, frequency_group),
            "by_source": group_scores(ranks, source),
            "by_language": group_scores(ranks, language),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true",
                        help="fetch missing pinned public weights; inference stays local")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--out", type=Path,
                        default=PATHS.results / "hf_completion_diagnostic.json")
    args = parser.parse_args()
    if args.n_boot < 1:
        parser.error("--n-boot must be positive")
    if args.out.suffix != ".json":
        parser.error("--out must end in .json; companion .csv and .npz files are derived")
    import torch

    generation = verify_training_corpus(PATHS.data, "v2")
    corpus = held_out_recipes("recipe-holdout")
    if corpus is None:
        raise ValueError("restore the recipe-holdout corpus before comparison")
    graph = load_ii_graph(GRAPH_FULL)
    train_graph = load_ii_graph("ii_graph_rh_train.npz")
    if corpus.itos != graph.itos or corpus.itos != train_graph.itos:
        raise ValueError("corpus and graph ingredient identities disagree")
    pins = load_pins()
    workspace = load_workspace()
    store: dict[str, np.ndarray] = {}
    provenance = {}
    timings = {}

    def score(name: str, matrix: np.ndarray, scorer=None) -> dict:
        if matrix.ndim != 2 or matrix.shape[0] != corpus.n_vocab:
            raise ValueError(f"{name}: embedding and canonical vocabulary disagree")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name}: non-finite embedding")
        start = time.perf_counter()
        got = completion_ranks(
            matrix, corpus, n_test=N_TEST, seed=SEED,
            unigram=graph.unigram, scorer=scorer, include_instances=True)
        if len(got.get("embedding", [])) != N_TEST:
            raise ValueError(f"{name}: fewer than {N_TEST} eligible instances")
        for key in ("popularity", "recipe_row", "target", "recipe_size"):
            if key in store and not np.array_equal(store[key], got[key]):
                raise ValueError(f"{name}: the shared evaluation draw changed")
            store[key] = got[key]
        store[f"{name}/raw"] = got["embedding"]
        if scorer is not None:
            store[f"{name}/native"] = got["native"]
        centered = completion_ranks(
            matrix - matrix.mean(0, keepdims=True), corpus,
            n_test=N_TEST, seed=SEED, include_instances=True)
        for key in ("recipe_row", "target", "recipe_size"):
            if not np.array_equal(store[key], centered[key]):
                raise ValueError(f"{name}: centered preprocessing changed the draw")
        store[f"{name}/centered"] = centered["embedding"]
        timings[name] = time.perf_counter() - start
        print(f"{name}: raw={(got['embedding'] <= 10).mean():.4f}, "
              f"centered={(centered['embedding'] <= 10).mean():.4f}"
              + (f", native={(got['native'] <= 10).mean():.4f}"
                 if "native" in got else ""), flush=True)
        return got

    for folder in iter_runs(PATHS.runs / workspace.training_sweep,
                            require_embedding=False):
        manifest = json.loads((folder / "manifest.json").read_text())
        if (manifest["environment"]["corpus_sha256"] != generation["sha256"]
                or manifest["params"]["split"] != "recipe-holdout"):
            raise ValueError(f"{folder}: incompatible corpus or split")
        name = f"llmmm/{manifest['model']}"
        if name in provenance:
            raise ValueError(f"multiple current runs for {name}; declare a single cohort")
        matrix = np.load(folder / "embedding.npy", allow_pickle=False)
        if list(matrix.shape) != manifest["shape"]:
            raise ValueError(f"{folder}: embedding shape differs from its manifest")
        got = score(name, matrix, load_native_scorer(folder, corpus.n_vocab))
        recorded = json.loads((folder / "metrics.json").read_text())
        recorded_deltas = {}
        precision_reproduction = {}
        for field, prefix in (("embedding", "M6_"), ("native", "M6_native_"),
                              ("popularity", "M6_popularity_")):
            if field not in got:
                continue
            measures = {
                "recall_at_10": float((got[field] <= 10).mean()),
                "mrr": float((1 / got[field]).mean()),
            }
            for metric, measured in measures.items():
                key = prefix + metric
                delta = measured - recorded[key]
                recorded_deltas[key] = delta
                # Historical training scored in-memory vectors before their
                # float32 serialization. Keep reloaded vector MRR observable.
                if abs(delta) > 1e-12 and not (field == "embedding" and metric == "mrr"):
                    raise ValueError(f"{folder}: {key} no longer reproduces")
                if abs(delta) > 1e-12:
                    precise = completion_ranks(
                        matrix.astype(np.float64), corpus, n_test=N_TEST, seed=SEED)
                    precise_mrr = float((1 / precise["embedding"]).mean())
                    if abs(precise_mrr - recorded[key]) > 1e-12:
                        raise ValueError(f"{folder}: {key} drift is not explained by scoring precision")
                    precision_reproduction[key] = precise_mrr
                    print(f"{name}: saved-vector {key} differs from its historical "
                          f"value by {delta:+.12g}; float64 scoring reproduces it",
                          flush=True)
        provenance[name] = {
            "run": str(folder.relative_to(PATHS.runs)),
            "manifest": fingerprint(folder / "manifest.json"),
            "weights": {path.name: fingerprint(path) for path in sorted(folder.glob("*.npy"))},
            "reloaded_minus_recorded_metrics": recorded_deltas,
            "reproduced_with_float64_scoring": precision_reproduction,
        }
    if REFERENCE not in store:
        raise ValueError("the current cohort lacks the masked-set native reference")

    encoding_seconds = None
    for model_id, spec in pins.items():
        if spec["role"] == "recipe_generator":
            continue
        local = (PATHS.prior_study / "data" / "raw" / model_id.split("/")[1]
                 if spec["role"] == "ingredient_embeddings" else None)
        folder = ensure_model(model_id, download=args.download, local_source=local)
        if spec["role"] == "ingredient_embeddings":
            matrix = epicure_embeddings(folder, corpus.itos)
        elif spec["role"] == "food_text_encoder":
            start = time.perf_counter()
            matrix = recipebert_embeddings(folder, corpus.itos)
            encoding_seconds = time.perf_counter() - start
        else:
            raise ValueError(f"{model_id}: unsupported comparison role")
        score(model_id, matrix)
        provenance[model_id] = {
            "revision": spec["revision"],
            "model_card": f"https://huggingface.co/{model_id}/blob/{spec['revision']}/README.md",
            "files": spec["files"],
            "embedding_shape": list(matrix.shape),
            "evaluated_embedding_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
        }

    rows = summarize(store, corpus, train_graph.unigram, args.n_boot)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ranks_path = args.out.with_suffix(".npz")
    np.savez_compressed(ranks_path, **store)
    source_files = [Path(__file__), LOCK, *[
        REPO / name for name in (
            "workspace.json", "ingredient_model/hf_baselines.py",
            "ingredient_model/config.py", "ingredient_model/artifacts.py",
            "ingredient_model/data/graphs.py", "ingredient_model/data/recipes.py",
            "ingredient_model/data/splits.py", "ingredient_model/eval/completion.py",
            "ingredient_model/eval/metrics.py", "models/set_transformer/train.py",
            "models/text_embedding/train.py", "scripts/m6_intervals.py",
        )
    ]]
    report = {
        "status": "diagnostic_only",
        "corpus_generation": generation["generation"],
        "corpus_sha256": generation["sha256"],
        "protocol": {
            "split": "recipe-holdout",
            "sample_frame": 80_000,
            "n_completion": N_TEST,
            "sampling_seed": SEED,
            "n_candidates": corpus.n_vocab,
            "context_exclusion": True,
            "ties": "midrank",
            "embedding_operator": "Sum unit context vectors; rank by dot product with unit candidate vectors.",
            "preprocessing": ["raw", "centered_before_unit_normalization"],
            "recipebert": "CPU float32; readable canonical ingredient names; mean final-layer tokens including special tokens, excluding padding; no recipe-level fine-tuning.",
            "popularity": "Historical full-corpus unigram, not a train-only baseline.",
            "frequency_groups": "Top 20 ingredients by train-graph unigram versus all others; stable ID tie break.",
            "macro_metric": "Equal weight for each target ingredient observed in this draw; unobserved targets omitted.",
            "n_bootstrap": args.n_boot,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "uncertainty": "Paired instance bootstrap; not duplicate-cluster, training-seed, model-selection or multiplicity-adjusted uncertainty.",
        },
        "limitations": [
            "Known or possible overlap with external pretraining; this is not a decontaminated unseen-data benchmark.",
            "Random recipe-row holdout does not exclude duplicate recipe families across partitions.",
            "RecipeBERT is adapted as a name-representation baseline, not evaluated on its original MLM objective or a contextual recipe encoder.",
            "Both raw and centered vectors are reported; neither is silently selected after seeing results.",
            "No comparison to T5 generation quality is made: the current llmmm checkpoint is not a recipe generator.",
            "No taste, cooking safety, quantities, dietary guarantees or broad model-superiority claim follows from these scores.",
            "Historical in-memory vector MRR may differ slightly from float32 reloads; all deltas are recorded. Native metrics and all headline recall values must reproduce.",
        ],
        "generation_baseline": {
            "model": "flax-community/t5-recipe-generation",
            "revision": pins["flax-community/t5-recipe-generation"]["revision"],
            "status": "not_run_no_llmmm_generator",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            **{name: importlib.metadata.version(name)
               for name in ("numpy", "torch", "transformers", "safetensors")},
            "torch_num_threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
        },
        "source_files": {str(path.relative_to(REPO)): fingerprint(path) for path in source_files},
        "derived_inputs": {
            "ii_graph.npz": fingerprint(PATHS.graphs / "ii_graph.npz"),
            "ii_graph_rh_train.npz": fingerprint(PATHS.graphs / "ii_graph_rh_train.npz"),
            "recipe_ids_rh_train.npz": fingerprint(PATHS.recipes / "recipe_ids_rh_train.npz"),
        },
        "provenance": provenance,
        "rank_artifact": {"filename": ranks_path.name, **fingerprint(ranks_path)},
        "timing_seconds": {"scoring_by_model": timings, "recipebert_encoding": encoding_seconds},
        "models": rows,
    }
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with args.out.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([
            "model", "n", "recall_at_10", "recall_ci95_low", "recall_ci95_high",
            "mrr", "macro_recall_at_10_observed_targets", "outside_top_20_recall_at_10",
            "masked_set_native_minus_this", "paired_ci95_low", "paired_ci95_high",
        ])
        for row in rows:
            writer.writerow([
                row["model"], row["n"], row["recall_at_10"],
                *row["recall_at_10_ci95"], row["mrr"],
                row["macro_recall_at_10_observed_targets"],
                row["by_train_frequency"]["outside_top_20"]["recall_at_10"],
                row["masked_set_native_minus_this"],
                *row["masked_set_native_minus_this_ci95"],
            ])
    print(f"Wrote {args.out}; diagnostic only, not a release benchmark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
