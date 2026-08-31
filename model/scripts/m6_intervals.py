"""Independent replication of the recipe-completion intervals.

`bootstrap_m6.py` is the canonical producer of `results/m6_intervals.json`,
and the site reads that file. This script is a *second, independently written*
implementation of the same estimand, kept deliberately rather than deleted:
the whole argument leans on "nine of sixteen models fall below popularity", so
one implementation re-deriving it is worth more than a comment asserting it.

It therefore writes `results/m6_replication.json`, never the canonical file,
and finishes by comparing itself against the canonical artefact and exiting
non-zero on disagreement. An earlier version of this script wrote to the
canonical path and silently overwrote it, which is precisely the failure this
repository exists to argue against.

Because it is a replication, differences are expected in the CI bounds and not
in the point estimates: the two scripts use different bootstrap seeds and
different resample counts, so bounds agree only to Monte-Carlo error, while the
point estimates are deterministic functions of the same stored ranks.

--- original description, still accurate for the estimand ---

M6 is the number the argument finally leans on: given a held-out recipe with
one ingredient hidden, does the model put that ingredient in its top ten
suggestions, and does it do better than recommending the globally popular
ingredients? The stored runs report the point estimates, but not their
sampling uncertainty, so "nine of sixteen models are below popularity" was a
bare count rather than an interval-backed statement.

This script does not retrain models and does not redefine M6. It restores the
saved artefacts, asks the existing completion ranker for the same 20,000
held-out instances, and bootstraps those instances. The lift interval is
paired: one resample of instance indices is used for both the model and the
popularity baseline. That matters because the two systems are tested on the
same hidden ingredients; resampling them independently would throw away the
shared recipe difficulty and overstate the uncertainty of their difference.

The limitation is correspondingly narrow. These intervals describe uncertainty
from the sampled completion instances, not uncertainty from training, corpus
construction, hyperparameter choice, or the current running seed sweep.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np

from ingredient_model.config import PATHS
from ingredient_model.data.graphs import GRAPH_FULL, load_ii_graph
from ingredient_model.data.splits import held_out_recipes
from ingredient_model.eval.completion import completion_ranks

SWEEP = "all-v2"
N_COMPLETION = 20_000
K = 10
OUT_JSON = PATHS.results / "m6_replication.json"
CANONICAL_JSON = PATHS.results / "m6_intervals.json"

POINT_TOL = 5e-4

Scores = Callable[[np.ndarray], np.ndarray]


def load_runs(root: Path) -> dict[str, dict]:
    """Map model name to its completed run directory and stored metrics."""
    out: dict[str, dict] = {}
    for path in sorted(root.glob("*/metrics.json")):
        manifest = path.parent / "manifest.json"
        if not manifest.exists():
            continue
        meta = json.loads(manifest.read_text())
        model = meta.get("model")
        if model:
            out[model] = {
                "dir": path.parent,
                "manifest": meta,
                "metrics": json.loads(path.read_text()),
            }
    return out


def native_scorer(run_dir: Path, n_vocab: int) -> Scores | None:
    """Restore the conditional scorer when the saved run has one.

    The native path is intentionally discovered from stored artefacts rather
    than from model names. EASE persists the item-item scoring matrix directly;
    the masked-set model persists a transformer state that its own restore
    helper turns back into the scorer used during evaluation.
    """
    item_scores = run_dir / "item_scores.npy"
    if item_scores.exists():
        table = np.load(item_scores)
        return lambda ctx: table[ctx].sum(1)
    if (run_dir / "state__tok__weight.npy").exists():
        from models.set_transformer.train import restore

        return restore(run_dir, n_vocab)
    return None


def ci95(x: np.ndarray) -> list[float]:
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


def summarise_ranks(model: np.ndarray, popularity: np.ndarray,
                    n_boot: int, rng: np.random.Generator) -> dict:
    """Point estimates and paired bootstrap intervals for one rank vector."""
    hit = (model <= K).astype(np.float64)
    pop_hit = (popularity <= K).astype(np.float64)
    rr = (1.0 / model).astype(np.float64)
    n = len(model)

    boot_recall = np.empty(n_boot, np.float64)
    boot_mrr = np.empty(n_boot, np.float64)
    boot_lift = np.empty(n_boot, np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        # The same index vector is the statistical object here: a replicate
        # is a new completion exam, not two unrelated exams for model and
        # baseline.
        boot_recall[b] = hit[idx].mean()
        boot_mrr[b] = rr[idx].mean()
        boot_lift[b] = (hit[idx] - pop_hit[idx]).mean()

    return {
        "recall_at_10": float(hit.mean()),
        "recall_at_10_bootstrap_mean": float(boot_recall.mean()),
        "recall_at_10_ci95": ci95(boot_recall),
        "mrr": float(rr.mean()),
        "mrr_bootstrap_mean": float(boot_mrr.mean()),
        "mrr_ci95": ci95(boot_mrr),
        "lift_over_popularity": float((hit - pop_hit).mean()),
        "lift_over_popularity_bootstrap_mean": float(boot_lift.mean()),
        "lift_over_popularity_ci95": ci95(boot_lift),
    }


def verdict(interval: list[float]) -> str:
    if interval[1] < 0:
        return "below"
    if interval[0] > 0:
        return "above"
    return "indistinguishable"


def stored_delta(stored: dict, got: dict,
                 pairs: list[tuple[str, str]]) -> float:
    diffs = [abs(float(stored[k]) - float(got[v])) for k, v in pairs
             if k in stored]
    return max(diffs) if diffs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=1_000)
    ap.add_argument("--seed", type=int, default=20260830)
    args = ap.parse_args()

    root = PATHS.results / "runs" / SWEEP
    runs = load_runs(root)
    if not runs:
        raise SystemExit(f"no completed runs under {root}")

    corpus = held_out_recipes("recipe-holdout")
    if corpus is None:
        raise SystemExit("recipe-holdout has no completion corpus")
    unigram = np.asarray(load_ii_graph(GRAPH_FULL).unigram, np.float64)

    rng = np.random.default_rng(args.seed)
    out = {
        "sweep": SWEEP,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "n_completion": N_COMPLETION,
        "k": K,
        "method": (
            "Paired non-parametric bootstrap over M6 completion instances. "
            "For lift, model and popularity hits are resampled with the same "
            "indices because they were observed on the same held-out cases."
        ),
        "models": [],
    }

    max_mismatch = 0.0
    pop_summary = None
    for model, run in sorted(runs.items()):
        W = np.load(run["dir"] / "embedding.npy")
        scorer = native_scorer(run["dir"], corpus.n_vocab)
        got = completion_ranks(W, corpus, n_test=N_COMPLETION,
                               unigram=unigram, scorer=scorer)
        if "popularity" not in got:
            raise SystemExit(f"{model}: popularity ranks were not computed")

        if pop_summary is None:
            pop_hit = (got["popularity"] <= K).astype(np.float32)
            pop_rr = (1.0 / got["popularity"]).astype(np.float32)
            pop_summary = {
                "recall_at_10": float(pop_hit.mean()),
                "mrr": float(pop_rr.mean()),
                "n": int(len(got["popularity"])),
            }

        embedding = summarise_ranks(got["embedding"], got["popularity"],
                                    args.n_boot, rng)
        mismatch = stored_delta(run["metrics"], embedding, [
            ("M6_recall_at_10", "recall_at_10"),
            ("M6_mrr", "mrr"),
            ("M6_lift_over_popularity", "lift_over_popularity"),
        ])
        row = {
            "model": model,
            "run_id": run["manifest"].get("run_id", run["dir"].name),
            "n": int(len(got["embedding"])),
            "embedding": embedding,
            "served": "embedding",
        }

        if "native" in got:
            native = summarise_ranks(got["native"], got["popularity"],
                                     args.n_boot, rng)
            mismatch = max(mismatch, stored_delta(run["metrics"], native, [
                ("M6_native_recall_at_10", "recall_at_10"),
                ("M6_native_mrr", "mrr"),
                ("M6_native_lift_over_popularity", "lift_over_popularity"),
            ]))
            row["native"] = native
            row["served"] = "native"

        row["verdict"] = verdict(row[row["served"]]
                                 ["lift_over_popularity_ci95"])
        row["max_abs_stored_mismatch"] = float(mismatch)
        max_mismatch = max(max_mismatch, mismatch)
        out["models"].append(row)

    if pop_summary is None:
        raise SystemExit("no popularity baseline was computed")
    out["popularity"] = pop_summary

    counts = {"below": 0, "above": 0, "indistinguishable": 0}
    for row in out["models"]:
        counts[row["verdict"]] += 1
    out["headline_counts"] = counts
    out["max_abs_stored_mismatch"] = float(max_mismatch)

    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n")

    print(f"popularity recall@10 {pop_summary['recall_at_10']:.4f}  "
          f"n={pop_summary['n']:,}")
    print(f"max stored-point mismatch {max_mismatch:.6g}")
    print(f"{'model':<18} {'scorer':<9} {'lift':>8} "
          f"{'ci low':>8} {'ci high':>8}  verdict")
    for row in sorted(out["models"],
                      key=lambda r: r[r["served"]]
                      ["lift_over_popularity"]):
        served = row[row["served"]]
        lo, hi = served["lift_over_popularity_ci95"]
        print(f"{row['model']:<18} {row['served']:<9} "
              f"{served['lift_over_popularity']:>+8.4f} "
              f"{lo:>+8.4f} {hi:>+8.4f}  {row['verdict']}")
        if "native" in row:
            emb = row["embedding"]
            lo_e, hi_e = emb["lift_over_popularity_ci95"]
            print(f"{row['model']:<18} {'embedding':<9} "
                  f"{emb['lift_over_popularity']:>+8.4f} "
                  f"{lo_e:>+8.4f} {hi_e:>+8.4f}  "
                  f"{verdict(emb['lift_over_popularity_ci95'])}")

    print("\nheadline counts: "
          f"{counts['below']} below, {counts['above']} above, "
          f"{counts['indistinguishable']} indistinguishable")
    print(f"wrote {OUT_JSON.relative_to(PATHS.results.parent)}")

    raise SystemExit(compare_to_canonical(out))


def compare_to_canonical(out: dict) -> int:
    """Check this replication against the canonical artefact.

    Only the point estimates and the below/above verdicts are asserted. The
    interval bounds are deliberately not, because the two implementations draw
    different numbers of bootstrap resamples from different seeds, so their
    bounds differ by Monte-Carlo error by construction. Asserting them would
    produce a check that fails for a reason that is not a defect.
    """
    if not CANONICAL_JSON.exists():
        print(f"\nno canonical artefact at "
              f"{CANONICAL_JSON.relative_to(PATHS.results.parent)}; "
              f"run bootstrap_m6.py first — nothing compared")
        return 1

    canon = json.loads(CANONICAL_JSON.read_text())
    theirs = {r["model"]: r for r in canon["models"]}
    mine = {r["model"]: r for r in out["models"]}

    missing = set(theirs) ^ set(mine)
    problems: list[str] = []
    if missing:
        problems.append(f"models present in only one file: "
                        f"{', '.join(sorted(missing))}")

    worst = 0.0
    for name in sorted(set(theirs) & set(mine)):
        a, b = mine[name], theirs[name]
        lift_mine = a[a["served"]]["lift_over_popularity"]
        lift_canon = b["served_minus_popularity"]
        gap = abs(lift_mine - lift_canon)
        worst = max(worst, gap)
        if gap > POINT_TOL:
            problems.append(f"{name}: lift {lift_mine:+.4f} here vs "
                            f"{lift_canon:+.4f} canonical (gap {gap:.4g})")
        if (a["verdict"] == "below") != bool(b["below_baseline"]):
            problems.append(f"{name}: verdict {a['verdict']} here vs "
                            f"{'below' if b['below_baseline'] else 'above'} "
                            f"canonical")

    n_below_canon = canon["n_below_popularity"]
    if out["headline_counts"]["below"] != n_below_canon:
        problems.append(f"headline count: {out['headline_counts']['below']} "
                        f"below here vs {n_below_canon} canonical")

    print(f"\nreplication vs canonical ({CANONICAL_JSON.name}):")
    print(f"  largest point-estimate gap  {worst:.6g}  "
          f"(tolerance {POINT_TOL:g})")
    if problems:
        print(f"  DISAGREES on {len(problems)} point(s):")
        for p in problems:
            print(f"    - {p}")
        return 1
    print(f"  agrees on all {len(mine)} models, all verdicts, and the "
          f"headline count of {n_below_canon} below popularity")
    return 0


if __name__ == "__main__":
    main()
