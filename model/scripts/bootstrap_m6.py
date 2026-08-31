"""Confidence intervals for the recipe-completion metric, and for the headline.

M2 and M4 already carry analytic intervals. M6 carried none, which is awkward,
because M6 is the metric the whole argument rests on: "nine of sixteen models
score below a popularity baseline" is a statement about M6 and nothing else.
Quoted bare, it invites the obvious question of whether nine is really nine.

Two things are estimated here.

The first is an interval on each model's recall@10, which is ordinary. The
second is an interval on the *count itself* — how many models fall below the
baseline — which is the number actually being published and which no per-model
interval implies. That count is a function of sixteen comparisons that all
share one set of evaluation instances, so it cannot be assembled from sixteen
independent intervals; it has to be resampled jointly.

Hence a paired bootstrap. Every model is evaluated on the same 20,000
(recipe, hidden ingredient) pairs in the same order, because the draw is a
deterministic function of the seed and the corpus. So one resample of instance
indices is applied to every model at once, and each replicate yields a whole
leaderboard. The correlation between models — most of which is instance
difficulty, since a recipe with an unguessable hidden ingredient is hard for
everyone — is then carried correctly rather than assumed away. Treating the
models as independent would inflate the interval on the count substantially.

Ranks are cached, because computing them is the expensive part (a few minutes
per model) and resampling them is free.

    python scripts/bootstrap_m6.py            # cached where possible
    python scripts/bootstrap_m6.py --refresh  # recompute ranks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ingredient_model.config import PATHS
from ingredient_model.data import load_recipes
from ingredient_model.data.recipes import RECIPE_IDS
from ingredient_model.data.splits import held_out_recipes
from ingredient_model.eval.completion import completion_ranks

SWEEP = "all-v2"
K = 10
RANKS_NPZ = PATHS.results / "m6_ranks.npz"
OUT_JSON = PATHS.results / "m6_intervals.json"


def _runs() -> dict[str, dict]:
    root = PATHS.results / "runs" / SWEEP
    out: dict[str, dict] = {}
    for mp in sorted(root.glob("*/metrics.json")):
        man = mp.parent / "manifest.json"
        if not man.exists():
            continue
        model = json.loads(man.read_text()).get("model")
        if model:
            out[model] = {"dir": mp.parent,
                          "metrics": json.loads(mp.read_text())}
    return out


def _native_scorer(model: str, run_dir: Path, n_vocab: int):
    """The scorer a model would actually serve, rebuilt from its run.

    Two of the sixteen factorise something richer than the vector table they
    export, and the gap between the two is one of the study's findings, so the
    bootstrap has to reach the served scorer rather than settle for the
    embedding. They store it differently — EASE as an item-item matrix, the
    set transformer as network weights — and the difference is discovered from
    what is on disk rather than hardcoded by name.

    Anything else has no native scorer, and its embedding ranking *is* what it
    would serve.
    """
    item_scores = run_dir / "item_scores.npy"
    if item_scores.exists():
        B = np.load(item_scores)
        return lambda c: B[c].sum(1)
    if (run_dir / "state__tok__weight.npy").exists():
        from models.set_transformer.train import restore
        return restore(run_dir, n_vocab)
    return None


def compute_ranks(runs: dict[str, dict]) -> dict[str, np.ndarray]:
    """Per-instance ranks for every model, plus the shared baseline.

    The popularity ranks are identical for every model by construction, so
    they are computed once and stored once under `popularity`.
    """
    corpus = held_out_recipes("recipe-holdout")
    unigram = np.bincount(load_recipes(RECIPE_IDS).flat,
                          minlength=corpus.n_vocab).astype(float)

    store: dict[str, np.ndarray] = {}
    for model, r in sorted(runs.items()):
        W = np.load(r["dir"] / "embedding.npy")
        scorer = _native_scorer(model, r["dir"], corpus.n_vocab)

        got = completion_ranks(W, corpus, unigram=unigram, scorer=scorer)
        store[model] = got["embedding"].astype(np.float32)
        if "native" in got:
            store[f"{model}::native"] = got["native"].astype(np.float32)
        store.setdefault("popularity", got["popularity"].astype(np.float32))

        # Cross-check against what the run recorded. A silent disagreement
        # here would mean the intervals describe a different computation than
        # the leaderboard, which is the one failure this script must not have.
        checks = [("M6_recall_at_10", store[model])]
        if "native" in got:
            checks.append(("M6_native_recall_at_10", store[f"{model}::native"]))
        bits = []
        for key, ranks in checks:
            want = r["metrics"].get(key)
            have = float((ranks <= K).mean())
            tag = "native" if "native" in key else "embed "
            bad = want is not None and abs(have - want) >= 5e-4
            bits.append(f"{tag} {have:.4f}"
                        + (f"  <-- MISMATCH, metrics.json {want:.4f}" if bad
                           else ""))
        print(f"  {model:16s} " + "   ".join(bits))
    return store


def bootstrap(store: dict[str, np.ndarray], n_boot: int, seed: int) -> dict:
    models = sorted(k for k in store
                    if k != "popularity" and not k.endswith("::native"))
    pop = store["popularity"]
    n = len(pop)
    rng = np.random.default_rng(seed)

    # Hit indicators, stacked once. Resampling then costs one gather per
    # replicate over the whole leaderboard rather than per model.
    served = {m: store.get(f"{m}::native", store[m]) for m in models}
    H_emb = np.stack([(store[m] <= K) for m in models]).astype(np.float32)
    H_srv = np.stack([(served[m] <= K) for m in models]).astype(np.float32)
    h_pop = (pop <= K).astype(np.float32)

    emb = np.empty((n_boot, len(models)), np.float32)
    srv = np.empty((n_boot, len(models)), np.float32)
    base = np.empty(n_boot, np.float32)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        emb[b] = H_emb[:, idx].mean(1)
        srv[b] = H_srv[:, idx].mean(1)
        base[b] = h_pop[idx].mean()

    # The published claim, resampled. Counted on the served scorer, which is
    # what the site's leaderboard ranks on.
    n_below = (srv < base[:, None]).sum(1)

    def ci(a, axis=0):
        lo, hi = np.percentile(a, [2.5, 97.5], axis=axis)
        return lo, hi

    lo_e, hi_e = ci(emb)
    lo_s, hi_s = ci(srv)
    lo_b, hi_b = ci(base)

    point_base = float(h_pop.mean())
    rows = []
    for i, m in enumerate(models):
        pe, ps = float(H_emb[i].mean()), float(H_srv[i].mean())
        # Paired difference against the baseline on the same instances, which
        # is a tighter and more honest test than comparing two intervals.
        d = srv[:, i] - base
        rows.append({
            "model": m,
            "has_native": f"{m}::native" in store,
            "embedding_recall_at_10": pe,
            "embedding_ci95": [float(lo_e[i]), float(hi_e[i])],
            "served_recall_at_10": ps,
            "served_ci95": [float(lo_s[i]), float(hi_s[i])],
            "served_minus_popularity": ps - point_base,
            "served_minus_popularity_ci95": [float(np.percentile(d, 2.5)),
                                             float(np.percentile(d, 97.5))],
            # Two-sided bootstrap p for "this model differs from the
            # baseline", by the proportion of replicates on the other side.
            # Floored at the resolution the replicate count can actually
            # resolve: with B replicates nothing smaller than 2/B is
            # distinguishable from zero, and printing 0 would claim a
            # precision the procedure does not have.
            "p_two_sided": max(float(2 * min((d <= 0).mean(), (d >= 0).mean())),
                               2.0 / n_boot),
            "p_is_bound": bool(min((d <= 0).mean(), (d >= 0).mean()) == 0.0),
            "below_baseline": ps < point_base,
        })

    return {
        "note": "Paired bootstrap over the completion instances. One resample "
                "of instance indices is shared by every model, so the count "
                "of models below the baseline is resampled jointly rather "
                "than assembled from independent per-model intervals.",
        "sweep": SWEEP, "k": K, "n_instances": int(n),
        "n_boot": n_boot, "seed": seed,
        "popularity_recall_at_10": point_base,
        "popularity_ci95": [float(lo_b), float(hi_b)],
        "n_below_popularity": int((np.array([r["served_recall_at_10"]
                                             for r in rows]) < point_base).sum()),
        "n_below_popularity_ci95": [int(np.percentile(n_below, 2.5)),
                                    int(np.percentile(n_below, 97.5))],
        "n_below_popularity_distribution": {
            str(v): int(c) for v, c in
            zip(*np.unique(n_below, return_counts=True))},
        "n_models": len(models),
        "models": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refresh", action="store_true",
                    help="recompute ranks instead of using the cache")
    a = ap.parse_args()

    if RANKS_NPZ.exists() and not a.refresh:
        print(f"ranks <- {RANKS_NPZ.relative_to(PATHS.results.parent)}")
        store = {k: v for k, v in np.load(RANKS_NPZ).items()}
    else:
        runs = _runs()
        if not runs:
            raise SystemExit(f"no runs under results/runs/{SWEEP}")
        print(f"scoring {len(runs)} models on recipe-holdout")
        store = compute_ranks(runs)
        RANKS_NPZ.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(RANKS_NPZ, **store)
        print(f"ranks -> {RANKS_NPZ.relative_to(PATHS.results.parent)}")

    out = bootstrap(store, a.n_boot, a.seed)
    OUT_JSON.write_text(json.dumps(out, indent=1))

    lo, hi = out["n_below_popularity_ci95"]
    print(f"\npopularity recall@10 {out['popularity_recall_at_10']:.4f} "
          f"[{out['popularity_ci95'][0]:.4f}, {out['popularity_ci95'][1]:.4f}]")
    print(f"below baseline: {out['n_below_popularity']} of {out['n_models']} "
          f"(95% CI {lo} to {hi})")
    for r in sorted(out["models"], key=lambda r: -r["served_recall_at_10"]):
        mark = "below" if r["below_baseline"] else "     "
        d0, d1 = r["served_minus_popularity_ci95"]
        p = ("p<%.0e" % r["p_two_sided"]) if r["p_is_bound"] \
            else "p=%.3g" % r["p_two_sided"]
        print(f"  {r['model']:16s} {r['served_recall_at_10']:.4f} "
              f"[{r['served_ci95'][0]:.4f}, {r['served_ci95'][1]:.4f}]  "
              f"{mark}  Δ {r['served_minus_popularity']:+.4f} "
              f"[{d0:+.4f}, {d1:+.4f}]  {p}")
    print(f"\n-> {OUT_JSON.relative_to(PATHS.results.parent)}")


if __name__ == "__main__":
    main()
