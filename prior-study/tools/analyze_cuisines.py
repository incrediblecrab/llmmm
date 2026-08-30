#!/usr/bin/env python3
"""H4: the food-pairing hypothesis across cuisines (Ahn et al. 2011, at scale).

Ahn et al. found that North American and Western European recipes combine
ingredients that *share* flavour compounds, while East Asian recipes avoid
doing so. They had ~57k recipes from three western sites. This runs the same
statistic over 4.6M recipes spanning ~20 national sources, which is the point:
their asymmetry was derived from a corpus thin outside the west, so testing it
on natively-sourced regional corpora is a real test rather than a restatement.

Statistic, following the paper:

    Delta_c = <N_s>_real - <N_s>_null

where N_s is the mean number of shared compounds over ingredient pairs in a
recipe. The null redraws each recipe at its true size from that cuisine's own
ingredient-frequency distribution, so Delta measures pairing preference and not
which ingredients the cuisine happens to use often.

Delta > 0 = food pairing; Delta < 0 = contrast pairing.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DERIVED = Path(os.environ.get("EPICURE_DERIVED", ROOT / "data" / "derived"))
OUT_DIR = Path(os.environ.get("EPICURE_OUT", ROOT / "models"))

SEED = 20260805
MAX_RECIPES = 30_000     # per cuisine, sampled; bounds runtime
N_NULL = 15              # null replicates -> CI on Delta
MIN_RECIPES = 500        # below this the estimate is too noisy to report
MAX_SIZE = 40            # recipes longer than this are rare and dominate cost

# Source -> (cuisine, region). Sources whose culinary origin is not defensible
# are mapped to None and excluded from the headline, but still reported.
SOURCES = {
    "01-recipenlg":       ("north_american", "North America"),
    "foodcom-522k":       ("north_american", "North America"),
    "foodcom-raw-231k":   ("north_american", "North America"),
    "allrecipes-33k":     ("north_american", "North America"),
    "02-xiachufang":      ("chinese", "East Asia"),
    "taiwan-1.8k":        ("taiwanese", "East Asia"),
    "japanese-3k":        ("japanese", "East Asia"),
    "05-vietnamese":      ("vietnamese", "Southeast Asia"),
    "08-indonesian":      ("indonesian", "Southeast Asia"),
    "filipino-2k":        ("filipino", "Southeast Asia"),
    "thai-1k":            ("thai", "Southeast Asia"),
    "thai-1k-seasoning":  ("thai", "Southeast Asia"),
    "07-indian":          ("indian", "South Asia"),
    "indian-7k":          ("indian", "South Asia"),
    "06-turkish":         ("turkish", "Middle East"),
    "turkish-102k":       ("turkish", "Middle East"),
    "persian-6k":         ("persian", "Middle East"),
    "hebrew-9.7k":        ("israeli", "Middle East"),
    "moroccan-4.6k":      ("moroccan", "North Africa"),
    "03-povarenok":       ("russian", "Eastern Europe"),
    "povarenok-detail":   ("russian", "Eastern Europe"),
    "romanian-881":       ("romanian", "Eastern Europe"),
    "09-chefkoch":        ("german", "Western Europe"),
    "04-spanish":         ("spanish", "Southern Europe"),
    "greek-5k":           ("greek", "Southern Europe"),
    "thefoodprocessor-74k": (None, None),
    "kaggle-food-13k":    (None, None),
    "bhuvii-17k":         (None, None),
    "halal-2k":           (None, None),   # dietary category, not a cuisine
}


def shared_compound_matrix(n: int) -> np.ndarray:
    """S[i, j] = number of flavour compounds ingredients i and j share."""
    z = np.load(DERIVED / "flavor_graph.npz", allow_pickle=True)
    s, d = z["src"].astype(int), z["dst"].astype(int)
    A = np.zeros((n, int(d.max()) + 1), np.float32)
    A[s, d] = 1.0
    S = A @ A.T
    np.fill_diagonal(S, 0.0)
    return S


def mean_shared(groups: dict[int, np.ndarray], S: np.ndarray) -> float:
    """Mean shared-compound count over all ingredient pairs, pooled over
    recipes that are grouped by size so the pair gather stays vectorised."""
    tot, npairs = 0.0, 0
    for k, idx in groups.items():
        if k < 2:
            continue
        for i in range(k - 1):
            a = idx[:, i]
            for j in range(i + 1, k):
                tot += float(S[a, idx[:, j]].sum())
        npairs += len(idx) * k * (k - 1) // 2
    return tot / max(npairs, 1)


def group_by_size(recipes: list[np.ndarray]) -> dict[int, np.ndarray]:
    by = defaultdict(list)
    for r in recipes:
        if 2 <= len(r) <= MAX_SIZE:
            by[len(r)].append(r)
    return {k: np.asarray(v, dtype=np.int32) for k, v in by.items()}


def null_groups(groups: dict[int, np.ndarray], freq: np.ndarray,
                rng: np.random.Generator) -> dict[int, np.ndarray]:
    """Redraw every recipe at its true size from the cuisine's own ingredient
    frequency distribution, without replacement within a recipe."""
    p = freq / freq.sum()
    live = np.flatnonzero(freq > 0)
    pl = p[live] / p[live].sum()
    out = {}
    for k, idx in groups.items():
        k_eff = min(k, len(live))
        # Gumbel top-k is a vectorised weighted sample-without-replacement
        g = rng.gumbel(size=(len(idx), len(live))) + np.log(pl)
        out[k] = live[np.argpartition(-g, k_eff - 1, axis=1)[:, :k_eff]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-recipes", type=int, default=MAX_RECIPES,
                    help="per-cuisine sample cap; raise to trade runtime for a "
                         "tighter estimate on the large corpora")
    ap.add_argument("--n-null", type=int, default=N_NULL,
                    help="null replicates; more shrinks the null SD, so report "
                         "effect size alongside z rather than z alone")
    a = ap.parse_args()
    max_recipes, n_null = a.max_recipes, a.n_null
    print(f"max_recipes={max_recipes:,} n_null={n_null}", flush=True)

    rng = np.random.default_rng(SEED)
    z = np.load(DERIVED / "recipe_ids.npz", allow_pickle=True)
    flat, offs, src = z["flat"], z["offsets"], z["source"]
    itos = [str(x) for x in z["itos"]]
    n = len(itos)
    print(f"{len(src):,} recipes, {n} ingredients", flush=True)

    S = shared_compound_matrix(n)
    print(f"shared-compound matrix {S.shape}, "
          f"mean {S[S > 0].mean():.2f} over {int((S > 0).sum()):,} pairs\n", flush=True)

    by_cuisine: dict[str, list[int]] = defaultdict(list)
    regions: dict[str, str] = {}
    for i, s in enumerate(src):
        cu, reg = SOURCES.get(str(s), (None, None))
        if cu:
            by_cuisine[cu].append(i)
            regions[cu] = reg

    results = {}
    for cu in sorted(by_cuisine, key=lambda c: -len(by_cuisine[c])):
        rows = np.array(by_cuisine[cu])
        if len(rows) < MIN_RECIPES:
            print(f"skip {cu}: only {len(rows)} recipes", flush=True)
            continue
        if len(rows) > max_recipes:
            rows = rng.choice(rows, max_recipes, replace=False)

        recipes = [flat[offs[r]:offs[r + 1]] for r in rows]
        groups = group_by_size(recipes)
        if not groups:
            continue

        freq = np.zeros(n)
        for r in recipes:
            np.add.at(freq, r, 1.0)

        real = mean_shared(groups, S)
        nulls = np.array([mean_shared(null_groups(groups, freq, rng), S)
                          for _ in range(n_null)])
        delta = real - nulls.mean()
        # Permutation-style spread: each null replicate redraws every recipe,
        # so its scatter is the variability of the statistic under the null.
        # It shrinks as the null mean is pinned down, which makes bare
        # significance a weak lens -- report effect size alongside it.
        sd = float(nulls.std(ddof=1))
        ci = 1.96 * sd
        used = sum(len(v) for v in groups.values())
        results[cu] = {
            "region": regions[cu], "recipes_total": len(by_cuisine[cu]),
            "recipes_used": int(used), "real": real,
            "null_mean": float(nulls.mean()), "null_sd": sd,
            "delta": float(delta), "ci95": float(ci),
            "z": float(delta / sd) if sd > 0 else float("inf"),
            "rel_delta": float(delta / nulls.mean()) if nulls.mean() else 0.0,
            "significant": bool(abs(delta) > ci),
            "direction": "pairing" if delta > 0 else "contrast",
        }
        print(f"{cu:<16} {regions[cu]:<16} n={used:>6,}  real={real:7.3f}  "
              f"null={nulls.mean():7.3f}  delta={delta:+7.3f}  "
              f"rel={delta / max(nulls.mean(), 1e-9):+6.1%}  "
              f"{'*' if abs(delta) > ci else ' '}", flush=True)

    by_region: dict[str, list[float]] = defaultdict(list)
    for cu, r in results.items():
        by_region[r["region"]].append(r["rel_delta"])
    region_summary = {k: {"mean_rel_delta": float(np.mean(v)), "cuisines": len(v)}
                      for k, v in by_region.items()}

    print("\nby region (mean relative delta; Ahn predicts west > 0 > east asia):")
    for k, v in sorted(region_summary.items(), key=lambda kv: -kv[1]["mean_rel_delta"]):
        print(f"  {k:<18} {v['mean_rel_delta']:+7.2%}  ({v['cuisines']} cuisines)")

    payload = {"cuisines": results, "regions": region_summary,
               "params": {"seed": SEED, "max_recipes": max_recipes,
                          "n_null": n_null, "min_recipes": MIN_RECIPES}}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "cuisine_pairing.json").write_text(json.dumps(payload, indent=2))
    print(f"\nsaved {OUT_DIR}/cuisine_pairing.json", flush=True)


if __name__ == "__main__":
    main()
