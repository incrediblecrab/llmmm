#!/usr/bin/env python3
"""H4 re-run: the food-pairing hypothesis across cuisines, with honest error bars.

This supersedes `analyze_cuisines.py` for the H4 verdict. The statistic is
unchanged:

    Delta_c = <N_s>_real - <N_s>_null

where N_s is the mean shared-compound count over ingredient pairs, pooled over
pairs, and the null redraws each recipe at its true size from that cuisine's own
ingredient-frequency distribution. Three things are fixed.

1. Sample cap. The original capped every cuisine at 30,000 recipes, using
   253,145 of 4,543,143 available (5.6%). It truncated hardest exactly where the
   Ahn comparison lives: north_american 30k of 2.67M, chinese 30k of 1.43M. Here
   the cap is configurable and defaults high enough that 16 of 18 cuisines are
   analysed complete.

2. Error bars. The original reported z = Delta / sd(null replicates). That
   denominator is the scatter of the *null* statistic and ignores sampling error
   in the observed statistic entirely, so it shrinks as the corpus grows and
   overstates significance without bound. Here Delta carries a recipe-level
   bootstrap CI, which is the quantity that actually decides whether a sign is
   resolved. Monte-Carlo error of the null mean is reported separately.

3. Comparison target. Ahn et al. 2011 (Sci. Rep. 1:196) state: "North American
   and Western European cuisines exhibit a statistically significant tendency
   towards recipes whose ingredients share flavor compounds. By contrast, East
   Asian *and Southern European* cuisines avoid recipes whose ingredients share
   flavor compounds." The pre-registration mis-stated this, predicting Delta > 0
   for Spanish and Greek, which are Southern European and which Ahn places on
   the negative side. Both the pre-registered list and Ahn's actual claim are
   scored below, separately.

Delta > 0 = food pairing; Delta < 0 = contrast pairing.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_cuisines import (
    DERIVED,
    MAX_SIZE,
    MIN_RECIPES,
    SEED,
    SOURCES,
    shared_compound_matrix,
)

OUT_DIR = Path(os.environ.get("EPICURE_OUT", Path(__file__).resolve().parents[1] / "results"))

# Ahn et al. 2011 grouped recipes into five regions and reported the sign of
# Delta for each. Cuisines outside those five are reported but not scored.
AHN_REGION = {
    "north_american": ("North American", +1),
    "german": ("Western European", +1),
    "spanish": ("Southern European", -1),
    "greek": ("Southern European", -1),
    "chinese": ("East Asian", -1),
    "japanese": ("East Asian", -1),
    "taiwanese": ("East Asian", -1),
}

# The pre-registration's own prediction list, scored as written.
PREREG = {
    "spanish": +1, "german": +1, "greek": +1, "romanian": +1, "russian": +1,
    "chinese": -1, "japanese": -1, "thai": -1, "vietnamese": -1,
    "filipino": -1, "indonesian": -1,
}


def per_recipe_pairs(groups: dict[int, np.ndarray], S: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Per-recipe (sum of shared compounds over pairs, number of pairs).

    Groups are visited in sorted key order so that the real and null arrays for
    a cuisine are aligned recipe-for-recipe, which is what lets the bootstrap
    resample them as pairs.
    """
    sums, counts = [], []
    for k in sorted(groups):
        idx = groups[k]
        if k < 2:
            continue
        acc = np.zeros(len(idx))
        for i in range(k - 1):
            a = idx[:, i]
            for j in range(i + 1, k):
                acc += S[a, idx[:, j]]
        sums.append(acc)
        counts.append(np.full(len(idx), k * (k - 1) // 2, dtype=np.float64))
    if not sums:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(sums), np.concatenate(counts)


def group_by_size(recipes: list[np.ndarray]) -> dict[int, np.ndarray]:
    by = defaultdict(list)
    for r in recipes:
        if 2 <= len(r) <= MAX_SIZE:
            by[len(r)].append(r)
    return {k: np.asarray(v, dtype=np.int32) for k, v in by.items()}


def null_groups(groups: dict[int, np.ndarray], freq: np.ndarray,
                rng: np.random.Generator, chunk: int) -> dict[int, np.ndarray]:
    """Gumbel top-k weighted sampling without replacement, in row chunks.

    Identical in distribution to the original implementation; chunked because an
    un-chunked (n_recipes x n_vocab) draw is tens of GB at full corpus size.
    """
    p = freq / freq.sum()
    live = np.flatnonzero(freq > 0)
    logp = np.log(p[live] / p[live].sum())
    out = {}
    for k, idx in groups.items():
        m, k_eff = len(idx), min(k, len(live))
        res = np.empty((m, k_eff), dtype=np.int32)
        for s in range(0, m, chunk):
            e = min(s + chunk, m)
            g = rng.gumbel(size=(e - s, len(live))) + logp
            res[s:e] = live[np.argpartition(-g, k_eff - 1, axis=1)[:, :k_eff]]
        out[k] = res
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-recipes", type=int, default=200_000)
    ap.add_argument("--n-null", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-max-n", type=int, default=300_000,
                    help="skip the bootstrap check above this many recipes; the "
                         "linearised SE is the primary interval either way")
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--dedup", action="store_true",
                    help="collapse recipes sharing an ingredient set to one "
                         "representative. The scraped corpora overlap (RecipeNLG "
                         "re-publishes food.com), so near-duplicates are common "
                         "and pseudo-replicate the CI without adding evidence.")
    ap.add_argument("--out", default="cuisine_pairing_v2.json")
    a = ap.parse_args()

    print(f"max_recipes={a.max_recipes:,} n_null={a.n_null} n_boot={a.n_boot}", flush=True)
    rng = np.random.default_rng(SEED)

    z = np.load(DERIVED / "recipe_ids.npz", allow_pickle=True)
    flat, offs, src = z["flat"], z["offsets"], z["source"]
    n = len(z["itos"])
    print(f"{len(src):,} recipes, {n} ingredients", flush=True)

    S = shared_compound_matrix(n)
    print(f"shared-compound matrix {S.shape}\n", flush=True)

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
        avail = len(rows)
        if avail < MIN_RECIPES:
            print(f"skip {cu}: only {avail} recipes", flush=True)
            continue
        capped = avail > a.max_recipes
        if capped:
            rows = rng.choice(rows, a.max_recipes, replace=False)

        recipes = [flat[offs[r]:offs[r + 1]] for r in rows]
        n_dup = 0
        if a.dedup:
            seen: set[frozenset[int]] = set()
            keep = []
            for rec in recipes:
                key = frozenset(rec.tolist())
                if key not in seen:
                    seen.add(key)
                    keep.append(rec)
            n_dup = len(recipes) - len(keep)
            recipes = keep

        groups = group_by_size(recipes)
        if not groups:
            continue

        freq = np.zeros(n)
        for k, idx in groups.items():
            np.add.at(freq, idx.ravel(), 1.0)

        r_sum, r_cnt = per_recipe_pairs(groups, S)
        real = r_sum.sum() / r_cnt.sum()

        # Accumulate per-recipe null sums across replicates, and keep each
        # replicate's aggregate so the null's own Monte-Carlo error is visible.
        n_acc = np.zeros_like(r_sum)
        n_agg = []
        for _ in range(a.n_null):
            ns, nc = per_recipe_pairs(null_groups(groups, freq, rng, a.chunk), S)
            n_acc += ns
            n_agg.append(ns.sum() / nc.sum())
        n_acc /= a.n_null
        null_mean = float(np.mean(n_agg))
        null_mc_se = float(np.std(n_agg, ddof=1) / np.sqrt(a.n_null)) if a.n_null > 1 else float("nan")
        delta = real - null_mean

        # Delta is a ratio estimator D/C over recipes. Linearise it: the
        # influence value of recipe i is (d_i - delta * c_i) / C, so the
        # recipe-level SE is O(m) to compute and needs no resampling. This is
        # what makes the full corpus tractable; the bootstrap below is kept as a
        # check on small cuisines, where the two agree.
        m = len(r_sum)
        d_i = r_sum - n_acc
        C = r_cnt.sum()
        infl = (d_i - delta * r_cnt) / C
        se = float(np.sqrt(m * np.var(infl, ddof=1)))
        lo, hi = delta - 1.96 * se, delta + 1.96 * se

        boot_lo = boot_hi = None
        if a.n_boot and m <= a.boot_max_n:
            boot = np.empty(a.n_boot)
            for b in range(a.n_boot):
                bi = rng.integers(0, m, m)
                boot[b] = (r_sum[bi].sum() - n_acc[bi].sum()) / r_cnt[bi].sum()
            boot_lo, boot_hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))

        results[cu] = {
            "region": regions[cu], "recipes_available": int(avail),
            "recipes_used": int(m), "capped": bool(capped),
            "duplicates_removed": int(n_dup),
            "real": float(real), "null_mean": null_mean,
            "null_mc_se": null_mc_se, "delta": float(delta),
            "delta_se": se, "delta_ci_lo": float(lo), "delta_ci_hi": float(hi),
            "boot_ci_lo": boot_lo, "boot_ci_hi": boot_hi,
            "rel_delta": float(delta / null_mean) if null_mean else 0.0,
            "rel_ci_lo": float(lo / null_mean) if null_mean else 0.0,
            "rel_ci_hi": float(hi / null_mean) if null_mean else 0.0,
            "sign_resolved": bool(lo > 0 or hi < 0),
            "direction": "pairing" if delta > 0 else "contrast",
        }
        r = results[cu]
        chk = ""
        if boot_lo is not None:
            chk = f" boot[{boot_lo:+7.3f},{boot_hi:+7.3f}]"
        print(f"{cu:<16} {regions[cu]:<16} n={m:>7,}{'*' if capped else ' '} "
              f"delta={delta:+7.3f} [{lo:+7.3f},{hi:+7.3f}] "
              f"rel={r['rel_delta']:+6.1%} "
              f"{'RESOLVED' if r['sign_resolved'] else 'unresolved'}{chk}", flush=True)

    scored = {}
    for label, table in (("prereg", PREREG), ("ahn_actual", {k: v[1] for k, v in AHN_REGION.items()})):
        ag = dis = unres = 0
        detail = {}
        for cu, pred in table.items():
            if cu not in results:
                continue
            r = results[cu]
            obs = 1 if r["delta"] > 0 else -1
            ok = obs == pred
            if not r["sign_resolved"]:
                unres += 1
            ag += ok
            dis += not ok
            detail[cu] = {"predicted": pred, "observed": obs, "agree": bool(ok),
                          "resolved": r["sign_resolved"]}
        scored[label] = {"agree": ag, "disagree": dis, "unresolved": unres,
                         "majority_disagree": bool(dis > ag), "detail": detail}
        print(f"\n{label}: agree {ag}, disagree {dis} "
              f"({unres} signs unresolved) -> majority disagree: {dis > ag}")

    by_region: dict[str, list[float]] = defaultdict(list)
    for cu, r in results.items():
        by_region[r["region"]].append(r["rel_delta"])
    region_summary = {k: {"mean_rel_delta": float(np.mean(v)), "cuisines": len(v)}
                      for k, v in by_region.items()}
    print("\nby region (mean relative delta):")
    for k, v in sorted(region_summary.items(), key=lambda kv: -kv[1]["mean_rel_delta"]):
        print(f"  {k:<18} {v['mean_rel_delta']:+7.2%}  ({v['cuisines']})")

    payload = {"cuisines": results, "regions": region_summary, "scored": scored,
               "params": {"seed": SEED, "max_recipes": a.max_recipes,
                          "n_null": a.n_null, "n_boot": a.n_boot,
                          "dedup": bool(a.dedup),
                          "min_recipes": MIN_RECIPES, "max_size": MAX_SIZE}}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / a.out
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved {out}", flush=True)


if __name__ == "__main__":
    main()
