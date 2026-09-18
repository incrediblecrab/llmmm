#!/usr/bin/env python3
"""Is the food-pairing signal just near-duplicate ingredients pairing with themselves?

Palermo et al. 2024 (arXiv:2406.15533) report that food pairing in Ahn's data is
"mostly due to trivial couplings of very similar ingredients". If that is the
whole story, then Delta should collapse once such pairs are removed, and any
cuisine-level contrast built on Delta is an artefact of how finely the
vocabulary splits the same food.

This is a controlled A/B. The recipe sample, the recipe sizes and the null draws
are byte-identical between arms; the only thing that changes is the
shared-compound matrix S. Two maskings are applied together:

  name    tokens(a) is a subset of tokens(b) or vice versa, so `almond` masks
          against `almond_milk` and `onion` against `red_onion`, while
          `soy_sauce` and `fish_sauce` are left alone because neither token set
          contains the other.
  chemical Jaccard(compounds_a, compounds_b) >= --jaccard, which catches
          near-identical profiles the names miss (cream against milk).

Delta surviving the mask is evidence the effect is about genuine cross-food
chemistry rather than vocabulary granularity.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_cuisines import DERIVED, MAX_SIZE, SEED, SOURCES, group_by_size, shared_compound_matrix
from analyze_cuisines_v2 import null_groups, per_recipe_pairs

OUT_DIR = Path(os.environ.get("EPICURE_OUT", Path(__file__).resolve().parents[1] / "results"))


def trivial_mask(itos: list[str], jaccard: float) -> tuple[np.ndarray, int, int]:
    """Boolean [n, n] marking ingredient pairs that are the same food twice."""
    n = len(itos)
    toks = [frozenset(name.split("_")) for name in itos]
    M = np.zeros((n, n), dtype=bool)

    # Name containment. Bucket by token so this stays far away from O(n^2) set ops.
    by_tok: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(toks):
        for tok in t:
            by_tok[tok].append(i)
    for members in by_tok.values():
        for a_pos, i in enumerate(members):
            for j in members[a_pos + 1:]:
                if toks[i] <= toks[j] or toks[j] <= toks[i]:
                    M[i, j] = M[j, i] = True
    n_name = int(M.sum() // 2)

    # Chemical near-identity, over ingredients that actually carry compounds.
    z = np.load(DERIVED / "flavor_graph.npz", allow_pickle=True)
    src, dst = z["src"].astype(int), z["dst"].astype(int)
    A = np.zeros((n, int(dst.max()) + 1), np.float32)
    A[src, dst] = 1.0
    inter = A @ A.T
    size = A.sum(1)
    union = size[:, None] + size[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        J = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
    np.fill_diagonal(J, 0.0)
    M |= J >= jaccard
    return M, n_name, int(M.sum() // 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-recipes", type=int, default=150_000)
    ap.add_argument("--n-null", type=int, default=8)
    ap.add_argument("--jaccard", type=float, default=0.9)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--out", default="trivial_pairs_audit.json")
    a = ap.parse_args()

    rng = np.random.default_rng(SEED)
    z = np.load(DERIVED / "recipe_ids.npz", allow_pickle=True)
    flat, offs, src = z["flat"], z["offsets"], z["source"]
    itos = [str(x) for x in z["itos"]]
    n = len(itos)

    S = shared_compound_matrix(n)
    M, n_name, n_tot = trivial_mask(itos, a.jaccard)
    S_kept = S.copy()
    S_kept[M] = 0.0
    nz = int((S > 0).sum() // 2)
    print(f"{n_name:,} pairs masked by name, {n_tot:,} total after chemical "
          f"Jaccard >= {a.jaccard}")
    print(f"masked pairs carrying compounds: "
          f"{int((M & (S > 0)).sum() // 2):,} of {nz:,} compound-sharing pairs\n")

    by_cuisine: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(src):
        cu = SOURCES.get(str(s), (None, None))[0]
        if cu:
            by_cuisine[cu].append(i)

    # Coverage confound: only a minority of the vocabulary carries compounds at
    # all, and a pair contributes nothing unless both ends are covered. If Delta
    # simply tracked how well FlavorDB covers a cuisine's larder, the cuisine
    # contrast would be a property of the database, not of the cooking.
    covered = np.zeros(n, dtype=bool)
    covered[np.unique(np.load(DERIVED / "flavor_graph.npz",
                              allow_pickle=True)["src"].astype(int))] = True
    coverage = {}
    for cu_name, idxs in by_cuisine.items():
        tok_hit = tok_all = 0
        pair_hit = pair_all = 0.0
        for r in idxs:
            ing = flat[offs[r]:offs[r + 1]]
            k = len(ing)
            if k < 2 or k > MAX_SIZE:
                continue
            nc = int(covered[ing].sum())
            tok_hit += nc
            tok_all += k
            pair_hit += nc * (nc - 1) / 2
            pair_all += k * (k - 1) / 2
        if tok_all:
            coverage[cu_name] = {"token_coverage": tok_hit / tok_all,
                                 "pair_coverage": pair_hit / pair_all}
    print(f"vocabulary carrying compounds: {int(covered.sum()):,} of {n:,}")
    print(f"per-cuisine pair coverage spans "
          f"{min(c['pair_coverage'] for c in coverage.values()):.1%}-"
          f"{max(c['pair_coverage'] for c in coverage.values()):.1%}\n")

    results = {}
    print(f"{'cuisine':16}{'n':>9}{'relD all':>11}{'relD kept':>11}{'shift':>9}")
    for cu in sorted(by_cuisine, key=lambda c: -len(by_cuisine[c])):
        rows = np.array(by_cuisine[cu])
        if len(rows) < 500:
            continue
        if len(rows) > a.max_recipes:
            rows = rng.choice(rows, a.max_recipes, replace=False)
        groups = group_by_size([flat[offs[r]:offs[r + 1]] for r in rows])
        if not groups:
            continue

        freq = np.zeros(n)
        for k, idx in groups.items():
            np.add.at(freq, idx.ravel(), 1.0)

        out = {}
        # Identical null draws for both arms: draw once, score twice.
        draws = [null_groups(groups, freq, rng, a.chunk) for _ in range(a.n_null)]
        for arm, mat in (("all", S), ("kept", S_kept)):
            r_sum, r_cnt = per_recipe_pairs(groups, mat)
            real = r_sum.sum() / r_cnt.sum()
            acc = np.zeros_like(r_sum)
            aggs = []
            for g in draws:
                ns, nc = per_recipe_pairs(g, mat)
                acc += ns
                aggs.append(ns.sum() / nc.sum())
            acc /= a.n_null
            null_mean = float(np.mean(aggs))
            delta = real - null_mean
            m = len(r_sum)
            C = r_cnt.sum()
            infl = ((r_sum - acc) - delta * r_cnt) / C
            se = float(np.sqrt(m * np.var(infl, ddof=1)))
            out[arm] = {"real": float(real), "null_mean": null_mean,
                        "delta": float(delta), "delta_se": se,
                        "rel_delta": float(delta / null_mean) if null_mean else 0.0,
                        "rel_ci_lo": float((delta - 1.96 * se) / null_mean) if null_mean else 0.0,
                        "rel_ci_hi": float((delta + 1.96 * se) / null_mean) if null_mean else 0.0,
                        "sign_resolved": bool(abs(delta) > 1.96 * se),
                        "recipes_used": int(m)}
        out["sign_flipped"] = bool((out["all"]["delta"] > 0) != (out["kept"]["delta"] > 0))
        results[cu] = out
        print(f"{cu:16}{out['all']['recipes_used']:>9,}"
              f"{out['all']['rel_delta']:>10.2%}{out['kept']['rel_delta']:>11.2%}"
              f"{out['kept']['rel_delta'] - out['all']['rel_delta']:>+9.2%}"
              f"{'  SIGN FLIP' if out['sign_flipped'] else ''}", flush=True)

    flips = [c for c, r in results.items() if r["sign_flipped"]]
    print(f"\nsign flips after masking: {len(flips)}/{len(results)}"
          f"{' -> ' + ', '.join(flips) if flips else ''}")

    # Does Delta track FlavorDB coverage rather than cuisine? Spearman, with a
    # permutation p-value so this needs no distributional assumption at n=18.
    shared = sorted(set(results) & set(coverage))
    cov_corr = None
    if len(shared) >= 4:
        d = np.array([results[c]["all"]["rel_delta"] for c in shared])
        cv = np.array([coverage[c]["pair_coverage"] for c in shared])

        def spearman(x: np.ndarray, y: np.ndarray) -> float:
            rx = np.argsort(np.argsort(x)).astype(float)
            ry = np.argsort(np.argsort(y)).astype(float)
            return float(np.corrcoef(rx, ry)[0, 1])

        obs = spearman(d, cv)
        perm = rng.permutation
        null_r = np.array([abs(spearman(d, perm(cv))) for _ in range(10_000)])
        pval = float((null_r >= abs(obs)).mean())
        cov_corr = {"spearman_r": obs, "perm_p": pval, "n_cuisines": len(shared)}
        print(f"Spearman r(relative delta, pair coverage) = {obs:+.3f} "
              f"(permutation p = {pval:.3f}, n = {len(shared)}) -> "
              f"{'NOT explained' if pval > 0.05 else 'CONFOUNDED'} by coverage")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"cuisines": results, "coverage": coverage,
               "coverage_correlation": cov_corr,
               "params": {"seed": SEED, "max_recipes": a.max_recipes,
                          "n_null": a.n_null, "jaccard": a.jaccard,
                          "masked_pairs_name": n_name, "masked_pairs_total": n_tot,
                          "vocab_with_compounds": int(covered.sum()),
                          "vocab_total": int(n), "max_size": MAX_SIZE}}
    (OUT_DIR / a.out).write_text(json.dumps(payload, indent=2))
    print(f"saved {OUT_DIR / a.out}")


if __name__ == "__main__":
    main()
