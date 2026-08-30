#!/usr/bin/env python3
"""Build the NPMI ingredient-ingredient graph from the normalised corpus.

The paper's Cooc model walks "a 203,508-edge ingredient-ingredient NPMI graph".
NPMI rather than raw co-occurrence because recipe co-occurrence is dominated by
a handful of near-universal ingredients: salt appears with everything, so raw
counts rank "salt-onion" above every genuinely informative pair. Normalised
pointwise mutual information divides that popularity out and lands in [-1, 1].

    pmi(i,j)  = log( p(i,j) / (p(i) p(j)) )
    npmi(i,j) = pmi(i,j) / -log p(i,j)

Probabilities are over recipes, not over ingredient slots: an ingredient is
counted once per recipe regardless of how many times it is listed, which is why
`normalize.dump` stores a *set* per recipe.

The edge count is a function of the retention threshold, so rather than assume
one, this sweeps candidate thresholds and reports the count at each, then picks
whichever reproduces the paper's 203,508 most closely.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IDS = ROOT / "data" / "derived" / "recipe_ids.npz"
OUT = ROOT / "data" / "derived" / "ii_graph.npz"
TARGET = 203_508


def cooccurrence(flat: np.ndarray, offsets: np.ndarray, n_vocab: int):
    """Upper-triangular pair counts plus per-ingredient recipe counts.

    Recipes are bucketed by ingredient count so each bucket is a dense (m, k)
    matrix and every pair in it can be produced by one `triu_indices` gather.
    A per-recipe Python loop over 4.6M recipes takes tens of minutes; this runs
    in well under one, and the bucketing is exact rather than approximate.

    Pairs are accumulated as a single flattened code (a * n_vocab + b) through
    `bincount`, which is far faster than `np.add.at` on a 2-D array.
    """
    lens = np.diff(offsets)
    uni = np.bincount(flat, minlength=n_vocab).astype(np.int64)
    acc = np.zeros(n_vocab * n_vocab, dtype=np.int64)

    for k in range(2, int(lens.max()) + 1):
        rows = np.nonzero(lens == k)[0]
        if not len(rows):
            continue
        iu, ju = np.triu_indices(k, 1)
        # Cap the working set at ~20M pairs per slice to bound peak memory.
        step = max(int(20_000_000 / len(iu)), 1)
        for s in range(0, len(rows), step):
            r = rows[s:s + step]
            idx = offsets[r][:, None] + np.arange(k)
            ids = flat[idx].astype(np.int64)      # already sorted per recipe
            a = ids[:, iu].ravel()
            b = ids[:, ju].ravel()
            acc += np.bincount(a * n_vocab + b, minlength=n_vocab * n_vocab)
        print(f"  k={k:<3} recipes {len(rows):>9,}", flush=True)

    nz = np.nonzero(acc)[0]
    return nz // n_vocab, nz % n_vocab, acc[nz], uni


def main(min_count: int | None = None) -> None:
    if not IDS.exists():
        raise SystemExit(f"missing {IDS}; run: python tools/normalize.py dump")
    z = np.load(IDS, allow_pickle=True)
    flat, offsets, itos = z["flat"].astype(np.int64), z["offsets"], z["itos"]
    n_vocab = len(itos)
    n_recipes = len(offsets) - 1
    print(f"{n_recipes:,} recipes, vocab {n_vocab}")

    ai0, bi0, cnt0, uni = cooccurrence(flat, offsets, n_vocab)
    cnt0 = cnt0.astype(np.float64)
    print(f"raw distinct pairs {len(cnt0):,}\n")

    def npmi_for(mc: int):
        keep = cnt0 >= mc
        a, b, c = ai0[keep], bi0[keep], cnt0[keep]
        pij = c / n_recipes
        pmi = np.log(pij / ((uni[a] / n_recipes) * (uni[b] / n_recipes)))
        return a, b, c, pmi / -np.log(pij)

    # Two knobs interact: a low min_count admits rare pairs whose NPMI is
    # inflated by the classic low-count PMI bias, while a high one caps the
    # achievable edge count outright. Search both rather than assuming either.
    grid_mc = [min_count] if min_count else [1, 2, 3, 5, 10]
    print(f"{'min_count':>10}{'pairs':>12}{'npmi>=0':>12}{'best t':>9}"
          f"{'edges@best':>12}")
    best = None
    for mc in grid_mc:
        a, b, c, npmi = npmi_for(mc)
        grid = np.arange(-0.30, 0.45, 0.005)
        counts = np.array([(npmi >= t).sum() for t in grid])
        j = int(np.argmin(np.abs(counts - TARGET)))
        print(f"{mc:>10}{len(c):>12,}{int((npmi>=0).sum()):>12,}"
              f"{grid[j]:>9.3f}{counts[j]:>12,}")
        if best is None or abs(counts[j] - TARGET) < abs(best[1] - TARGET):
            best = (mc, int(counts[j]), float(grid[j]), a, b, c, npmi)

    mc, got, thr, ai, bi, cnt, npmi = best
    sel = npmi >= thr
    print(f"\nchosen min_count {mc}, npmi >= {thr:.3f} -> {int(sel.sum()):,} "
          f"edges ({int(sel.sum())/TARGET:.1%} of paper's 203,508)")

    src, dst, w = ai[sel], bi[sel], npmi[sel]
    deg = np.bincount(np.concatenate([src, dst]), minlength=n_vocab)
    print(f"connected vocab terms {int((deg > 0).sum()):,}/{n_vocab}"
          f" | isolated {int((deg == 0).sum()):,}")
    order = np.argsort(-w)[:12]
    print("\nstrongest pairs:")
    for k in order:
        print(f"  {itos[src[k]]:24}{itos[dst[k]]:24}{w[k]:.3f}")

    np.savez_compressed(OUT, src=src.astype(np.int32), dst=dst.astype(np.int32),
                        npmi=w.astype(np.float32), count=cnt[sel].astype(np.int32),
                        uni=uni, n_recipes=n_recipes, threshold=thr,
                        min_count=mc, itos=itos)
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
