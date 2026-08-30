#!/usr/bin/env python3
"""Build the *unpruned* NPMI ingredient-ingredient graph (H8).

`build_ii_graph.py` searches for whichever NPMI threshold reproduces the
paper's 203,508 edges. That is the right thing to do when replicating, but it
is a replication constraint, not a modelling decision: of the 284,919 pairs
that genuinely co-occur somewhere in 4.6M recipes, it keeps 203,504 and
discards 81,415 -- and severing those edges leaves 47 ingredients with no
neighbour at all, which is a cold-start problem manufactured by the threshold
rather than by the data.

This keeps every observed pair. The held-out edges are removed exactly as in
`make_holdout.py`, so M4 stays leak-free and directly comparable to every
other model in the study: the evaluation set is unchanged, only the training
graph grows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from build_ii_graph import cooccurrence

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
IDS = DERIVED / "recipe_ids.npz"
HELDOUT = DERIVED / "ii_graph_heldout.npz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-count", type=int, default=1,
                    help="pairs seen in fewer recipes than this are dropped; "
                         "1 keeps everything observed")
    ap.add_argument("--out", default="ii_graph_full_train.npz")
    a = ap.parse_args()

    z = np.load(IDS, allow_pickle=True)
    flat, offsets, itos = z["flat"].astype(np.int64), z["offsets"], z["itos"]
    n_vocab, n_recipes = len(itos), len(offsets) - 1
    print(f"{n_recipes:,} recipes, vocab {n_vocab}")

    src, dst, cnt, uni = cooccurrence(flat, offsets, n_vocab)
    cnt = cnt.astype(np.float64)
    print(f"\nobserved pairs {len(cnt):,}")

    keep = cnt >= a.min_count
    src, dst, cnt = src[keep], dst[keep], cnt[keep]
    pij = cnt / n_recipes
    pmi = np.log(pij / ((uni[src] / n_recipes) * (uni[dst] / n_recipes)))
    npmi = pmi / -np.log(pij)
    print(f"after min_count>={a.min_count}: {len(cnt):,}")

    # Drop exactly the evaluation edges. Pairs are upper-triangular in both
    # files, so a single flattened code is an exact key.
    h = np.load(HELDOUT, allow_pickle=True)
    held = set((h["src"].astype(np.int64) * n_vocab
                + h["dst"].astype(np.int64)).tolist())
    code = src.astype(np.int64) * n_vocab + dst.astype(np.int64)
    mask = ~np.isin(code, np.fromiter(held, np.int64, len(held)))
    n_removed = int((~mask).sum())
    print(f"removed {n_removed:,} held-out edges "
          f"({n_removed}/{len(held)} of the eval set found)")
    src, dst, cnt, npmi = src[mask], dst[mask], cnt[mask], npmi[mask]

    deg = np.bincount(np.concatenate([src, dst]), minlength=n_vocab)
    print(f"train edges {len(src):,} | connected {int((deg>0).sum()):,}"
          f"/{n_vocab} | isolated {int((deg==0).sum()):,}")

    out = DERIVED / a.out
    np.savez_compressed(out, src=src.astype(np.int32), dst=dst.astype(np.int32),
                        npmi=npmi.astype(np.float32),
                        count=cnt.astype(np.int32), uni=uni,
                        n_recipes=n_recipes, threshold=-np.inf,
                        min_count=a.min_count, itos=itos)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
