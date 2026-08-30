#!/usr/bin/env python3
"""Split ii_graph edges into train/held-out for unbiased link prediction.

Training must never see the held-out edges, otherwise M4 measures memorisation
rather than generalisation. The split is by edge with a fixed seed so every
model in the study is trained and scored on exactly the same partition.

The graph is undirected but stored with both directions present, so the split
is computed on canonical (min,max) pairs and then applied to both directions.
Splitting raw rows would leak an edge's mirror into training.

    python tools/make_holdout.py --apply
"""
from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
SEED = 20260805
FRACTION = 0.10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    z = np.load(DERIVED / "ii_graph.npz", allow_pickle=True)
    src, dst = z["src"].astype(np.int64), z["dst"].astype(np.int64)

    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    key = lo * len(z["itos"]) + hi
    uniq, inverse = np.unique(key, return_inverse=True)
    print(f"rows {len(src):,}  undirected edges {len(uniq):,}")

    rng = np.random.default_rng(SEED)
    n_hold = int(round(len(uniq) * FRACTION))
    held_ids = rng.choice(len(uniq), size=n_hold, replace=False)
    is_held_edge = np.zeros(len(uniq), bool)
    is_held_edge[held_ids] = True
    row_held = is_held_edge[inverse]

    print(f"held-out edges {n_hold:,} ({FRACTION:.0%})  rows removed {row_held.sum():,}")
    keep = ~row_held

    # a node stranded with no training edges cannot be learned at all.
    # load_ii() symmetrises, so degree counts both endpoints.
    deg = (np.bincount(src[keep], minlength=len(z["itos"]))
           + np.bincount(dst[keep], minlength=len(z["itos"])))
    stranded = int((deg == 0).sum())
    print(f"nodes with no training edge: {stranded}")

    if not a.apply:
        print("dry run; pass --apply to write")
        return

    out_train = DERIVED / "ii_graph_train.npz"
    np.savez_compressed(
        out_train,
        src=src[keep].astype(np.int32), dst=dst[keep].astype(np.int32),
        npmi=z["npmi"][keep], count=z["count"][keep],
        uni=z["uni"], n_recipes=z["n_recipes"], threshold=z["threshold"],
        min_count=z["min_count"], itos=z["itos"])

    held_lo, held_hi = uniq[is_held_edge] // len(z["itos"]), uniq[is_held_edge] % len(z["itos"])
    out_hold = DERIVED / "ii_graph_heldout.npz"
    np.savez_compressed(out_hold, src=held_lo.astype(np.int32),
                        dst=held_hi.astype(np.int32), seed=SEED,
                        fraction=FRACTION, itos=z["itos"])
    print(f"wrote {out_train.name} and {out_hold.name}")


if __name__ == "__main__":
    main()
