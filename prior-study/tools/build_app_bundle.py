#!/usr/bin/env python3
"""Build the shippable on-device graph from the full 4.65M-recipe corpus.

The bundle currently in data/derived/bundle ships an NPMI table built from 2.1M
English recipes, and its own metadata flags the consequence: "Skews American
home cooking: miso appears 272x, butter 87,215x". This rebuilds the same
structure from all 4,647,847 multilingual recipes.

Format change from the old `keys.bin` pair list: CSR. The old layout required a
binary search over 186k sorted uint64 keys for every (context, candidate) pair,
which is 1,790 searches per basket item. CSR walks only the neighbours that
actually exist -- roughly 250 per ingredient -- so scoring a 10-item basket
touches ~2.5k entries instead of ~18k searches, and the same structure still
answers single-pair lookups by binary searching within one row.

Emitted (all little-endian):
  graph.indptr.u32    [1791] uint32   row start offsets into the arrays below
  graph.indices.u16   [nnz]  uint16   neighbour ingredient id, ascending per row
  graph.npmi.f16      [nnz]  float16  NPMI, unthresholded so the pairing screen
                                      can still report negative evidence
  graph.count.u16     [nnz]  uint16   co-occurrence count, saturated at 65535
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_ii_graph import cooccurrence

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
OUT = DERIVED / "app_bundle"
MIN_COUNT = 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-count", type=int, default=MIN_COUNT)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    z = np.load(DERIVED / "recipe_ids.npz", allow_pickle=True)
    flat, offs, itos = z["flat"].astype(np.int64), z["offsets"], z["itos"]
    n_vocab, n_rec = len(itos), len(offs) - 1
    print(f"{n_rec:,} recipes, {n_vocab:,} ingredients")

    src, dst, cnt, uni = cooccurrence(flat, offs, n_vocab)
    keep = cnt >= a.min_count
    src, dst, cnt = src[keep], dst[keep], cnt[keep].astype(np.float64)
    pij = cnt / n_rec
    npmi = (np.log(pij / ((uni[src] / n_rec) * (uni[dst] / n_rec)))
            / -np.log(pij))
    print(f"{len(src):,} pairs at min_count>={a.min_count} | "
          f"npmi {npmi.min():.3f}..{npmi.max():.3f}")

    # Symmetrise: every pair appears in both rows so a row walk is complete.
    i = np.concatenate([src, dst])
    j = np.concatenate([dst, src])
    w = np.concatenate([npmi, npmi])
    c = np.concatenate([cnt, cnt])
    order = np.lexsort((j, i))
    i, j, w, c = i[order], j[order], w[order], c[order]

    indptr = np.zeros(n_vocab + 1, np.uint32)
    indptr[1:] = np.cumsum(np.bincount(i, minlength=n_vocab))
    deg = np.diff(indptr)
    print(f"nnz {len(j):,} | degree min {deg.min()} median "
          f"{int(np.median(deg))} max {deg.max()} | isolated "
          f"{int((deg == 0).sum())}")

    (out / "graph.indptr.u32").write_bytes(indptr.astype("<u4").tobytes())
    (out / "graph.indices.u16").write_bytes(j.astype("<u2").tobytes())
    (out / "graph.npmi.f16").write_bytes(w.astype("<f2").tobytes())
    (out / "graph.count.u16").write_bytes(
        np.minimum(c, 65535).astype("<u2").tobytes())

    freq = uni.astype(np.int64)
    (out / "vocab.json").write_text(json.dumps(
        {"ingredients": [str(s) for s in itos],
         "freq": freq.tolist()}))

    meta = {
        "n_ingredients": int(n_vocab),
        "n_recipes": int(n_rec),
        "n_pairs": int(len(src)),
        "nnz": int(len(j)),
        "min_count": int(a.min_count),
        "isolated": int((deg == 0).sum()),
        "layout": {
            "graph.indptr.u32": "uint32 LE [n_ingredients+1], CSR row offsets",
            "graph.indices.u16": "uint16 LE [nnz], neighbour id, ascending per row",
            "graph.npmi.f16": "float16 LE [nnz], unthresholded",
            "graph.count.u16": "uint16 LE [nnz], saturated at 65535",
            "vocab.json": "ingredients[i] names row i; freq[i] is recipe count",
        },
        "ranker": {
            "note": "See tools/reference_ranker.py. Swift must match it exactly.",
            "npmi_floor": -0.15,
            "beta": 0.5,
        },
        "supersedes": "recipe_cooc.* (2.1M English-only recipes)",
    }
    (out / "graph.meta.json").write_text(json.dumps(meta, indent=2))
    total = sum(f.stat().st_size for f in out.iterdir())
    print(f"wrote {out} | {total / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
