#!/usr/bin/env python3
"""Build the recipe-level evaluation split.

Holds out 10% of *recipes*, rebuilds the co-occurrence graph from the remaining
90% at the same fixed operating point, and takes as held-out edges those pairs
which the full corpus supports but the training corpus does not.

This is the only protocol under which graph models and recipe models can be
compared. Under the edge-level split a recipe model still sees every recipe that
produced a "held-out" pair, so its link-prediction score is memorisation.

    python scripts/build_splits.py            # report only
    python scripts/build_splits.py --apply
"""
from __future__ import annotations

import argparse

import numpy as np

from ingredient_model.config import PATHS, SEED
from ingredient_model.data.build import MIN_COUNT, THRESHOLD, edge_keys, npmi_graph
from ingredient_model.data.graphs import GRAPH_FULL, load_ii_graph
from ingredient_model.data.recipes import load_recipes
from ingredient_model.data.splits import SPLITS

FRACTION = 0.30
"""Chosen for statistical power, not by convention.

Held-out edges here are pairs the training recipes no longer support, which is a
much smaller set than the fraction of recipes removed: at 10% only 6,637 edges
drop out, giving a confidence interval nearly twice the width of the edge-level
protocol's. 30% yields 20,980 — matching the edge protocol's 20,350 to within
3%, so the two are equally powered and directly comparable — while still leaving
3.25M recipes to train on.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--fraction", type=float, default=FRACTION)
    a = ap.parse_args()

    split = SPLITS["recipe-holdout"]
    corpus = load_recipes()
    full = load_ii_graph(GRAPH_FULL)
    n = corpus.n_vocab
    print(f"corpus {corpus.n_recipes:,} recipes, vocab {n}")
    print(f"full graph {full.n_edges:,} edges "
          f"(min_count={MIN_COUNT}, npmi>={THRESHOLD})")

    train, test = corpus.split(frac=a.fraction, seed=SEED)
    print(f"train {train.n_recipes:,}  test {test.n_recipes:,}")

    print("rebuilding graph from the training recipes...")
    g = npmi_graph(train, MIN_COUNT, THRESHOLD, progress=False)
    print(f"train graph {len(g['src']):,} edges")

    full_keys = edge_keys(np.asarray(full.src), np.asarray(full.dst), n)
    train_keys = edge_keys(g["src"].astype(np.int64), g["dst"].astype(np.int64), n)
    held_mask = ~np.isin(full_keys, train_keys)
    hu = np.asarray(full.src)[held_mask]
    hv = np.asarray(full.dst)[held_mask]
    print(f"held-out edges {len(hu):,} "
          f"({len(hu) / full.n_edges:.1%} of the full graph) — pairs the "
          f"training recipes do not support")
    print("  note: these are rare pairs by construction. A pair that appears in "
          "millions\n  of recipes cannot be hidden from a recipe model without "
          "gutting the corpus,\n  so honest recipe-level link prediction is "
          "necessarily a rare-pair task.")

    deg = (np.bincount(g["src"].astype(np.int64), minlength=n)
           + np.bincount(g["dst"].astype(np.int64), minlength=n))
    print(f"nodes with no training edge: {int((deg == 0).sum())}")
    if len(hu) < 2000:
        print("  WARNING: too few held-out edges for a stable AUC; "
              "raise --fraction")

    if not a.apply:
        print("\ndry run; pass --apply to write")
        return 0

    PATHS.ensure()
    np.savez_compressed(PATHS.graphs / split.graph, **g)
    np.savez_compressed(
        PATHS.graphs / split.heldout, src=hu.astype(np.int32),
        dst=hv.astype(np.int32), seed=np.int64(SEED),
        fraction=np.float64(a.fraction), itos=np.array(corpus.itos))
    np.savez_compressed(
        PATHS.recipes / split.corpus, flat=train.flat, offsets=train.offsets,
        lang=train.lang, source=train.source, itos=np.array(train.itos))
    print(f"\nwrote {split.graph}, {split.heldout}, {split.corpus}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
