#!/usr/bin/env python3
"""Populate the workspace from the prior study's derived artefacts.

Copies graphs, the tokenised corpus and the evaluation catalogue into this
workspace's layout. Roughly 50 MB. The raw recipe corpus is *not* copied — it
lives once at ``<repo>/raw-data`` and nothing here re-derives it routinely.

    python scripts/import_data.py

``--from`` overrides the source, which is only needed for a prior-study tree
held outside the repository.

Idempotent: existing files are skipped unless --force is given.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO))

from ingredient_model.config import PATHS  # noqa: E402

# (source relative to the prior-study tree, destination relative to data/)
FILES = [
    ("data/derived/ii_graph.npz", "graphs/ii_graph.npz"),
    ("data/derived/ii_graph_train.npz", "graphs/ii_graph_train.npz"),
    ("data/derived/ii_graph_heldout.npz", "graphs/ii_graph_heldout.npz"),
    ("data/derived/ii_graph_full_train.npz", "graphs/ii_graph_full_train.npz"),
    ("data/derived/ii_graph_recipe_train.npz", "graphs/ii_graph_recipe_train.npz"),
    ("data/derived/flavor_graph.npz", "graphs/flavor_graph.npz"),
    ("data/derived/recipe_ids.npz", "recipes/recipe_ids.npz"),
    ("data/derived/recipe_cooc.npz", "recipes/recipe_cooc.npz"),
    ("data/catalog/substitutions.parquet", "catalog/substitutions.parquet"),
    ("data/catalog/ingredients.csv", "catalog/ingredients.csv"),
    ("data/catalog/ingredients.parquet", "catalog/ingredients.parquet"),
]

OPTIONAL = {"graphs/ii_graph_full_train.npz", "graphs/ii_graph_recipe_train.npz",
            "recipes/recipe_cooc.npz", "catalog/ingredients.csv",
            "catalog/ingredients.parquet"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default=None,
                    help="path to the prior-study tree "
                         "(default: <repo>/prior-study)")
    ap.add_argument("--to", dest="dst", default=str(REPO / "data"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    src = Path(a.src).expanduser() if a.src else PATHS.prior_study
    dst = Path(a.dst).expanduser()
    if not src.exists():
        sys.exit(f"source not found: {src}")

    copied = skipped = missing = 0
    for rel_src, rel_dst in FILES:
        s, d = src / rel_src, dst / rel_dst
        if not s.exists():
            if rel_dst not in OPTIONAL:
                print(f"  MISSING  {rel_src}")
                missing += 1
            continue
        if d.exists() and not a.force:
            skipped += 1
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
        print(f"  copied   {rel_dst}  ({d.stat().st_size / 1e6:.1f} MB)")
        copied += 1

    corpus = REPO.parent / "raw-data"
    if corpus.exists():
        print(f"\n  raw corpus in place at {corpus}")

    print(f"\n{copied} copied, {skipped} already present, {missing} missing")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
