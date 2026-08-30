#!/usr/bin/env python3
"""Make a rebuilt corpus the canonical one, and record that it happened.

    python scripts/promote_corpus.py recipe_ids_v2.npz          # report only
    python scripts/promote_corpus.py recipe_ids_v2.npz --apply

``rebuild_corpus.py`` writes a candidate beside the corpus in use rather than
over it, which is what makes a before/after comparison possible at all. This is
the other half: the step that swaps them once the candidate is trusted.

Three things have to happen together or the workspace is left inconsistent, and
"inconsistent" here means every number produced afterwards is quietly wrong
rather than obviously broken:

1. The candidate becomes ``recipe_ids.npz`` and the incumbent is archived under
   its generation name, so the swap is reversible.
2. ``ii_graph.npz`` is rebuilt from it. The full graph is a *derived summary of
   the corpus* — a corpus with 924,315 more ingredient occurrences implies a
   different graph, and ``build_splits.py`` computes held-out edges as the pairs
   the full graph supports but the training graph does not. Leaving a stale
   graph in place produces held-out labels that describe the old corpus.
3. ``GENERATION.json`` records what is now canonical, so every run manifest
   written afterwards names the corpus that produced it.

What this does *not* do is rebuild the recipe-level split or the text index.
Both are separate, slower steps with their own scripts, and both are checked
here so that a half-promoted tree reports itself:

    python scripts/build_splits.py --apply
    python scripts/build_text_index.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ingredient_model.config import PATHS, SEED  # noqa: E402
from ingredient_model.data.build import (MIN_COUNT, THRESHOLD,  # noqa: E402
                                         npmi_graph)
from ingredient_model.data.graphs import GRAPH_FULL  # noqa: E402
from ingredient_model.data.recipes import RECIPE_IDS, load_recipes  # noqa: E402
from ingredient_model.data.splits import SPLITS  # noqa: E402

CANONICAL = RECIPE_IDS


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _describe(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {"recipes": int(len(z["offsets"]) - 1),
                "slots": int(len(z["flat"])),
                "vocab": int(len(z["itos"])),
                "itos_sha": hashlib.sha256(
                    "\x1f".join(map(str, z["itos"])).encode()).hexdigest()[:16]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", nargs="?", default="recipe_ids_v2.npz",
                    help="filename under data/recipes/ to promote")
    ap.add_argument("--generation", default=None,
                    help="name for the new generation "
                         "(default: inferred from the candidate filename)")
    ap.add_argument("--normalizer", default="corrected",
                    choices=("corrected", "base"),
                    help="which normaliser built the candidate. 'corrected' is "
                         "rebuild_corpus.py's default; 'base' is the prior "
                         "study's unmodified one, i.e. --upstream. Recorded so "
                         "sanity_check.py can replay the raw sources through "
                         "the same one rather than guessing.")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    recipes = PATHS.recipes
    cand = recipes / a.candidate
    live = recipes / CANONICAL
    if not cand.exists():
        sys.exit(f"no candidate at {cand}")
    if cand.resolve() == live.resolve():
        sys.exit(f"{a.candidate} is already the canonical corpus")

    stem = Path(a.candidate).stem
    gen = a.generation or (stem.rsplit("_", 1)[-1]
                           if stem.startswith("recipe_ids_") else stem)

    new = _describe(cand)
    old = _describe(live) if live.exists() else None
    print(f"candidate  {a.candidate}")
    print(f"  {new['recipes']:,} recipes, {new['slots']:,} slots, "
          f"vocab {new['vocab']}")
    if old:
        print(f"incumbent  {CANONICAL}")
        print(f"  {old['recipes']:,} recipes, {old['slots']:,} slots, "
              f"vocab {old['vocab']}")
        print(f"\ndelta  recipes {new['recipes'] - old['recipes']:+,} "
              f"({new['recipes'] / old['recipes'] - 1:+.3%})   "
              f"slots {new['slots'] - old['slots']:+,} "
              f"({new['slots'] / old['slots'] - 1:+.3%})")

        # The vocabulary is the row index of every embedding ever trained here.
        # If it moved, no stored result is row-comparable with a new one and the
        # whole point of rebuilding to a fixed vocabulary has been lost — so
        # this refuses rather than warns.
        if new["itos_sha"] != old["itos_sha"]:
            sys.exit(
                "\nREFUSING: the vocabulary differs between the two corpora.\n"
                "Embedding rows are vocabulary positions, so promoting this "
                "would silently invalidate every stored run rather than "
                "produce a comparable leaderboard.")
        print(f"vocabulary identical ({new['vocab']} concepts) — "
              f"embeddings stay row-comparable")

    archive = recipes / f"recipe_ids_{_incumbent_name(live)}.npz"
    print(f"\nplan:")
    print(f"  archive  {CANONICAL} -> {archive.name}")
    print(f"  promote  {a.candidate} -> {CANONICAL}")
    print(f"  rebuild  {GRAPH_FULL} from the promoted corpus")
    print(f"  record   GENERATION.json  (generation {gen!r})")

    if not a.apply:
        print("\ndry run; pass --apply to write")
        return 0

    if live.exists():
        if archive.exists():
            sys.exit(f"archive target {archive.name} already exists — "
                     f"refusing to overwrite it")
        shutil.move(str(live), str(archive))
        print(f"\narchived {CANONICAL} -> {archive.name}")
    shutil.move(str(cand), str(live))
    print(f"promoted {a.candidate} -> {CANONICAL}")

    corpus = load_recipes()
    print(f"\nrebuilding {GRAPH_FULL} from {corpus.n_recipes:,} recipes "
          f"(min_count={MIN_COUNT}, npmi>={THRESHOLD}) ...")
    t0 = time.time()
    g = npmi_graph(corpus, MIN_COUNT, THRESHOLD, progress=False)
    np.savez_compressed(PATHS.graphs / GRAPH_FULL, **g)
    print(f"  {len(g['src']):,} edges in {time.time() - t0:.0f}s")

    marker = {
        "generation": gen,
        "corpus": CANONICAL,
        "normalizer": a.normalizer,
        "promoted_from": a.candidate,
        "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "recipes": new["recipes"],
        "slots": new["slots"],
        "vocab": new["vocab"],
        "sha256": _sha256(live),
        "ii_graph_edges": int(len(g["src"])),
        "min_count": MIN_COUNT,
        "threshold": THRESHOLD,
        "seed": SEED,
        "previous": {"generation": _incumbent_name(archive), **old} if old else None,
    }
    PATHS.generation_file.write_text(json.dumps(marker, indent=2) + "\n")
    print(f"recorded {PATHS.generation_file.name}")

    print("\nSTALE — these are derived from the corpus and must be rebuilt "
          "before any model trains:")
    split = SPLITS["recipe-holdout"]
    for label, p, cmd in (
            ("recipe-holdout split", PATHS.recipes / split.corpus,
             "python scripts/build_splits.py --apply"),
            ("text index", PATHS.recipes / "recipe_text.parquet",
             "python scripts/build_text_index.py")):
        state = "present but stale" if p.exists() else "absent"
        print(f"  {label:22s} {state:18s} {cmd}")
    return 0


def _incumbent_name(path: Path) -> str:
    """Generation label for the corpus being replaced.

    Taken from the marker if one exists, so a tree promoted twice archives to
    distinct names rather than colliding on a guess.
    """
    from ingredient_model.config import corpus_generation
    g = corpus_generation().get("generation", "unknown")
    return "v1" if g in ("unknown", "unreadable") else str(g)


if __name__ == "__main__":
    raise SystemExit(main())
