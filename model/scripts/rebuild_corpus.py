#!/usr/bin/env python
"""Rebuild the recipe corpus with the corrected normaliser.

    ./.venv/bin/python scripts/rebuild_corpus.py --out recipe_ids_v2.npz

This mirrors llmmm's `normalize.dump()` exactly — same readers, same
deduplication by raw ingredient text, same "drop recipes that matched nothing"
rule, same `sorted(set(ids))` storage — and changes one thing: which normaliser
decides what an ingredient line means.

Everything else is held fixed on purpose. The vocabulary is byte-identical, so
embeddings trained on the rebuilt corpus stay row-comparable with every run
already recorded, and a leaderboard difference can only come from the coverage
fix. Writing to a new filename rather than overwriting keeps the old corpus
available, which is what makes the comparison possible at all.

One consequence worth stating plainly: better coverage rescues recipes that
previously matched nothing and were dropped, so the rebuilt corpus has *more*
recipes and the indices do not line up with the old one. The text index is
aligned to the corpus it was built from and must be rebuilt alongside it.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingredient_model.config import PATHS  # noqa: E402
from ingredient_model.data.normalizer import get_normalizer  # noqa: E402

LLMMM = PATHS.prior_tools


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="recipe_ids_v2.npz")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--upstream", action="store_true",
                    help="rebuild with the unmodified normaliser, to prove "
                         "this script reproduces the existing corpus")
    a = ap.parse_args()

    sys.path.insert(0, str(LLMMM))
    import corpus  # type: ignore

    nz = get_normalizer(fix_zh_qty=not a.upstream, extra=not a.upstream)
    if not a.upstream:
        print(f"aliases: {nz.added}")
        if nz.rejected:
            print(f"REJECTED {len(nz.rejected)} aliases naming unknown "
                  f"concepts: {list(nz.rejected)[:5]}", file=sys.stderr)
            return 1

    flat: list[int] = []
    off = [0]
    langs: list[str] = []
    srcs: list[str] = []
    seen = dup = empty = 0
    fp: set[int] = set()
    t0 = time.time()

    for key, lang, items in corpus.iter_all(a.limit):
        seen += 1
        h = hash("\u241f".join(sorted(str(i).strip().lower() for i in items)))
        if h in fp:
            dup += 1
            continue
        fp.add(h)
        ids = nz.normalize(lang, items)
        if not ids:
            empty += 1
            continue
        flat.extend(sorted(ids))
        off.append(len(flat))
        langs.append(lang)
        srcs.append(key)
        if seen % 500_000 == 0:
            print(f"  {seen:,} scanned, {len(off) - 1:,} kept, {dup:,} dup, "
                  f"{empty:,} empty", flush=True)

    dest = PATHS.recipes / a.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest,
        flat=np.asarray(flat, dtype=np.uint16),
        offsets=np.asarray(off, dtype=np.int64),
        lang=np.asarray(langs), source=np.asarray(srcs),
        itos=np.asarray(nz.itos),
    )
    kept = len(off) - 1
    print(f"\nscanned {seen:,} | dup {dup:,} | matched nothing {empty:,} | "
          f"kept {kept:,} | {len(flat):,} slots "
          f"({len(flat) / max(kept, 1):.2f}/recipe) in "
          f"{(time.time() - t0) / 60:.1f} min\n-> {dest}")

    old = PATHS.recipes / "recipe_ids.npz"
    if old.exists() and old != dest:
        with np.load(old, allow_pickle=False) as z:
            o_off, o_flat = z["offsets"], z["flat"]
        o_kept = len(o_off) - 1
        print(f"\nagainst the existing corpus:")
        print(f"  recipes     {o_kept:,} -> {kept:,} "
              f"({kept / o_kept - 1:+.2%})")
        print(f"  ingredients {len(o_flat):,} -> {len(flat):,} "
              f"({len(flat) / len(o_flat) - 1:+.2%})")
        print(f"  per recipe  {len(o_flat) / o_kept:.2f} -> "
              f"{len(flat) / kept:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
