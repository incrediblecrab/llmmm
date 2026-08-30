"""Build the dish library: ingredient combinations that many cooks independently
arrived at.

Why this is not "shipping recipes"
----------------------------------
We are not permitted to reproduce recipes. Every corpus behind this app is
research-only or all-rights-reserved, and contract terms bind us whatever
copyright says.

What ships here is different in kind. An entry exists only if at least
`MIN_COUNT` *different* recipes in the corpus reduce to exactly the same set of
canonical ingredients. That makes each entry a statement about the corpus —
"this combination recurs" — rather than an extract of anyone's record. It also
throws away everything that makes a recipe a work: no title, no author, no
quantities, no method, no order, no prose. What survives is a sorted list of
ingredient ids and a frequency, which is a measurement we made.

US law is unusually clear that the ingredient list alone is not protectable
(Publications Int'l v. Meredith Corp., 88 F.3d 473, 7th Cir. 1996: "the
identification of ingredients necessary for the preparation of each dish is a
statement of facts"). We do not lean on that alone — the recurrence threshold
is what keeps this on the right side of the contract terms too.

    cd tools && ../.venv/bin/python make_dishes.py
"""

from __future__ import annotations

import collections
import json
import time
from pathlib import Path

import numpy as np

MIN_COUNT = 3  # a combination must recur in at least this many recipes
MIN_LEN = 3  # two ingredients is a pairing, not a dish; we have a graph for those
MAX_LEN = 16  # beyond this it is a shopping list, and almost certainly a parse error

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/derived/app_bundle"


def main() -> None:
    t0 = time.time()
    corpus = np.load(ROOT / "data/derived/recipe_ids.npz", allow_pickle=True)
    flat, offsets = corpus["flat"], corpus["offsets"].astype(np.int64)
    n_recipes = offsets.size - 1
    names = [str(x) for x in corpus["itos"]]
    n_ing = len(names)

    print(f"reducing {n_recipes:,} recipes to ingredient sets...", flush=True)
    tally: collections.Counter[bytes] = collections.Counter()
    for i in range(n_recipes):
        tally[np.unique(flat[offsets[i] : offsets[i + 1]]).tobytes()] += 1
    print(f"  {len(tally):,} distinct sets  {time.time() - t0:.0f}s")

    kept = [
        (np.frombuffer(k, dtype=np.uint16), v)
        for k, v in tally.items()
        if v >= MIN_COUNT and MIN_LEN <= len(k) // 2 <= MAX_LEN
    ]
    kept.sort(key=lambda kv: -kv[1])
    n_dish = len(kept)
    covered = sum(v for _, v in kept)
    print(
        f"kept {n_dish:,} dishes (>= {MIN_COUNT} cooks, {MIN_LEN}-{MAX_LEN} ingredients), "
        f"covering {covered:,} recipes ({100 * covered / n_recipes:.1f}%)"
    )

    lengths = np.array([len(s) for s, _ in kept], dtype=np.int64)
    dish_flat = np.concatenate([s for s, _ in kept]).astype("<u2")
    dish_offsets = np.concatenate(([0], np.cumsum(lengths))).astype("<u4")
    dish_count = np.array([v for _, v in kept], dtype="<u4")

    # Inverted index, so the app can intersect postings instead of scanning
    # 100k+ dishes on every basket change.
    print("building inverted index...", flush=True)
    dish_of_slot = np.repeat(np.arange(n_dish, dtype=np.uint32), lengths)
    order = np.argsort(dish_flat, kind="stable")
    postings = dish_of_slot[order].astype("<u4")
    ptr = np.concatenate(
        ([0], np.cumsum(np.bincount(dish_flat.astype(np.int64), minlength=n_ing)))
    ).astype("<u4")

    dish_flat.tofile(OUT / "dishes.flat.u16")
    dish_offsets.tofile(OUT / "dishes.offsets.u32")
    dish_count.tofile(OUT / "dishes.count.u32")
    postings.tofile(OUT / "dishes.postings.u32")
    ptr.tofile(OUT / "dishes.ptr.u32")

    meta = json.loads((OUT / "graph.meta.json").read_text())
    meta["dishes"] = {
        "n": n_dish,
        "min_count": MIN_COUNT,
        "min_len": MIN_LEN,
        "max_len": MAX_LEN,
        "recipes_covered": int(covered),
        "note": "Ingredient sets recurring in >= min_count distinct recipes. "
        "No titles, quantities, order or method — those are not ours to ship.",
    }
    meta["layout"]["dishes.flat.u16"] = "uint16 LE, concatenated ascending ingredient ids"
    meta["layout"]["dishes.offsets.u32"] = "uint32 LE [n+1], row offsets into dishes.flat"
    meta["layout"]["dishes.count.u32"] = "uint32 LE [n], recipes reducing to this set"
    meta["layout"]["dishes.ptr.u32"] = "uint32 LE [n_ingredients+1], inverted index offsets"
    meta["layout"]["dishes.postings.u32"] = "uint32 LE, dish ids per ingredient, ascending"
    (OUT / "graph.meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    mb = sum(
        (OUT / f).stat().st_size
        for f in [
            "dishes.flat.u16",
            "dishes.offsets.u32",
            "dishes.count.u32",
            "dishes.postings.u32",
            "dishes.ptr.u32",
        ]
    ) / 1e6
    print(f"wrote {mb:.1f} MB  ({time.time() - t0:.0f}s total)")
    print(f"\nlongest tail: {dish_count.min()} cooks; most repeated: {dish_count.max()} cooks")
    for s, v in kept[:5]:
        print(f"  {v:>5} x  " + ", ".join(names[i] for i in s))


if __name__ == "__main__":
    main()
