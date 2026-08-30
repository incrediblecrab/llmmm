"""Build the co-occurrence portrait: for every pair in the shipped graph, what
else is in the recipes that contain both.

Why this exists
---------------
The app tells you "chicken and saffron appear together in 1,247 recipes" and
then asks you to take that on faith. You cannot check it. Every recipe source
in the corpus is non-commercial or all-rights-reserved (see LICENSE_AUDIT.md),
so we cannot show you a single recipe title, let alone its text.

What we can show is what those 1,247 recipes are *made of*. Ingredient sets are
facts, and aggregate statistics over them are our own measurements rather than
anyone's expression. So instead of one arbitrary recipe we give the shape of
all of them at once: rice in 71%, onion in 64%, garlic in 41%. That is paella,
described without reproducing a word of anybody's cookbook.

It is also the one thing a language model cannot do. Ask any of them what
fraction of chicken-and-saffron recipes contain rice and you will get a number
that was never counted.

Output aligns slot-for-slot with the existing CSR arrays, so the app needs no
new index: portrait row `s` describes the pair on edge `s` of graph.indices.

    cd tools && ../.venv/bin/python make_triples.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

K = 10  # thirds kept per pair
ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "data/derived/app_bundle"
OUT = ROOT / "data/derived/app_bundle"


def ragged_gather(flat: np.ndarray, offsets: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Concatenate the CSR segments for `rows` without a Python loop."""
    starts = offsets[rows]
    lens = offsets[rows + 1] - starts
    total = int(lens.sum())
    if total == 0:
        return np.empty(0, dtype=flat.dtype)
    # Build the flat index list: repeat each start, then add a within-segment ramp.
    idx = np.repeat(starts - np.concatenate(([0], np.cumsum(lens[:-1]))), lens)
    idx += np.arange(total, dtype=np.int64)
    return flat[idx]


def main() -> None:
    corpus = np.load(ROOT / "data/derived/recipe_ids.npz", allow_pickle=True)
    flat = corpus["flat"].astype(np.uint16)
    offsets = corpus["offsets"].astype(np.int64)
    n_recipes = offsets.size - 1
    print(f"corpus: {n_recipes:,} recipes, {flat.size:,} ingredient slots")

    meta = json.loads((BUNDLE / "graph.meta.json").read_text())
    n_ing = meta["n_ingredients"]
    indptr = np.fromfile(BUNDLE / "graph.indptr.u32", dtype="<u4").astype(np.int64)
    indices = np.fromfile(BUNDLE / "graph.indices.u16", dtype="<u2").astype(np.int64)
    nnz = indices.size
    print(f"graph:  {n_ing} ingredients, {nnz:,} directed edges")

    # Column view: for each ingredient, the sorted recipe ids containing it.
    # Sorting 35.8M (ingredient, recipe) pairs once beats 448k linear scans.
    print("inverting corpus to postings lists...", flush=True)
    t0 = time.time()
    rec_of_slot = np.repeat(
        np.arange(n_recipes, dtype=np.int64), np.diff(offsets)
    )
    order = np.argsort(flat, kind="stable")
    postings = rec_of_slot[order]
    col_ptr = np.concatenate(([0], np.cumsum(np.bincount(flat, minlength=n_ing)))).astype(
        np.int64
    )
    del rec_of_slot, order
    print(f"  {time.time() - t0:.1f}s")

    top_ids = np.zeros((nnz, K), dtype="<u2")
    top_pm = np.zeros((nnz, K), dtype="<u2")  # permille of the pair's recipes
    pair_count = np.zeros(nnz, dtype="<u4")  # exact, unlike the saturated u16

    # Each pair is counted once, from whichever endpoint is RARER: we mark the
    # common ingredient's recipes in a bitmap, then walk the rare ingredient's
    # much shorter postings list through it. Always scanning the short side
    # costs 2.0e9 element reads instead of 2.1e10 — the difference between
    # minutes and most of an afternoon.
    print("counting thirds...", flush=True)
    t0 = time.time()
    src = np.repeat(np.arange(n_ing, dtype=np.int64), np.diff(indptr))
    freq = col_ptr[1:] - col_ptr[:-1]
    mask = np.zeros(n_recipes, dtype=bool)
    owned = np.zeros(nnz, dtype=bool)
    done = 0

    for a in range(n_ing):
        lo, hi = int(indptr[a]), int(indptr[a + 1])
        if lo == hi:
            continue
        nbrs = indices[lo:hi]
        # Own the pair when we are the commoner endpoint; ties go to the lower id.
        mine = (freq[nbrs] < freq[a]) | ((freq[nbrs] == freq[a]) & (nbrs > a))
        slots = np.nonzero(mine)[0] + lo
        if slots.size == 0:
            continue
        owned[slots] = True
        ra = postings[col_ptr[a] : col_ptr[a + 1]]
        mask[ra] = True
        for slot in slots:
            b = int(indices[slot])
            rb = postings[col_ptr[b] : col_ptr[b + 1]]
            both = rb[mask[rb]]
            n_both = both.size
            pair_count[slot] = n_both
            if n_both == 0:
                continue
            counts = np.bincount(ragged_gather(flat, offsets, both), minlength=n_ing)
            counts[a] = 0
            counts[b] = 0
            k = min(K, int((counts > 0).sum()))
            if k == 0:
                continue
            part = np.argpartition(counts, -k)[-k:]
            part = part[np.argsort(-counts[part], kind="stable")]
            top_ids[slot, :k] = part
            top_pm[slot, :k] = np.round(counts[part] * 1000.0 / n_both).astype(np.uint16)
        mask[ra] = False
        done += slots.size
        if a % 100 == 0:
            print(f"  ingredient {a}/{n_ing}  {done:,} pairs  {time.time() - t0:.0f}s", flush=True)
    print(f"counted {done:,} pairs in {time.time() - t0:.0f}s")

    # Mirror i<j onto j>i so every directed slot is populated.
    print("mirroring...", flush=True)
    key = np.minimum(src, indices) * n_ing + np.maximum(src, indices)
    canon = np.argsort(key, kind="stable")
    # The two directed slots for a pair are adjacent once sorted by key. Copy
    # from whichever one the loop above claimed.
    pairs = canon.reshape(-1, 2)
    have = np.where(owned[pairs[:, 0]], pairs[:, 0], pairs[:, 1])
    want = np.where(owned[pairs[:, 0]], pairs[:, 1], pairs[:, 0])
    assert owned[have].all(), "every pair must have been counted exactly once"
    top_ids[want] = top_ids[have]
    top_pm[want] = top_pm[have]
    pair_count[want] = pair_count[have]

    top_ids.tofile(OUT / "triples.ids.u16")
    top_pm.tofile(OUT / "triples.pm.u16")
    pair_count.tofile(OUT / "graph.count.u32")

    saturated = int((pair_count > 65535).sum())
    meta["layout"]["triples.ids.u16"] = f"uint16 LE [nnz][{K}], third-ingredient ids, 0-padded"
    meta["layout"]["triples.pm.u16"] = f"uint16 LE [nnz][{K}], permille of the pair's recipes"
    meta["layout"]["graph.count.u32"] = "uint32 LE [nnz], exact pair count (u16 saturates)"
    meta["triples_k"] = K
    meta["saturated_u16_edges"] = saturated
    (OUT / "graph.meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"\nwrote {nnz:,} portraits, K={K}")
    print(f"edges where the shipped u16 count was WRONG (>65535): {saturated:,}")
    ex = int(np.argmax(pair_count))
    itos = json.loads((BUNDLE / "vocab.json").read_text())["ingredients"]
    print(f"largest pair: {itos[int(src[ex])]} + {itos[int(indices[ex])]} = {pair_count[ex]:,}")
    for i in range(K):
        t = int(top_ids[ex, i])
        if top_pm[ex, i]:
            print(f"    {itos[t]:<24} {top_pm[ex, i] / 10:.1f}%")


if __name__ == "__main__":
    main()
