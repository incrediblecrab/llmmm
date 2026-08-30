#!/usr/bin/env python3
"""End-to-end simulation of the LLMMM app as a real user would hit it.

The engine metrics (participation ratio, neighbour overlap) say whether the
geometry is healthy. They do NOT say whether a parent gets a usable dinner.
This script walks the three journeys the product actually depends on:

    1. FRIDGE   -- resolve messy real-world pantry text to the 1,790 vocab
    2. PAIR     -- which of the things she already has go together
    3. STORE    -- which single purchase most expands what she can cook
    4. SUB      -- "I'm out of X", answered from 74,465 labelled recipes
                   rather than from the embedding

Journey 1 is the one that silently kills the app: every downstream number is
meaningless for an ingredient the vocabulary cannot name, so unresolved input
is reported as a first-class failure rather than skipped.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pairing  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUBS = (ROOT / "recipe" / "expansion" / "ingredient-substitutions-74k" /
        "data" / "train-00000-of-00001.parquet")

# What is actually in a fridge, written the way a person would write it -- not
# pre-normalised to the vocabulary. Half of these are deliberately awkward.
FRIDGE = [
    "chicken breast", "rotisserie chicken", "milk", "cheddar cheese", "eggs",
    "baby carrots", "broccoli", "garlic", "onion", "soy sauce", "rice",
    "spaghetti", "ketchup", "greek yogurt", "frozen peas", "lemon",
    "olive oil", "butter", "tortillas", "Trader Joe's orange chicken",
]


def load_subs() -> dict[str, Counter]:
    """Index `Ingredient: a, b, c` lines into ingredient -> Counter(alts)."""
    import pandas as pd
    df = pd.read_parquet(SUBS, columns=["ingredients_alternatives"])
    idx: dict[str, Counter] = {}
    for blob in df["ingredients_alternatives"].dropna():
        for line in str(blob).splitlines():
            if ":" not in line:
                continue
            head, tail = line.split(":", 1)
            key = head.strip().lower()
            if not key or len(key) > 40:
                continue
            alts = [a.strip().lower() for a in tail.split(",") if a.strip()]
            idx.setdefault(key, Counter()).update(a for a in alts if len(a) < 40)
    return idx


def resolve_fridge(P, items):
    """Map free text to vocabulary, reporting misses. Mirrors real app input."""
    hit, miss = [], []
    for raw in items:
        r = P.resolve(raw)
        if r is None:
            # try dropping brand/qualifier words, which is what an app would do
            toks = re.sub(r"[^a-z ]", " ", raw.lower()).split()
            for n in range(len(toks), 0, -1):
                for i in range(len(toks) - n + 1):
                    r = P.resolve(" ".join(toks[i:i + n]))
                    if r:
                        break
                if r:
                    break
        (hit.append((raw, r)) if r else miss.append(raw))
    return hit, miss


def pantry_pairs(P, names, top=8):
    """Every in-pantry pair, ranked. This is the 'what goes together' view."""
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            v = P.pair(names[i], names[j])
            if not isinstance(v, str):
                out.append((v.overall, names[i], names[j], v.label))
    out.sort(reverse=True)
    return out[:top], out[-3:]


def best_purchase(P, names, k=10):
    """Marginal gain: which single ingredient most lifts the whole pantry.

    Scored as the mean pairing percentile of a candidate against everything she
    already owns, so it rewards an ingredient that ties the basket together
    rather than one that merely matches a single item well.
    """
    vocab = P.S["cooc"].vocab
    have = set(names)
    rows = []
    for cand in vocab:
        if cand in have:
            continue
        s = [P.pair(cand, n) for n in names]
        s = [v.overall for v in s if not isinstance(v, str)]
        if s:
            rows.append((float(np.mean(s)), cand))
    rows.sort(reverse=True)
    return rows[:k]


def main():
    P = pairing.Pairing()
    print(f"engine: {len(P.S['cooc'].vocab)} ingredients x 3 siblings\n")

    print("=" * 70)
    print("  JOURNEY 1 — she photographs / types her fridge")
    print("=" * 70)
    hit, miss = resolve_fridge(P, FRIDGE)
    for raw, r in hit:
        flag = "" if raw.lower().replace(" ", "_") == r else f"   <- '{raw}'"
        print(f"  OK   {r}{flag}")
    for m in miss:
        print(f"  MISS {m}   (not in vocabulary — app must ask or ignore)")
    print(f"\n  resolved {len(hit)}/{len(FRIDGE)}  "
          f"({100*len(hit)/len(FRIDGE):.0f}%)")

    names = sorted({r for _, r in hit})

    print("\n" + "=" * 70)
    print("  JOURNEY 2 — what she already has that goes together")
    print("=" * 70)
    best, worst = pantry_pairs(P, names)
    for sc, a, b, lab in best:
        print(f"  {sc:5.1f}  {a:<16} + {b:<16} {lab}")
    print("  ...")
    for sc, a, b, lab in worst:
        print(f"  {sc:5.1f}  {a:<16} + {b:<16} {lab}")

    print("\n" + "=" * 70)
    print("  JOURNEY 3 — one thing to buy that unlocks the most")
    print("=" * 70)
    for sc, c in best_purchase(P, names):
        print(f"  {sc:5.1f}  {c}")

    print("\n" + "=" * 70)
    print("  JOURNEY 4 — 'I'm out of X' (labelled data, not geometry)")
    print("=" * 70)
    idx = load_subs()
    print(f"  indexed {len(idx):,} ingredients with alternatives\n")
    for q in ["butter", "milk", "egg", "soy sauce", "lemon", "rice"]:
        c = idx.get(q)
        if not c:
            print(f"  {q:<12} -> (none)")
            continue
        alts = ", ".join(f"{a} ({n})" for a, n in c.most_common(4))
        print(f"  {q:<12} -> {alts}")


if __name__ == "__main__":
    main()
