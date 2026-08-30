#!/usr/bin/env python3
"""Calibration + regression test for the pairing engine.

There is no public ground truth for ingredient pairing, so we calibrate against
two hand-built reference sets: pairings any cook would call canonical, and
pairings that should not work. The engine must separate them.

This doubles as a regression test — if a change to the scoring drops the AUC or
lets a canonical pairing fall into "clash", the change is wrong.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pairing import Pairing  # noqa: E402

CANONICAL = [
    ("tomato", "basil"), ("lamb", "mint"), ("garlic", "olive_oil"),
    ("chocolate", "orange"), ("apple", "cinnamon"), ("salmon", "dill"),
    ("pork", "fennel"), ("beef", "black_pepper"), ("lemon", "thyme"),
    ("duck", "orange"), ("shrimp", "garlic"), ("mozzarella_cheese", "tomato"),
    ("chicken", "rosemary"), ("potato", "rosemary"), ("carrot", "ginger"),
    ("mushroom", "thyme"), ("coconut_milk", "lemongrass"), ("soy_sauce", "ginger"),
    ("cucumber", "dill"), ("egg", "chive"),
]
IMPLAUSIBLE = [
    ("banana", "parsley"), ("coffee", "garlic"), ("chocolate", "fish_sauce"),
    ("marshmallow", "anchovy"), ("yogurt", "bacon"), ("oyster", "maple_syrup"),
    ("mustard", "strawberry"), ("vanilla", "horseradish"), ("tuna", "caramel"),
    ("blue_cheese", "watermelon"),
]


def score(P, pairs):
    out = []
    for a, b in pairs:
        v = P.pair(a, b)
        if isinstance(v, str):
            print(f"   skip {a}+{b}: {v}")
            continue
        out.append((f"{a}+{b}", v.overall, v.label))
    return out


def auc(pos, neg):
    """Probability a random canonical pair outranks a random implausible one."""
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main():
    P = Pairing()
    C, I = score(P, CANONICAL), score(P, IMPLAUSIBLE)
    cv = np.array([x[1] for x in C])
    iv = np.array([x[1] for x in I])

    print(f"\n{'='*66}\nCALIBRATION\n{'='*66}")
    print(f"  canonical   (n={len(cv):2d}): min={cv.min():5.1f}  median={np.median(cv):5.1f}  max={cv.max():5.1f}")
    print(f"  implausible (n={len(iv):2d}): min={iv.min():5.1f}  median={np.median(iv):5.1f}  max={iv.max():5.1f}")
    a = auc(cv, iv)
    print(f"\n  separation AUC = {a:.3f}   (1.0 = perfect, 0.5 = coin flip)")

    fails = [(n, s, l) for n, s, l in C if l == "clash"]
    fps = [(n, s, l) for n, s, l in I if l in ("excellent", "strong")]
    print(f"  canonical mislabelled 'clash' : {len(fails)}  {[n for n,_,_ in fails]}")
    print(f"  implausible sold as strong    : {len(fps)}  {[n for n,_,_ in fps]}")

    print("\n  canonical, ranked:")
    for n, s, l in sorted(C, key=lambda x: -x[1]):
        print(f"     {s:5.1f}  {l:<12} {n}")
    print("\n  implausible, ranked:")
    for n, s, l in sorted(I, key=lambda x: -x[1]):
        print(f"     {s:5.1f}  {l:<12} {n}")

    ok = a >= 0.95 and not fails and not fps
    print(f"\n  {'PASS' if ok else 'FAIL'} — "
          f"{'engine separates the reference sets' if ok else 'calibration regressed'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
