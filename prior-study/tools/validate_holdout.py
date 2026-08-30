#!/usr/bin/env python3
"""Held-out validation of the pairing engine.

tools/calibrate_pairing.py reports AUC 0.995 -- but it measures on the SAME 30
pairs that were used to choose the tier boundaries. That is training accuracy.
This script scores a disjoint set that was never used for fitting, which is the
only number that says anything about real users.
"""
from __future__ import annotations

import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "tools")
from calibrate_pairing import CANONICAL, IMPLAUSIBLE  # noqa: E402
from pairing import Pairing  # noqa: E402

HELD_GOOD = [
    ("pork", "apple"), ("beef", "mushroom"), ("duck", "cherry"),
    ("fig", "prosciutto"), ("pear", "blue_cheese"), ("watermelon", "feta_cheese"),
    ("beet", "goat_cheese"), ("pumpkin", "sage"), ("scallop", "bacon"),
    ("chicken", "lemon"), ("salmon", "lemon"), ("shrimp", "lime"),
    ("pineapple", "ham"), ("basil", "pine_nut"), ("cucumber", "mint"),
    ("corn", "butter"), ("lamb", "rosemary"), ("chocolate", "chili_pepper"),
    ("honey", "goat_cheese"), ("sage", "brown_butter"), ("miso", "butter"),
    ("tuna", "soy_sauce"), ("mango", "chili_pepper"), ("beef", "horseradish"),
    ("cabbage", "caraway_seed"), ("eggplant", "miso"), ("clam", "chorizo"),
    ("apricot", "almond"), ("rhubarb", "ginger"), ("leek", "potato"),
]

HELD_BAD = [
    ("pickle", "ice_cream"), ("sardine", "chocolate"), ("garlic", "ice_cream"),
    ("oyster", "cinnamon"), ("honey", "tuna"), ("soy_sauce", "strawberry"),
    ("vanilla", "onion"), ("banana", "oyster"), ("liver", "marshmallow"),
    ("anchovy", "vanilla"), ("kimchi", "custard"), ("blueberry", "fish_sauce"),
    ("horseradish", "peach"), ("mustard", "melon"), ("cheese", "gummy_candy"),
]


def score_set(P, pairs, tag):
    out, missing = [], []
    for a, b in pairs:
        v = P.pair(a, b)
        if v is None or isinstance(v, str):
            missing.append(f"{a}+{b}")
            continue
        out.append((f"{a}+{b}", v.overall, v.label))
    if missing:
        print(f"  [{tag}] not in vocab, skipped {len(missing)}: {', '.join(missing)}")
    return out


def report(name, good, bad):
    gs = [s for _, s, _ in good]
    bs = [s for _, s, _ in bad]
    auc = roc_auc_score([1] * len(gs) + [0] * len(bs), gs + bs)
    print(f"\n  {name}: n={len(gs)}+{len(bs)}  AUC = {auc:.3f}")
    print(f"    good  min {min(gs):5.1f}  median {np.median(gs):5.1f}  max {max(gs):5.1f}")
    print(f"    bad   min {min(bs):5.1f}  median {np.median(bs):5.1f}  max {max(bs):5.1f}")
    miss = [(n, s, l) for n, s, l in good if l == "clash"]
    fp = [(n, s, l) for n, s, l in bad if s >= 80]
    print(f"    good pairs called 'clash'      : {len(miss)}/{len(gs)}"
          f"  ({100 * len(miss) / len(gs):.0f}%)")
    for n, s, l in sorted(miss, key=lambda x: x[1])[:10]:
        print(f"        {n:34s} {s:5.1f}")
    print(f"    bad pairs sold as plausible+   : {len(fp)}/{len(bs)}")
    for n, s, l in fp:
        print(f"        {n:34s} {s:5.1f}  {l}")
    return auc


def main():
    P = Pairing()
    print("=" * 70)
    print("HELD-OUT VALIDATION  (pairs never used to fit the thresholds)")
    print("=" * 70)

    print("\n[fit set — the 0.995 number]")
    fg = score_set(P, CANONICAL, "fit-good")
    fb = score_set(P, IMPLAUSIBLE, "fit-bad")
    a_fit = report("FIT (in-sample)", fg, fb)

    print("\n[held-out set — the honest number]")
    hg = score_set(P, HELD_GOOD, "held-good")
    hb = score_set(P, HELD_BAD, "held-bad")
    a_held = report("HELD-OUT", hg, hb)

    print("\n" + "=" * 70)
    print(f"  in-sample AUC {a_fit:.3f}   held-out AUC {a_held:.3f}   "
          f"generalisation gap {a_fit - a_held:+.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
