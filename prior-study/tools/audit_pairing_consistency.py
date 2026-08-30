#!/usr/bin/env python3
"""Consistency audit of the pairing model.

The engine's thresholds were calibrated on ONE random sample. This asks whether
the formula is actually stable, or whether the numbers are an artefact of that
sample. Checks:

  1. determinism   -- same seed, same answer
  2. symmetry      -- pair(x,y) == pair(y,x)
  3. seed drift    -- how far do scores move when the RNG sample changes?
  4. label churn   -- do any verdicts actually flip across seeds?
  5. n sensitivity -- does the null sample size change the answer?
  6. monotonicity  -- is the combined score monotone in its inputs?
  7. saturation    -- do the CDF tails clip?
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "tools")
from pairing import SIBLINGS, Pairing, _combine  # noqa: E402

PAIRS = [
    ("tomato", "basil"), ("strawberry", "basil"), ("pork", "apple"),
    ("chocolate", "chili_pepper"), ("lamb", "mint"), ("beef", "mushroom"),
    ("salmon", "dill"), ("miso", "caramel"), ("oyster", "maple_syrup"),
    ("white_chocolate", "caviar"), ("banana", "anchovy"), ("garlic", "olive_oil"),
    ("marshmallow", "anchovy"), ("cinnamon", "tuna"), ("coffee", "cardamom"),
]


def score(P, pairs):
    out = {}
    for a, b in pairs:
        v = P.pair(a, b)
        if v is not None and not isinstance(v, str):
            out[(a, b)] = (v.overall, v.label)
    return out


def main():
    print("=" * 66)
    print("PAIRING MODEL — CONSISTENCY AUDIT")
    print("=" * 66)

    P0 = Pairing(seed=0)
    base = score(P0, PAIRS)

    # 1. determinism -------------------------------------------------------
    again = score(Pairing(seed=0), PAIRS)
    det = all(abs(base[k][0] - again[k][0]) < 1e-9 and base[k][1] == again[k][1]
              for k in base)
    print(f"\n1. DETERMINISM (seed=0 twice)            {'PASS' if det else 'FAIL'}")

    # 2. symmetry ----------------------------------------------------------
    asym = []
    for a, b in PAIRS:
        f, r = P0.pair(a, b), P0.pair(b, a)
        if f is None or isinstance(f, str) or r is None or isinstance(r, str):
            continue
        if abs(f.overall - r.overall) > 1e-9 or f.label != r.label:
            asym.append((a, b, f.overall, r.overall, f.label, r.label))
    print(f"2. SYMMETRY  pair(x,y) == pair(y,x)      "
          f"{'PASS' if not asym else 'FAIL'}  ({len(asym)} asymmetric)")
    for row in asym[:5]:
        print(f"     {row[0]}+{row[1]}: {row[2]:.2f}/{row[4]} vs {row[3]:.2f}/{row[5]}")

    # 3-4. seed drift and label churn --------------------------------------
    runs = {}
    for s in (1, 2, 3):
        runs[s] = score(Pairing(seed=s), PAIRS)
    keys = [k for k in base if all(k in r for r in runs.values())]
    drift = {k: [base[k][0]] + [runs[s][k][0] for s in runs] for k in keys}
    spreads = {k: max(v) - min(v) for k, v in drift.items()}
    churn = {k: sorted({base[k][1]} | {runs[s][k][1] for s in runs})
             for k in keys if len({base[k][1]} | {runs[s][k][1] for s in runs}) > 1}
    arr = np.array(list(spreads.values()))
    print(f"\n3. SEED DRIFT over 4 seeds (score points)")
    print(f"     mean {arr.mean():.3f}   median {np.median(arr):.3f}   "
          f"max {arr.max():.3f}")
    worst = sorted(spreads.items(), key=lambda x: -x[1])[:3]
    for k, v in worst:
        print(f"     {k[0]}+{k[1]:<16s} spread {v:.2f}  "
              f"({', '.join(f'{x:.1f}' for x in drift[k])})")
    print(f"\n4. LABEL CHURN across seeds              "
          f"{'PASS' if not churn else str(len(churn)) + ' FLIPPED'}")
    for k, v in churn.items():
        print(f"     {k[0]}+{k[1]}: {' / '.join(v)}")

    # 5. null sample size --------------------------------------------------
    big = score(Pairing(seed=0, n_null=200_000), PAIRS)
    d = np.array([abs(base[k][0] - big[k][0]) for k in base if k in big])
    flips = [k for k in base if k in big and base[k][1] != big[k][1]]
    print(f"\n5. n_null 30k -> 200k                    "
          f"mean |delta| {d.mean():.3f}, max {d.max():.3f}, "
          f"{len(flips)} label flips")
    for k in flips:
        print(f"     {k[0]}+{k[1]}: {base[k][1]} -> {big[k][1]}")

    # 6. monotonicity of the combination rule ------------------------------
    rng = np.random.default_rng(7)
    bad = 0
    for _ in range(20_000):
        p = rng.uniform(0, 100, 3)
        k = rng.integers(0, 3)
        q = p.copy()
        q[k] = min(100.0, p[k] + rng.uniform(0.1, 20))
        if _combine(list(q)) < _combine(list(p)) - 1e-9:
            bad += 1
    print(f"\n6. MONOTONICITY of 0.5hi+0.3mid+0.2lo    "
          f"{'PASS' if bad == 0 else 'FAIL'}  ({bad}/20000 violations)")
    w = [0.5, 0.3, 0.2]
    print(f"     weights sum to {sum(w)}; score of an all-100 pair = "
          f"{_combine([100, 100, 100]):.1f}, all-0 = {_combine([0, 0, 0]):.1f}")

    # 7. CDF tail saturation -----------------------------------------------
    print(f"\n7. CDF SATURATION (fraction of vocab pairs clipping to p100)")
    for s in SIBLINGS:
        sib = P0.S[s]
        top = sib.quantiles[-1]
        r2 = np.random.default_rng(11)
        i, j = r2.integers(0, len(sib.E), 200_000), r2.integers(0, len(sib.E), 200_000)
        m = i != j
        cos = np.sum(sib.E[i[m]] * sib.E[j[m]], axis=1)
        print(f"     {s:5s} max quantile {top:+.4f}   "
              f"clipped {100 * (cos >= top).mean():.4f}%")

    print("\n" + "=" * 66)


if __name__ == "__main__":
    main()
