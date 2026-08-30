#!/usr/bin/env python3
"""Tune how real-recipe evidence is blended into the pairing score.

Two candidate statistics, and they disagree in an important way:

  count  how many of 2.1M recipes use both. Practical: "people do this."
  nPMI   association strength above chance. Distinctive: "this is a signature
         combination", but it PENALISES pairs of individually common
         ingredients -- chicken+lemon appears in 23,940 recipes yet scores
         nPMI +0.011, because chicken and lemon are both everywhere.

A user asking "can I pair chicken and lemon?" wants the count answer, not the
nPMI answer. But nPMI is what stops "salt + everything" scoring perfectly. So
this sweeps the blend and the overall weight against two disjoint yardsticks:
the hand-curated held-out set, and corpus-derived pairs.
"""
from __future__ import annotations

import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "tools")
from pairing import Pairing, _combine, SIBLINGS  # noqa: E402
from validate_holdout import HELD_BAD, HELD_GOOD  # noqa: E402


def main():
    P = Pairing()
    rec = P.rec
    n = rec["n"]
    lc = np.log1p(rec["cnt"].astype(np.float64))
    q_cnt = np.quantile(lc, np.linspace(0, 1, 1001))
    good = np.isfinite(rec["npmi"])
    q_pmi = np.quantile(rec["npmi"][good], np.linspace(0, 1, 1001))

    def parts(a, b):
        """(embedding percentile-vs-null, count pct, nPMI pct) for one pair."""
        x, y = P.resolve(a), P.resolve(b)
        if not x or not y:
            return None
        i, j = P.S["cooc"].vocab[x], P.S["cooc"].vocab[y]
        if i == j:
            return None
        pcts = []
        for s in SIBLINGS:
            sb = P.S[s]
            pcts.append(sb.pct(float(sb.E[i] @ sb.E[j])))
        emb = float(np.searchsorted(P.null, _combine(pcts)) / len(P.null) * 100)
        lo, hi = (i, j) if i < j else (j, i)
        k = np.searchsorted(rec["keys"], lo * n + hi)
        if k >= len(rec["keys"]) or rec["keys"][k] != lo * n + hi:
            return emb, None, None
        c = float(np.searchsorted(q_cnt, np.log1p(rec["cnt"][k])) / 1001 * 100)
        v = rec["npmi"][k]
        pm = (float(np.searchsorted(q_pmi, v) / 1001 * 100)
              if np.isfinite(v) else None)
        return emb, c, pm

    G = [p for p in (parts(*x) for x in HELD_GOOD) if p]
    B = [p for p in (parts(*x) for x in HELD_BAD) if p]
    y = [1] * len(G) + [0] * len(B)
    print(f"held-out: {len(G)} good, {len(B)} bad "
          f"({sum(1 for p in G + B if p[1] is None)} unseen in corpus)")

    def blend(p, w_rec, w_pmi):
        emb, c, pm = p
        if c is None:
            return emb
        r = w_pmi * (pm if pm is not None else 50.0) + (1 - w_pmi) * c
        return w_rec * r + (1 - w_rec) * emb

    print("\n  W_RECIPE x w_nPMI  ->  held-out AUC")
    best = (0, None)
    for w_rec in (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 1.0):
        row = []
        for w_pmi in (0.0, 0.25, 0.5, 0.75, 1.0):
            s = [blend(p, w_rec, w_pmi) for p in G + B]
            a = roc_auc_score(y, s)
            row.append(f"{a:.3f}")
            if a > best[0]:
                best = (a, (w_rec, w_pmi))
        print(f"    W={w_rec:<4}  " + "  ".join(row))
    print("    w_nPMI:        0.00   0.25   0.50   0.75   1.00")
    print(f"\n  best held-out AUC {best[0]:.3f} at W_RECIPE={best[1][0]}, "
          f"w_nPMI={best[1][1]}")

    # sanity: the pairs that motivated this
    print("\n  sanity check at the chosen blend:")
    for a, b in [("chicken", "lemon"), ("corn", "butter"), ("pork", "apple"),
                 ("tomato", "basil"), ("salt", "sugar"), ("garlic", "ice_cream"),
                 ("vanilla", "onion"), ("banana", "oyster")]:
        p = parts(a, b)
        if p:
            print(f"    {a}+{b:14s} emb {p[0]:5.1f}  cnt "
                  f"{p[1] if p[1] is None else round(p[1], 1)!s:>5}  pmi "
                  f"{p[2] if p[2] is None else round(p[2], 1)!s:>5}  -> "
                  f"{blend(p, *best[1]):5.1f}")


if __name__ == "__main__":
    main()
