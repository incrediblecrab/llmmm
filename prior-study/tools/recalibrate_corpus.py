#!/usr/bin/env python3
"""Recalibrate and re-validate the pairing engine against real recipes.

The tier boundaries in tools/pairing.py were fitted to 30 pairs I chose by
hand, and tools/validate_holdout.py showed the cost: in-sample AUC 0.995,
held-out 0.873. Hand-picked sets are small and encode the picker's assumptions
-- `duck+orange` is in the "canonical" list but appears in ZERO of 2.1M recipes.

This replaces intuition with the corpus:

  positives  pairs that genuinely co-occur far above chance
  negatives  pairs of common ingredients that essentially never co-occur

IMPORTANT INTERPRETATION LIMIT. This measures whether the engine predicts
RECIPE CO-OCCURRENCE, which is what `cooc` was trained on -- so some agreement
is expected by construction. It is still informative, because the embedding is
a 300-d lossy compression trained on a different corpus mix (RecipeNLG is 54%
of Epicure's corpus; this is RecipeNLG alone), so it measures how much of the
real signal survived. It does NOT measure "is this a good pairing" in the human
sense -- see the plate-vs-dish caveat in docs/FINDINGS.md.
"""
from __future__ import annotations

import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "tools")
from build_recipe_cooc import OUT  # noqa: E402
from pairing import Pairing  # noqa: E402

MIN_UNI = 300      # both ingredients must be common enough to be informative
MIN_CNT = 40       # a positive must co-occur this often
N_EACH = 1500


def main():
    z = np.load(OUT / "recipe_cooc.npz", allow_pickle=True)
    pairs, cnt, npmi, uni = z["pairs"], z["count"], z["npmi"], z["uni"]
    itos = list(z["itos"])
    n_rec = int(z["n_recipes"])
    P = Pairing()
    sib = P.S["cooc"]
    rng = np.random.default_rng(0)

    common = np.where(uni >= MIN_UNI)[0]
    seen = {(int(a), int(b)) for a, b in pairs}
    print("=" * 72)
    print("CORPUS-DERIVED CALIBRATION")
    print("=" * 72)
    print(f"\n  recipes {n_rec:,}   ingredients with >={MIN_UNI} appearances "
          f"{len(common):,}")

    # positives: strong, well-supported co-occurrence
    ok = (cnt >= MIN_CNT) & (uni[pairs[:, 0]] >= MIN_UNI) & (uni[pairs[:, 1]] >= MIN_UNI)
    cand = np.where(ok)[0]
    cand = cand[np.argsort(npmi[cand])[::-1][:N_EACH * 3]]
    pos = rng.choice(cand, min(N_EACH, len(cand)), replace=False)
    POS = [(int(pairs[k, 0]), int(pairs[k, 1])) for k in pos]

    # negatives: both common, yet never co-occur
    NEG = []
    while len(NEG) < N_EACH:
        i, j = rng.choice(common, 2, replace=False)
        a, b = (int(i), int(j)) if i < j else (int(j), int(i))
        if (a, b) not in seen:
            NEG.append((a, b))

    print(f"  positives {len(POS):,} (count>={MIN_CNT}, top nPMI)   "
          f"negatives {len(NEG):,} (never co-occur, both common)")
    print(f"  e.g. + {itos[POS[0][0]]}+{itos[POS[0][1]]}   "
          f"- {itos[NEG[0][0]]}+{itos[NEG[0][1]]}")

    def score(lst):
        out = []
        for a, b in lst:
            v = P.pair(itos[a], itos[b])
            if v is not None and not isinstance(v, str):
                out.append(v.overall)
        return np.array(out)

    print("\n  scoring (this takes a minute)...", flush=True)
    sp, sn = score(POS), score(NEG)
    y = [1] * len(sp) + [0] * len(sn)
    auc = roc_auc_score(y, np.concatenate([sp, sn]))

    print(f"\n  CORPUS AUC = {auc:.3f}   (hand-picked in-sample 0.995, "
          f"hand-picked held-out 0.873)")
    print(f"    positives  median {np.median(sp):5.1f}   "
          f"pct below 65 (called clash): {100 * (sp < 65).mean():.1f}%")
    print(f"    negatives  median {np.median(sn):5.1f}   "
          f"pct above 80 (sold as good): {100 * (sn >= 80).mean():.1f}%")

    # where should the boundary actually sit?
    grid = np.arange(20, 96, 1.0)
    j = [(np.mean(sp >= t) - np.mean(sn >= t), t) for t in grid]
    best = max(j)
    print(f"\n  best single threshold (Youden J) = {best[1]:.0f}  "
          f"(J={best[0]:.3f});  engine currently uses 65 for clash")
    for t in (50, 60, 65, 70, 80):
        print(f"    t={t:2d}   recall {100 * np.mean(sp >= t):5.1f}%   "
              f"false-positive {100 * np.mean(sn >= t):5.1f}%")

    # per-sibling: which evidence type actually predicts co-occurrence?
    print("\n  which sibling predicts real co-occurrence best?")
    for s in ("cooc", "core", "chem"):
        sb = P.S[s]
        cp = np.array([float(sb.E[a] @ sb.E[b]) for a, b in POS])
        cn = np.array([float(sb.E[a] @ sb.E[b]) for a, b in NEG])
        print(f"    {s:5s} AUC {roc_auc_score([1] * len(cp) + [0] * len(cn), np.concatenate([cp, cn])):.3f}")
    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
