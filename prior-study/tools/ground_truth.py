#!/usr/bin/env python3
"""Ground-truth validation: real recipes vs the Epicure `cooc` embedding.

Until now every check in this project compared our code against Epicure's own
derived artefacts. 100% parity with their live service proves we match their
ARTEFACT -- it says nothing about whether the artefact matches reality, because
we never looked at a recipe.

This asks the questions that require real data:

  1. Does cooc cosine actually track recipe co-occurrence (PMI)?
  2. When the pairing engine calls a real pairing a "clash", is that a
     compression artefact or are those ingredients genuinely never combined?
  3. Would raw recipe statistics beat the 300-d embedding for pairing?
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, "tools")
from pairing import Pairing  # noqa: E402
from validate_holdout import HELD_BAD, HELD_GOOD  # noqa: E402
from build_recipe_cooc import OUT  # noqa: E402


def load():
    z = np.load(OUT / "recipe_cooc.npz", allow_pickle=True)
    pairs, cnt = z["pairs"], z["count"]
    key = {(int(a), int(b)): k for k, (a, b) in enumerate(pairs)}
    return z, key


def main():
    z, key = load()
    pairs, cnt, pmi, npmi = z["pairs"], z["count"], z["pmi"], z["npmi"]
    uni, n_rec = z["uni"], int(z["n_recipes"])
    P = Pairing()
    sib = P.S["cooc"]

    print("=" * 72)
    print("GROUND TRUTH — real recipes vs the cooc embedding")
    print("=" * 72)
    print(f"\n  recipes {n_rec:,}   distinct co-occurring pairs {len(pairs):,}"
          f"   vocab seen {int((uni > 0).sum()):,}/{len(uni)}")

    # ---- 1. does the embedding track real co-occurrence? -----------------
    E = sib.E
    cos = np.sum(E[pairs[:, 0]] * E[pairs[:, 1]], axis=1)
    m = cnt >= 5                       # ignore once-off noise
    r_all = spearmanr(cos, npmi).statistic
    r_5 = spearmanr(cos[m], npmi[m]).statistic
    r_cnt = spearmanr(cos[m], np.log(cnt[m])).statistic
    print("\n1. DOES cooc COSINE TRACK REAL CO-OCCURRENCE?")
    print(f"   spearman(cos, nPMI)        all pairs   rho = {r_all:+.3f}")
    print(f"   spearman(cos, nPMI)        count>=5    rho = {r_5:+.3f}  "
          f"(n={int(m.sum()):,})")
    print(f"   spearman(cos, log count)   count>=5    rho = {r_cnt:+.3f}")

    # how much of the top-PMI signal survives compression?
    for k in (100, 1000):
        top = set(map(tuple, pairs[m][np.argsort(npmi[m])[-k:]]))
        topc = set(map(tuple, pairs[m][np.argsort(cos[m])[-k:]]))
        print(f"   overlap of top-{k:<5d} by nPMI vs by cosine: "
              f"{100 * len(top & topc) / k:.1f}%")

    # ---- 2. diagnose the held-out misses ---------------------------------
    print("\n2. THE HELD-OUT MISSES — compression artefact, or genuinely rare?")
    print(f"   {'pair':32s} {'engine':>7} {'recipes':>9} {'nPMI':>7}  verdict")
    rows = []
    for a, b in HELD_GOOD:
        x, y = P.resolve(a), P.resolve(b)
        if not x or not y:
            continue
        v = P.pair(a, b)
        if v is None or isinstance(v, str):
            continue
        i, j = sorted((sib.vocab[x], sib.vocab[y]))
        k = key.get((i, j))
        c = int(cnt[k]) if k is not None else 0
        p = float(npmi[k]) if k is not None else float("nan")
        rows.append((f"{a}+{b}", v.overall, v.label, c, p))
    rows.sort(key=lambda r: r[1])
    for name, ov, lab, c, p in rows:
        if lab != "clash":
            continue
        if c >= 500:
            verdict = "COMPRESSION LOSS — common in real recipes"
        elif c >= 50:
            verdict = "partial loss — moderately common"
        else:
            verdict = "genuinely rare in corpus"
        print(f"   {name:32s} {ov:7.1f} {c:9,} {p:+7.3f}  {verdict}")

    # ---- 3. would raw recipe stats beat the embedding? -------------------
    print("\n3. RAW RECIPE STATS AS A PAIRING SIGNAL (vs the embedding)")
    from sklearn.metrics import roc_auc_score

    def feats(pairs_list):
        out = []
        for a, b in pairs_list:
            x, y = P.resolve(a), P.resolve(b)
            if not x or not y:
                continue
            v = P.pair(a, b)
            if v is None or isinstance(v, str):
                continue
            i, j = sorted((sib.vocab[x], sib.vocab[y]))
            k = key.get((i, j))
            out.append((v.overall,
                        float(npmi[k]) if k is not None else -1.0,
                        np.log1p(int(cnt[k]) if k is not None else 0)))
        return out

    g, b = feats(HELD_GOOD), feats(HELD_BAD)
    y = [1] * len(g) + [0] * len(b)
    for idx, nm in [(0, "engine score (embedding)"), (1, "raw nPMI"),
                    (2, "log recipe count")]:
        s = [r[idx] for r in g] + [r[idx] for r in b]
        print(f"   held-out AUC — {nm:26s} {roc_auc_score(y, s):.3f}")
    comb = ([0.5 * r[0] / 100 + 0.5 * min(r[2] / 10, 1) for r in g]
            + [0.5 * r[0] / 100 + 0.5 * min(r[2] / 10, 1) for r in b])
    print(f"   held-out AUC — {'embedding + log count':26s} "
          f"{roc_auc_score(y, comb):.3f}")
    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
