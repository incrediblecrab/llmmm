#!/usr/bin/env python3
"""Rebuild a `cooc`-style embedding from raw recipes and compare to Epicure's.

This is the real test of the paper's *reasoning*, as opposed to its artifacts.
Everything else in this repo checks whether we can reproduce Epicure's numbers
from Epicure's files. This asks a harder question: run the same idea over our
own 2.1M-recipe corpus and see whether the published `cooc` embedding is what
falls out.

Method is the standard count-based recipe: PPMI-weighted co-occurrence matrix,
truncated SVD to 300 dimensions, eigenvalue weighting, L2-normalise. That is
what "co-occurrence embedding" meant before everything became a transformer,
and the paper describes nothing more exotic.

Comparisons, in increasing order of how much they'd tell us:
  1. neighbour overlap @10   -- do we recover the same local structure?
  2. pairwise-cosine Spearman -- do we recover the same global geometry?
  3. held-out pairing AUC     -- and is theirs actually better than ours?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

DIM = 300


def build_ppmi_svd(dim=DIM, alpha=0.75, eig=0.5):
    """PPMI + truncated SVD over our own recipe co-occurrence counts.

    `alpha` is context distribution smoothing (Levy & Goldberg 2015): raising
    unigram counts to 0.75 damps the pull of ubiquitous ingredients, which is
    the same butter-is-everywhere problem that forced FLOOR_COUNT in pairing.py.
    `eig=0.5` is the symmetric eigenvalue weighting that makes SVD embeddings
    behave like word2vec rather than like raw spectral coordinates.
    """
    z = np.load(ROOT / "data" / "derived" / "recipe_cooc.npz", allow_pickle=True)
    pairs, cnt, uni = z["pairs"], z["count"].astype(np.float64), z["uni"]
    itos = [str(s) for s in z["itos"]]
    n = len(itos)
    tot = float(z["n_recipes"])

    ctx = uni.astype(np.float64) ** alpha
    ctx_tot = ctx.sum()
    pi = uni[pairs[:, 0]] / tot
    pj_s = ctx[pairs[:, 1]] / ctx_tot
    pij = cnt / tot
    ppmi = np.maximum(np.log(pij / (pi * pj_s)), 0.0)

    keep = ppmi > 0
    r, c, v = pairs[keep, 0], pairs[keep, 1], ppmi[keep]
    M = coo_matrix((np.concatenate([v, v]),
                    (np.concatenate([r, c]), np.concatenate([c, r]))),
                   shape=(n, n)).tocsr()
    print(f"  PPMI matrix: {M.nnz:,} non-zeros, density "
          f"{100*M.nnz/(n*n):.2f}%")

    U, S, _ = svds(M, k=dim)
    order = np.argsort(-S)
    E = U[:, order] * (S[order] ** eig)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    return E, itos


def neighbours(E, i, k=10):
    s = E @ E[i]
    s[i] = -np.inf
    return set(np.argsort(-s)[:k])


def main() -> None:
    from pairing import Pairing

    print("Rebuilding a cooc embedding from 2.1M recipes (PPMI + SVD)...")
    E, itos = build_ppmi_svd()

    P = Pairing(recipes=False)
    ref = P.S["cooc"]
    # align our row order to theirs; both index the same 1,790 names
    idx = np.array([ref.vocab[n] for n in itos])
    R = ref.E[idx]

    print(f"\n  ours {E.shape}   theirs {R.shape}")

    print("\n" + "=" * 66)
    print("  1. NEIGHBOUR OVERLAP @10")
    print("=" * 66)
    ov = [len(neighbours(E, i) & neighbours(R, i)) for i in range(len(itos))]
    ov = np.array(ov)
    print(f"    mean overlap      {ov.mean():.2f} / 10")
    print(f"    median            {np.median(ov):.0f} / 10")
    print(f"    zero overlap      {int((ov == 0).sum()):,} ingredients "
          f"({100*(ov==0).mean():.1f}%)")
    print(f"    >=5 overlap       {int((ov >= 5).sum()):,} ingredients "
          f"({100*(ov>=5).mean():.1f}%)")

    print("\n    where we agree most:")
    for i in np.argsort(-ov)[:6]:
        print(f"      {itos[i]:22s} {ov[i]}/10")
    print("    where we agree least:")
    for i in np.argsort(ov)[:6]:
        mine = ", ".join(itos[j] for j in list(neighbours(E, i))[:3])
        print(f"      {itos[i]:22s} {ov[i]}/10   ours: {mine}")

    print("\n" + "=" * 66)
    print("  2. GLOBAL GEOMETRY (pairwise cosine, 40k random pairs)")
    print("=" * 66)
    rng = np.random.default_rng(0)
    a = rng.integers(0, len(itos), 40_000)
    b = rng.integers(0, len(itos), 40_000)
    m = a != b
    a, b = a[m], b[m]
    co_ours = np.einsum("ij,ij->i", E[a], E[b])
    co_theirs = np.einsum("ij,ij->i", R[a], R[b])
    rho = spearmanr(co_ours, co_theirs).statistic
    print(f"    spearman(ours, theirs) = {rho:+.3f}")
    print(f"    pearson                = "
          f"{np.corrcoef(co_ours, co_theirs)[0,1]:+.3f}")

    print("\n" + "=" * 66)
    print("  3. HELD-OUT PAIRING AUC -- is their embedding better than ours?")
    print("=" * 66)
    from sklearn.metrics import roc_auc_score
    from validate_holdout import HELD_BAD, HELD_GOOD

    def score(M_, x, y):
        i, j = P.resolve(x), P.resolve(y)
        if i is None or j is None:
            return None
        pos = {n: k for k, n in enumerate(itos)}
        return float(M_[pos[i]] @ M_[pos[j]])

    ys, so, st = [], [], []
    for lbl, pairs_ in ((1, HELD_GOOD), (0, HELD_BAD)):
        for x, y in pairs_:
            u, v = score(E, x, y), score(R, x, y)
            if u is None or v is None:
                continue
            ys.append(lbl)
            so.append(u)
            st.append(v)
    print(f"    n = {sum(ys)} good / {len(ys)-sum(ys)} bad")
    print(f"    OUR embedding (from raw recipes)   AUC {roc_auc_score(ys, so):.3f}")
    print(f"    THEIR published cooc embedding     AUC {roc_auc_score(ys, st):.3f}")

    np.savez_compressed(ROOT / "data" / "derived" / "replicated_cooc.npz",
                        E=E.astype(np.float32), itos=np.array(itos, dtype=object))
    print("\n  -> data/derived/replicated_cooc.npz")


if __name__ == "__main__":
    main()
