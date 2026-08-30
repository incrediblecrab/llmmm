"""Metrics M1-M5.

Each is a pure function of an embedding and a fixed evaluation context, so the
same code scores a random-walk model, a factorisation and a transformer on
identical terms. Chance level is stated for every metric that has one; a number
without its chance level is not interpretable.

Sampling uses one fixed seed throughout, so two models see the *same* triplets
and the same negatives. Re-drawing per model would add sampling noise on top of
the effect being measured.
"""
from __future__ import annotations

import numpy as np

from ..config import SEED

N_TRIPLETS = 20_000


def unit(W: np.ndarray) -> np.ndarray:
    return W / np.clip(np.linalg.norm(W, axis=1, keepdims=True), 1e-12, None)


def all_but_top(W: np.ndarray, k: int = 3) -> np.ndarray:
    """Mu & Viswanath's all-but-the-top: centre, then project out the leading
    ``k`` principal directions.

    Diagnostic for whether popularity is low-rank. If it is, removing these
    directions should cut M5 sharply while leaving M2/M4 intact; if M2 collapses
    too, the popularity signal is load-bearing and the cure is worse than the
    disease.
    """
    X = W - W.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return X - (X @ Vt[:k].T) @ Vt[:k]


def _prop_ci(p: float, n: int) -> tuple[float, float]:
    """Point estimate with a 95% half-width, so two models can be told apart.
    Differences smaller than the interval are reported as indistinguishable."""
    return float(p), float(1.96 * np.sqrt(max(p * (1 - p), 1e-12) / max(n, 1)))


def m1_participation_ratio(W: np.ndarray) -> float:
    """Effective dimensionality: ``(Σλ)² / Σλ²`` over covariance eigenvalues.

    Collapse shows up as PR near 1. Not to be maximised — strict isotropy
    conflicts with clustering, and a healthy embedding sits around 100-220 out
    of 300, not at the ceiling.
    """
    X = W - W.mean(0)
    lam = np.linalg.svd(X, compute_uv=False) ** 2
    return float(lam.sum() ** 2 / np.maximum((lam ** 2).sum(), 1e-30))


def m2_triplet_accuracy(U: np.ndarray, ctx, tier: str) -> tuple[float, float]:
    """``cos(a, known substitute) > cos(a, random ingredient)``. Chance 0.50.

    Ties score 0.5 rather than 0, so a degenerate constant-similarity model
    lands exactly at chance instead of being spuriously penalised.
    """
    pairs = ctx.subs.tier(tier)
    if not pairs:
        return float("nan"), float("nan")
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(pairs), N_TRIPLETS)
    a = np.fromiter((pairs[i][0] for i in idx), np.int64, N_TRIPLETS)
    b = np.fromiter((pairs[i][1] for i in idx), np.int64, N_TRIPLETS)
    c = rng.integers(0, ctx.n, N_TRIPLETS)
    ok = c != a
    pos = np.einsum("ij,ij->i", U[a[ok]], U[b[ok]])
    neg = np.einsum("ij,ij->i", U[a[ok]], U[c[ok]])
    return _prop_ci((pos > neg).mean() + 0.5 * (pos == neg).mean(), int(ok.sum()))


def m3_recall_at_10(U: np.ndarray, ctx, tier: str) -> float:
    """Fraction of an ingredient's known substitutes inside its 10 nearest
    neighbours, over anchors with at least 3 known substitutes."""
    m = ctx.subs.anchors(tier, min_subs=3)
    if not m:
        return float("nan")
    keys = np.array(sorted(m))
    S = U[keys] @ U.T
    S[np.arange(len(keys)), keys] = -np.inf
    top = np.argpartition(-S, 10, axis=1)[:, :10]
    return float(np.mean([len(m[k] & set(top[i].tolist())) / len(m[k])
                          for i, k in enumerate(keys)]))


def m4_link_auc(U: np.ndarray, ctx) -> tuple[float, float]:
    """Rank held-out edges above degree-matched non-edges. Chance 0.50.

    Negatives are matched on degree rank so the metric cannot be won by
    memorising which ingredients are popular — that failure mode is what M5
    measures, and conflating the two would hide it.
    """
    if ctx.held is None:
        return float("nan"), float("nan")
    hu, hv = ctx.held
    pos = np.einsum("ij,ij->i", U[hu], U[hv])
    neg = np.einsum("ij,ij->i", U[hu], U[ctx.link_negatives])
    return _prop_ci((pos > neg).mean() + 0.5 * (pos == neg).mean(), len(hu))


def m5_popularity(W: np.ndarray, ctx) -> dict:
    """Do the leading directions encode frequency?

    If they do, any feature that aggregates cosine similarity silently becomes a
    popularity chart — which is a product failure, not just a geometric one.
    """
    X = W - W.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    proj = X @ Vt[:3].T
    logf = np.log1p(ctx.unigram)
    r = [float(abs(np.corrcoef(proj[:, k], logf)[0, 1])) for k in range(3)]

    U = unit(W)
    score = (U @ U.T).mean(1)
    top_model = set(np.argsort(-score)[:20].tolist())
    top_freq = set(np.argsort(-ctx.unigram)[:20].tolist())
    inter = len(top_model & top_freq)
    return {"pc_freq_corr": r, "max_pc_freq_corr": max(r),
            "top20_freq_jaccard": inter / (40 - inter)}


def sample_degree_matched_negatives(ctx, seed: int = SEED) -> np.ndarray:
    """For each held-out edge ``(u, v)``, a non-edge ``(u, v')`` whose degree
    rank is within a ±10%-of-vocab band of ``v``.

    Computed once per context and reused by every model, so all models are
    scored against exactly the same negatives.
    """
    hu, hv = ctx.held
    n, deg, edges = ctx.n, ctx.degree, ctx.edge_set
    rng = np.random.default_rng(seed)
    order = np.argsort(deg)
    rank = np.empty(n, np.int64)
    rank[order] = np.arange(n)
    band = max(n // 10, 1)

    neg = np.empty(len(hv), np.int64)
    for i, v in enumerate(hv):
        r, u, cand = rank[v], hu[i], v
        for _ in range(40):
            cand = order[int(np.clip(r + rng.integers(-band, band + 1), 0, n - 1))]
            if cand != u and (min(u, cand) * n + max(u, cand)) not in edges:
                break
        neg[i] = cand
    return neg
