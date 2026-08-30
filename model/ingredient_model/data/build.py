"""Building co-occurrence graphs from a recipe corpus.

Kept in the core library rather than in a script because three different things
need it: constructing the training graph, constructing leak-free evaluation
splits, and constructing per-cuisine graphs for stratified analysis. A second
implementation of the same NPMI would eventually disagree with the first.

Weights are normalised pointwise mutual information:

    pmi(i,j)  = log( p(i,j) / (p(i) p(j)) )
    npmi(i,j) = pmi(i,j) / -log p(i,j)

Probabilities are over recipes, not ingredient slots, and each ingredient counts
once per recipe. NPMI rather than raw counts because co-occurrence is dominated
by near-universal ingredients: salt appears with everything, so counts rank
"salt + onion" above every genuinely informative pair.
"""
from __future__ import annotations

import numpy as np

# The operating point the shipped graph settled on. Changing either value makes
# a rebuilt graph incomparable with existing results, so they are named
# constants rather than casual defaults.
MIN_COUNT = 2
THRESHOLD = -0.15


def cooccurrence(corpus, progress: bool = False):
    """Upper-triangular pair counts plus per-ingredient recipe counts.

    Recipes are bucketed by ingredient count so each bucket is a dense ``(m, k)``
    block and every pair in it comes from one ``triu_indices`` gather. Pairs are
    accumulated as a flattened code through ``bincount``, which is far faster
    than scattering into a 2-D array.
    """
    n = corpus.n_vocab
    lens = corpus.sizes
    acc = np.zeros(n * n, np.int64)
    uni = np.bincount(corpus.flat.astype(np.int64), minlength=n).astype(np.int64)

    for k in range(2, int(lens.max()) + 1):
        rows = np.nonzero(lens == k)[0]
        if not len(rows):
            continue
        iu, ju = np.triu_indices(k, 1)
        step = max(int(20_000_000 / max(len(iu), 1)), 1)
        for s in range(0, len(rows), step):
            r = rows[s:s + step]
            ids = corpus.flat[corpus.offsets[r][:, None] + np.arange(k)].astype(np.int64)
            acc += np.bincount(ids[:, iu].ravel() * n + ids[:, ju].ravel(),
                               minlength=n * n)
        if progress:
            print(f"    k={k:<3} recipes {len(rows):>9,}", flush=True)

    nz = np.nonzero(acc)[0]
    return nz // n, nz % n, acc[nz], uni


def npmi_graph(corpus, min_count: int = MIN_COUNT, threshold: float = THRESHOLD,
               progress: bool = False) -> dict:
    """Edges of the NPMI graph at a fixed operating point.

    Both knobs are held fixed rather than re-tuned per corpus. Re-tuning would
    make the edge count a free parameter, and two graphs built at different
    operating points cannot be compared.
    """
    src, dst, cnt, uni = cooccurrence(corpus, progress=progress)
    cnt = cnt.astype(np.float64)
    keep = cnt >= min_count
    src, dst, cnt = src[keep], dst[keep], cnt[keep]

    n_recipes = corpus.n_recipes
    pij = cnt / n_recipes
    pmi = np.log(pij / ((uni[src] / n_recipes) * (uni[dst] / n_recipes)))
    npmi = pmi / -np.log(pij)

    sel = npmi >= threshold
    return {
        "src": src[sel].astype(np.int32), "dst": dst[sel].astype(np.int32),
        "npmi": npmi[sel].astype(np.float32), "count": cnt[sel].astype(np.int32),
        "uni": uni, "n_recipes": np.int64(n_recipes),
        "threshold": np.float64(threshold), "min_count": np.int64(min_count),
        "itos": np.array(corpus.itos),
    }


def edge_keys(src: np.ndarray, dst: np.ndarray, n: int) -> np.ndarray:
    lo = np.minimum(src, dst).astype(np.int64)
    hi = np.maximum(src, dst).astype(np.int64)
    return lo * n + hi
