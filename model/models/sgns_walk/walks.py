"""Weighted random walks over ingredient graphs.

The walkers are vectorised: every walker takes its step in one NumPy call. A
per-node Python loop is roughly a thousand times slower at this walk volume
(1,790 nodes x 100 walks x 50 steps per epoch), which is the difference between
a model that trains in minutes and one that does not finish.
"""
from __future__ import annotations

import numpy as np

from ingredient_model.data import ChemGraph, IIGraph


class CSR:
    """Weighted adjacency in compressed-sparse-row form.

    The globally cumulative weight array is what makes weighted walking
    vectorisable: each node's edges are contiguous and the cumsum is monotonic
    across the whole array, so one ``searchsorted`` samples a neighbour for
    every walker simultaneously.
    """

    def __init__(self, n: int, src, dst, w):
        src = np.asarray(src, np.int64)
        dst = np.asarray(dst, np.int64)
        w = np.asarray(w, np.float64)
        order = np.argsort(src, kind="stable")
        src, dst, w = src[order], dst[order], w[order]
        self.n = n
        self.indptr = np.zeros(n + 1, np.int64)
        np.add.at(self.indptr, src + 1, 1)
        np.cumsum(self.indptr, out=self.indptr)
        self.indices = dst
        self.cum = np.concatenate([[0.0], np.cumsum(w)])
        self.deg = np.diff(self.indptr)

    def step(self, cur: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One weighted hop for every walker. Dead ends stay put."""
        lo, hi = self.indptr[cur], self.indptr[cur + 1]
        clo, chi = self.cum[lo], self.cum[hi]
        target = clo + rng.random(len(cur)) * (chi - clo)
        j = np.searchsorted(self.cum, target, side="right") - 1
        j = np.clip(j, lo, np.maximum(hi - 1, lo))
        nxt = self.indices[j]
        dead = self.deg[cur] == 0
        nxt[dead] = cur[dead]
        return nxt


def _run(g: CSR, start: np.ndarray, wlen: int, rng) -> np.ndarray:
    out = np.empty((len(start), wlen + 1), np.int32)
    out[:, 0] = start
    cur = start.copy()
    for t in range(wlen):
        cur = g.step(cur, rng)
        out[:, t + 1] = cur
    return out


def walks_cooc(ii: IIGraph, rng, walks_per_node: int, walk_length: int
               ) -> np.ndarray:
    s, d, w = ii.symmetric()
    g = CSR(ii.n_vocab, s, d, w)
    start = np.repeat(np.arange(ii.n_vocab)[g.deg > 0], walks_per_node)
    rng.shuffle(start)
    return _run(g, start, walk_length, rng)


def walks_chem(ii: IIGraph, chem: ChemGraph, rng, walks_per_node: int,
               walk_length: int) -> np.ndarray:
    """Typed ``I -> C -> I`` metapaths, one compound category per walk.

    Pinning the category is what makes the metapath typed rather than an untyped
    bipartite stroll: a walk confined to organosulfur compounds connects
    garlic/onion/leek, one confined to lactones connects dairy and stone fruit.
    Mixing categories inside a walk averages those channels together and
    destroys exactly the structure this schema exists to capture.

    Compound nodes are offset by ``n_vocab`` so ingredients and compounds share
    one embedding table but can never collide.
    """
    n_vocab = ii.n_vocab
    n_all = n_vocab + chem.n_compounds
    n_cat = int(chem.ctype.max()) + 1
    per_cat = max(walks_per_node // n_cat, 1)
    out = []
    for c in range(n_cat):
        m = chem.ctype[chem.dst] == c
        if m.sum() < 2:
            continue
        cs, cd = chem.src[m], chem.dst[m] + n_vocab
        fwd = CSR(n_all, cs, cd, np.ones(int(m.sum())))
        rev = CSR(n_all, cd, cs, np.ones(int(m.sum())))
        live = np.arange(n_vocab)[fwd.deg[:n_vocab] > 0]
        if not len(live):
            continue
        start = np.repeat(live, per_cat)
        rng.shuffle(start)
        w = np.empty((len(start), walk_length + 1), np.int32)
        w[:, 0] = start
        cur = start.copy()
        for t in range(walk_length):
            cur = (fwd if t % 2 == 0 else rev).step(cur, rng)
            w[:, t + 1] = cur
        out.append(w)
    if not out:
        raise RuntimeError("chemistry graph produced no walks")
    return np.concatenate(out)


def walks_core(ii: IIGraph, chem: ChemGraph, rng, walks_per_node: int,
               walk_length: int, ii_repeat: float = 10.0) -> np.ndarray:
    """Union walk: chemistry edges plus ingredient-ingredient edges up-weighted
    by ``ii_repeat``.

    Both edge families live in one graph, so a single walk can hop
    ``garlic -> allicin -> onion`` (chemistry) and then ``onion -> tomato``
    (recipe context). Weights are normalised by their mean before scaling, so
    ``ii_repeat`` means "how many times more likely than a chemistry edge"
    independently of the NPMI scale.
    """
    n_vocab = ii.n_vocab
    n_all = n_vocab + chem.n_compounds
    ic_s = np.concatenate([chem.src, chem.dst + n_vocab])
    ic_d = np.concatenate([chem.dst + n_vocab, chem.src])
    ic_w = np.ones(len(ic_s))
    s2, d2, w2 = ii.symmetric()
    w2 = w2 / w2.mean() * ii_repeat
    g = CSR(n_all, np.concatenate([ic_s, s2]), np.concatenate([ic_d, d2]),
            np.concatenate([ic_w, w2]))
    start = np.repeat(np.arange(n_vocab)[g.deg[:n_vocab] > 0], walks_per_node)
    rng.shuffle(start)
    return _run(g, start, walk_length, rng)
