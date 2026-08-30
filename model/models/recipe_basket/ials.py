"""Implicit alternating least squares (Hu, Koren & Volinsky, ICDM 2008).

Treats a recipe as a user and every ingredient it contains as an implicit
positive with confidence ``1 + alpha``. Ingredients absent from a recipe are
weak negatives at confidence 1, not missing data — which is the right model
here, because a recipe that omits saffron is genuinely evidence about saffron.

The item factors are the embedding; recipe factors are nuisance parameters that
exist only to explain away recipe-level effects such as length and style.

Two things keep this tractable at corpus scale:

* **The Gram trick.** ``Yᵀ C_u Y = YᵀY + α · Y_uᵀ Y_u`` where ``Y_u`` covers only
  the ingredients recipe ``u`` contains, so per-recipe cost depends on recipe
  length rather than on vocabulary size.
* **Length bucketing.** Recipes of equal length form a dense ``(m, k, d)`` block,
  so a whole bucket's normal equations are one ``einsum`` and one batched solve.
  A per-recipe Python loop over millions of recipes is not viable.
"""
from __future__ import annotations

import time

import numpy as np

from ingredient_model.data import load_recipes
from ingredient_model.data.recipes import RECIPE_IDS
from ingredient_model.registry import register
from ingredient_model.spec import TrainContext, TrainResult

DEFAULTS = dict(d_model=128, iterations=15, reg=0.05, alpha=40.0,
                max_recipes=400_000, chunk=4096)


def _solve_users(Y: np.ndarray, corpus, alpha: float, reg: float,
                 chunk: int) -> np.ndarray:
    """Least-squares update for every recipe factor, bucketed by length."""
    d = Y.shape[1]
    YtY = Y.T @ Y
    base = YtY + reg * np.eye(d)
    X = np.zeros((corpus.n_recipes, d), np.float32)
    lens = corpus.sizes
    for k in range(1, int(lens.max()) + 1):
        rows = np.nonzero(lens == k)[0]
        if not len(rows):
            continue
        for s in range(0, len(rows), chunk):
            r = rows[s:s + chunk]
            ids = corpus.flat[corpus.offsets[r][:, None] + np.arange(k)].astype(np.int64)
            Yu = Y[ids]                                        # (m, k, d)
            A = base + alpha * np.einsum("mkd,mke->mde", Yu, Yu)
            b = (1.0 + alpha) * Yu.sum(1)                      # (m, d)
            X[r] = np.linalg.solve(A, b[:, :, None])[:, :, 0]
    return X


def _group_slots(corpus, n_vocab: int) -> tuple[np.ndarray, np.ndarray]:
    """Index every (ingredient, recipe) slot once, grouped by ingredient.

    Returns the recipe id of each slot sorted by its ingredient, plus the
    boundaries of each ingredient's run. The grouping depends only on the
    corpus, so it is built once and reused by every iteration.
    """
    n = corpus.n_recipes
    items = corpus.flat[corpus.offsets[0]:corpus.offsets[n]].astype(np.int64)
    owner = np.repeat(np.arange(n, dtype=np.int64), corpus.sizes[:n])
    order = np.argsort(items, kind="stable")
    bounds = np.searchsorted(items[order], np.arange(n_vocab + 1))
    return owner[order], bounds


def _solve_items(X: np.ndarray, corpus, alpha: float, reg: float,
                 n_vocab: int, owner: np.ndarray,
                 bounds: np.ndarray) -> np.ndarray:
    """Least-squares update for every ingredient factor.

    The transpose of the recipe problem. Each ingredient's normal equations are
    the sum of outer products over the recipes containing it, which is the
    Gram matrix of those recipe factors — so it is one BLAS call per
    ingredient over a gathered block, not a scatter of per-slot outer products.

    The distinction is not stylistic. Materialising the outer products first
    costs (slots, d, d) floats, which at this corpus size is over a hundred
    gigabytes and is killed by the OS; the Gram form never holds more than one
    ingredient's rows at a time. At 1,790 ingredients the loop overhead is
    negligible and the accumulator stays dense.
    """
    d = X.shape[1]
    base = X.T @ X + reg * np.eye(d)
    A = np.repeat(base[None], n_vocab, axis=0)
    b = np.zeros((n_vocab, d), np.float64)

    for i in range(n_vocab):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        if hi <= lo:
            continue
        Xi = X[owner[lo:hi]]
        A[i] += alpha * (Xi.T @ Xi)
        b[i] = (1.0 + alpha) * Xi.sum(0)
    return np.linalg.solve(A, b[:, :, None])[:, :, 0].astype(np.float32)


@register(name="ials", family="recipe_basket", cost_hint="heavy",
          defaults=DEFAULTS, tags=("recipes", "implicit-feedback"),
          requires=("recipes",),
          description="Implicit ALS over the recipe x ingredient matrix")
def train_ials(ctx: TrainContext) -> TrainResult:
    p = {**DEFAULTS, **dict(ctx.params)}
    d, alpha, reg = int(p["d_model"]), float(p["alpha"]), float(p["reg"])
    corpus = load_recipes(ctx.corpus or RECIPE_IDS)
    max_r = int(p["max_recipes"])
    if max_r and corpus.n_recipes > max_r:
        # Recipe factors are (n_recipes, d) floats held in memory, so the corpus
        # is subsampled rather than the model shrunk. Ingredient factors — the
        # thing we keep — converge long before the recipe count is exhausted.
        rng = np.random.default_rng(ctx.seed)
        corpus = corpus.select(
            np.sort(rng.choice(corpus.n_recipes, max_r, replace=False)))
    n_vocab = corpus.n_vocab
    print(f"  {corpus.n_recipes:,} recipes x {n_vocab} ingredients, d={d}",
          flush=True)

    rng = np.random.default_rng(ctx.seed)
    Y = (rng.normal(size=(n_vocab, d)) * 0.01).astype(np.float32)
    owner, bounds = _group_slots(corpus, n_vocab)
    t0 = time.time()
    for it in range(int(p["iterations"])):
        X = _solve_users(Y, corpus, alpha, reg, int(p["chunk"]))
        Y = _solve_items(X, corpus, alpha, reg, n_vocab, owner, bounds)
        print(f"  iteration {it + 1:>2}/{p['iterations']}  "
              f"|Y| {np.linalg.norm(Y):.2f}  {time.time() - t0:.0f}s", flush=True)
    return TrainResult(embedding=Y.astype(np.float64),
                       metadata={"n_recipes": corpus.n_recipes, **p})
