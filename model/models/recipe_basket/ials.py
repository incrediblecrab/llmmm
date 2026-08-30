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


def _solve_items(X: np.ndarray, corpus, alpha: float, reg: float,
                 n_vocab: int) -> np.ndarray:
    """Least-squares update for every ingredient factor.

    The transpose of the user problem, but the "items" grouping is ragged, so
    instead of bucketing this accumulates each ingredient's normal equations by
    scattering over the recipes that contain it. At 1,790 ingredients the
    accumulator is small enough to hold densely.
    """
    d = X.shape[1]
    XtX = X.T @ X
    base = XtX + reg * np.eye(d)
    A = np.repeat(base[None], n_vocab, axis=0)
    b = np.zeros((n_vocab, d), np.float64)

    # Walk recipes in blocks and scatter their outer products to the
    # ingredients they contain. `np.add.at` on the (n_vocab, d, d) accumulator
    # is the only scatter available, so blocks are kept large to amortise it.
    block = 200_000
    for s in range(0, corpus.n_recipes, block):
        e = min(s + block, corpus.n_recipes)
        lo, hi = corpus.offsets[s], corpus.offsets[e]
        items = corpus.flat[lo:hi].astype(np.int64)
        owner = np.repeat(np.arange(s, e), corpus.sizes[s:e])
        Xu = X[owner]
        np.add.at(A, items, alpha * np.einsum("md,me->mde", Xu, Xu))
        np.add.at(b, items, (1.0 + alpha) * Xu)
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
    t0 = time.time()
    for it in range(int(p["iterations"])):
        X = _solve_users(Y, corpus, alpha, reg, int(p["chunk"]))
        Y = _solve_items(X, corpus, alpha, reg, n_vocab)
        print(f"  iteration {it + 1:>2}/{p['iterations']}  "
              f"|Y| {np.linalg.norm(Y):.2f}  {time.time() - t0:.0f}s", flush=True)
    return TrainResult(embedding=Y.astype(np.float64),
                       metadata={"n_recipes": corpus.n_recipes, **p})
