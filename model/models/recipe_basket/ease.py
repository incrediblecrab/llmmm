"""EASE — Embarrassingly Shallow Autoencoder (Steck, WWW 2019).

Fits an item-item weight matrix ``B`` minimising ``||X - XB||² + λ||B||²``
subject to ``diag(B) = 0``. The constraint is the whole idea: without it the
optimum is the identity, and with it every ingredient must be reconstructed from
the *other* ingredients in its recipes.

The closed form is

    P = (XᵀX + λI)⁻¹
    B = I - P · diagMat(1 / diag(P))

so the corpus enters only through the Gram matrix ``XᵀX``. That is what makes
this practical at 4.6M recipes: the Gram matrix is 1790×1790 regardless of how
many recipes produced it, and the whole fit is one matrix inverse.

Unlike a co-occurrence count, ``B`` is a *conditional* quantity — the inverse
divides out ingredients that co-occur only because both co-occur with salt.
This is the model most likely to escape the popularity degeneration that
cosine-on-counts suffers from.
"""
from __future__ import annotations

import numpy as np

from ingredient_model.data import load_recipes
from ingredient_model.data.recipes import RECIPE_IDS
from ingredient_model.registry import register
from ingredient_model.spec import TrainContext, TrainResult

DEFAULTS = dict(d_model=300, reg=250.0, max_recipes=0)


def gram_matrix(corpus, max_recipes: int = 0, seed: int = 0) -> np.ndarray:
    """``XᵀX`` — recipes in which each pair of ingredients both appear.

    Accumulated by ingredient-count bucket so each bucket is a dense ``(m, k)``
    block and every pair in it comes from one ``triu_indices`` gather. A
    per-recipe Python loop over 4.6M recipes takes tens of minutes; this runs in
    well under one.
    """
    if max_recipes and corpus.n_recipes > max_recipes:
        rng = np.random.default_rng(seed)
        corpus = corpus.select(
            np.sort(rng.choice(corpus.n_recipes, max_recipes, replace=False)))
    n = corpus.n_vocab
    lens = corpus.sizes
    flat = corpus.flat
    # One flat int64 accumulator rather than a 2-D add per chunk: `bincount`
    # into a flat buffer is several times faster than `np.add.at` on a matrix,
    # and avoids allocating a fresh 25 MB matrix per chunk.
    acc = np.zeros(n * n, np.int64)
    for k in range(2, int(lens.max()) + 1):
        rows = np.nonzero(lens == k)[0]
        if not len(rows):
            continue
        iu, ju = np.triu_indices(k, 1)
        step = max(int(20_000_000 / max(len(iu), 1)), 1)
        for s in range(0, len(rows), step):
            r = rows[s:s + step]
            ids = flat[corpus.offsets[r][:, None] + np.arange(k)].astype(np.int64)
            acc += np.bincount(ids[:, iu].ravel() * n + ids[:, ju].ravel(),
                               minlength=n * n)
    G = acc.reshape(n, n).astype(np.float64)
    G += G.T
    np.fill_diagonal(G, np.bincount(flat.astype(np.int64), minlength=n))
    return G


def embed_from_scores(B: np.ndarray, d: int) -> np.ndarray:
    """Factor an item-item score matrix into vectors whose dot products
    approximate it.

    ``B`` is asymmetric and indefinite, so it is symmetrised and truncated to
    its top ``d`` *positive* eigenvalues. Negative eigenvalues encode
    anti-similarity, which a dot-product space cannot represent — discarding
    them is a real approximation and is why the retained fraction is reported in
    the run metadata rather than left implicit.
    """
    S = (B + B.T) / 2.0
    vals, vecs = np.linalg.eigh(S)
    order = np.argsort(-vals)
    vals, vecs = vals[order], vecs[:, order]
    keep = vals > 0
    d = min(d, int(keep.sum()))
    if d == 0:
        raise RuntimeError("score matrix has no positive eigenvalues")
    W = vecs[:, :d] * np.sqrt(vals[:d])
    retained = float(vals[:d].sum() / np.abs(vals).sum())
    return W, retained


@register(name="ease", family="recipe_basket", cost_hint="cheap",
          defaults=DEFAULTS, tags=("closed-form", "recipes", "conditional"),
          requires=("recipes",),
          description="Closed-form item-item ridge autoencoder over recipe baskets")
def train_ease(ctx: TrainContext) -> TrainResult:
    p = {**DEFAULTS, **dict(ctx.params)}
    corpus = load_recipes(ctx.corpus or RECIPE_IDS)
    print(f"  gram matrix over {corpus.n_recipes:,} recipes", flush=True)
    G = gram_matrix(corpus, int(p["max_recipes"]), ctx.seed)

    lam = float(p["reg"])
    G_reg = G.copy()
    np.fill_diagonal(G_reg, np.diag(G_reg) + lam)
    P = np.linalg.inv(G_reg)
    B = -P / np.diag(P)[None, :]
    np.fill_diagonal(B, 0.0)

    W, retained = embed_from_scores(B, int(p["d_model"]))
    print(f"  d={W.shape[1]}  spectrum retained {retained:.1%}", flush=True)
    def scorer(ctx_ids: np.ndarray) -> np.ndarray:
        """Sum the item-item weights of the visible ingredients.

        This is EASE's actual prediction rule, ``x @ B``. The exported embedding
        is a symmetrised, positive-truncated factorisation of ``B`` that
        discards roughly half its spectrum, so scoring through ``B`` directly is
        the only way to see what the model learned rather than what survived
        being squeezed into a vector table.
        """
        return B[ctx_ids].sum(1)

    return TrainResult(
        embedding=W,
        scorer=scorer,
        metadata={"spectrum_retained": retained, "n_recipes": corpus.n_recipes, **p},
        extra_arrays={"item_scores": B.astype(np.float32)})
