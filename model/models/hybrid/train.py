"""Blends of previously trained spaces."""
from __future__ import annotations

import numpy as np

from ingredient_model.artifacts import load_embedding, resolve_run
from ingredient_model.config import PATHS
from ingredient_model.eval.metrics import unit
from ingredient_model.registry import register
from ingredient_model.spec import TrainContext, TrainResult

CONCAT_DEFAULTS = dict(runs="", weights="", d_model=0, whiten_each=False)
RESIDUAL_DEFAULTS = dict(base="", correction="", d_model=0, ridge=1.0,
                         strength=1.0)


def _load_runs(spec: str, root=None) -> tuple[list[str], list[np.ndarray]]:
    names = [s.strip() for s in str(spec).split(",") if s.strip()]
    if len(names) < 2:
        raise ValueError(
            "hybrid models combine at least two runs: "
            "--set runs=svd-ppmi-recipe-holdout-s0,ease-recipe-holdout-s0")
    mats = []
    for n in names:
        # Resolved rather than assumed to sit at runs/<id>: inside a sweep the
        # inputs are siblings under runs/<experiment>/, which is exactly where
        # a blend declared in that same sweep will look for them.
        d = resolve_run(n, root=root)
        mats.append(load_embedding(d).astype(np.float64))
    rows = {m.shape[0] for m in mats}
    if len(rows) != 1:
        raise ValueError(f"runs have different vocabularies: {sorted(rows)}")
    return names, mats


@register(name="concat", family="hybrid", cost_hint="cheap",
          defaults=CONCAT_DEFAULTS, tags=("blend", "post-hoc"),
          requires=("ii_graph_train",),
          description="Weighted concatenation of two or more trained spaces")
def train_concat(ctx: TrainContext) -> TrainResult:
    """Normalise each space, scale it, and stack.

    Row-normalising before stacking is what makes the weights mean anything. The
    spaces arrive on wildly different scales — an SVD of a PPMI matrix carries
    singular values in the hundreds while a trained SGNS table sits near unit
    norm — so concatenating them raw is not a 50/50 blend, it is whichever space
    happened to have the larger numbers.
    """
    p = {**CONCAT_DEFAULTS, **dict(ctx.params)}
    names, mats = _load_runs(p["runs"], root=ctx.out_dir.parent)
    if p["weights"]:
        w = [float(x) for x in str(p["weights"]).split(",")]
        if len(w) != len(mats):
            raise ValueError(f"{len(w)} weights for {len(mats)} runs")
    else:
        w = [1.0] * len(mats)

    parts = []
    for name, M, wi in zip(names, mats, w):
        U = unit(M)
        if p["whiten_each"]:
            from ingredient_model.eval.metrics import all_but_top
            U = unit(all_but_top(U, k=3))
        parts.append(U * wi)
        print(f"  {name:<40}d={M.shape[1]:<5} weight {wi}")

    W = np.hstack(parts)
    d = int(p["d_model"])
    if d and d < W.shape[1]:
        # Reduce *after* stacking, so the projection is chosen knowing both
        # spaces. Truncating each one first would discard directions that only
        # look redundant in isolation.
        Wc = W - W.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(Wc, full_matrices=False)
        W = Wc @ Vt[:d].T
    return TrainResult(embedding=W.astype(np.float32),
                       metadata={"sources": names, "weights": w,
                                 "d_out": int(W.shape[1]), **p})


@register(name="residual", family="hybrid", cost_hint="cheap",
          defaults=RESIDUAL_DEFAULTS, tags=("blend", "post-hoc"),
          requires=("ii_graph_train",),
          description="Add only the part of a second space the first cannot express")
def train_residual(ctx: TrainContext) -> TrainResult:
    """Keep the base space, append what the correction knows that it does not.

    A plain concatenation of two spaces that largely agree mostly duplicates
    information and inflates the dimension for nothing. Regressing the
    correction onto the base and keeping the *residual* appends only genuinely
    new directions, so the combined space grows by roughly the amount of new
    information rather than by the size of the second table.

    ``strength`` scales the residual. At 0 this is exactly the base space, which
    makes it the control the blend must beat.
    """
    p = {**RESIDUAL_DEFAULTS, **dict(ctx.params)}
    if not (p["base"] and p["correction"]):
        raise ValueError("residual needs --set base=<run> --set correction=<run>")
    _, (B, C) = _load_runs(f"{p['base']},{p['correction']}",
                           root=ctx.out_dir.parent)

    Bu, Cu = unit(B), unit(C)
    X = np.hstack([Bu, np.ones((len(Bu), 1))])
    lam = float(p["ridge"])
    M = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Cu)
    resid = Cu - X @ M

    explained = 1.0 - float(np.linalg.norm(resid) ** 2 / np.linalg.norm(Cu) ** 2)
    print(f"  base explains {explained:.1%} of the correction space; "
          f"appending the remaining {1 - explained:.1%}")

    rn = np.linalg.norm(resid, axis=1, keepdims=True)
    resid = resid / np.maximum(rn, 1e-12) * float(p["strength"])
    W = np.hstack([Bu, resid])
    d = int(p["d_model"])
    if d and d < W.shape[1]:
        Wc = W - W.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(Wc, full_matrices=False)
        W = Wc @ Vt[:d].T
    return TrainResult(
        embedding=W.astype(np.float32),
        metadata={"base": p["base"], "correction": p["correction"],
                  "variance_explained_by_base": explained,
                  "d_out": int(W.shape[1]), **p},
        extra_arrays={"projection": M.astype(np.float32)})
