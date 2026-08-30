"""SVD-PPMI, GloVe and chemistry SVD."""
from __future__ import annotations

import numpy as np

from ingredient_model.config import SEED
from ingredient_model.data import load_chem_graph, load_ii_graph
from ingredient_model.registry import register
from ingredient_model.spec import TrainContext, TrainResult

DEFAULTS = dict(d_model=300)
GLOVE_DEFAULTS = dict(d_model=300, epochs=200, lr=0.05, x_max=100.0, alpha=0.75,
                      chunk=40_000)


def ppmi(C: np.ndarray, shift: float = 1.0) -> np.ndarray:
    """Shifted positive PMI — the matrix SGNS implicitly factors."""
    tot = C.sum()
    if tot <= 0:
        return C
    row, col = C.sum(1, keepdims=True), C.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        M = np.log((C * tot) / np.maximum(row * col, 1e-12)) - np.log(shift)
    return np.nan_to_num(np.maximum(M, 0.0), nan=0.0, posinf=0.0, neginf=0.0)


def svd_embed(M: np.ndarray, d: int) -> np.ndarray:
    """Symmetric factorisation ``U * sqrt(s)``.

    Using ``U`` alone yields systematically sharper geometry than SGNS and would
    make the comparison a comparison of scalings rather than of methods.
    """
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    d = min(d, len(s))
    return U[:, :d] * np.sqrt(s[:d])


@register(name="svd-ppmi", family="factorization", cost_hint="cheap",
          defaults=DEFAULTS, tags=("baseline", "closed-form"),
          requires=("ii_graph_train",),
          description="Truncated SVD of the shifted-PPMI co-occurrence matrix")
def train_svd_ppmi(ctx: TrainContext) -> TrainResult:
    p = {**DEFAULTS, **dict(ctx.params)}
    C = load_ii_graph(ctx.graph).dense("count")
    W = svd_embed(ppmi(C, float(p.get("shift", 1.0))), int(p["d_model"]))
    return TrainResult(embedding=W, metadata=dict(p))


@register(name="glove", family="factorization", cost_hint="moderate",
          defaults=GLOVE_DEFAULTS, tags=("closed-form",),
          requires=("ii_graph_train",),
          description="GloVe weighted log-count factorisation by full-batch AdaGrad")
def train_glove(ctx: TrainContext) -> TrainResult:
    """Full-batch AdaGrad over the non-zero entries.

    Gradients are accumulated in chunks and applied once per epoch, so this is
    still exact full-batch AdaGrad — chunking changes memory, not arithmetic.
    It is not optional: at 366k non-zeros and d=300 a single ``(nnz, d)``
    temporary is 879 MB and the update needs four live at once.
    """
    p = {**GLOVE_DEFAULTS, **dict(ctx.params)}
    d, chunk = int(p["d_model"]), int(p["chunk"])
    C = load_ii_graph(ctx.graph).dense("count")
    rng = np.random.default_rng(ctx.seed or SEED)
    n = C.shape[0]
    i, j = np.nonzero(C)
    x = C[i, j]
    w = np.minimum(1.0, (x / float(p["x_max"])) ** float(p["alpha"]))
    logx = np.log(x)

    W = (rng.random((n, d)) - 0.5) / d
    Wc = (rng.random((n, d)) - 0.5) / d
    b, bc = np.zeros(n), np.zeros(n)
    gW, gWc = np.ones_like(W), np.ones_like(Wc)
    gb, gbc = np.ones(n), np.ones(n)
    history = []

    for ep in range(int(p["epochs"])):
        aW, aWc = np.zeros_like(W), np.zeros_like(Wc)
        ab, abc = np.zeros(n), np.zeros(n)
        loss = 0.0
        for s in range(0, len(i), chunk):
            ii, jj = i[s:s + chunk], j[s:s + chunk]
            diff = (np.einsum("ij,ij->i", W[ii], Wc[jj])
                    + b[ii] + bc[jj] - logx[s:s + chunk])
            wd = w[s:s + chunk] * diff
            loss += float((wd * diff).sum())
            np.add.at(aW, ii, wd[:, None] * Wc[jj])
            np.add.at(aWc, jj, wd[:, None] * W[ii])
            np.add.at(ab, ii, wd)
            np.add.at(abc, jj, wd)
        for acc, P, G in ((aW, W, gW), (aWc, Wc, gWc), (ab, b, gb), (abc, bc, gbc)):
            G += acc ** 2
            P -= float(p["lr"]) * acc / np.sqrt(G)
        history.append(loss)
        if ep % 50 == 0 or ep == int(p["epochs"]) - 1:
            print(f"  glove epoch {ep + 1}/{p['epochs']}  loss {loss:,.1f}",
                  flush=True)
    # Summing both matrices, as the reference implementation recommends.
    return TrainResult(embedding=W + Wc,
                       metadata={"loss_history": history, **p})


@register(name="chem-svd", family="factorization", cost_hint="cheap",
          defaults=DEFAULTS, tags=("chemistry", "closed-form"),
          requires=("chem_graph", "ii_graph_train"),
          description="IDF-weighted SVD of the ingredient-compound incidence matrix")
def train_chem_svd(ctx: TrainContext) -> TrainResult:
    """The fair chemistry-only baseline.

    Uses exactly the information the chemistry walk has, without random walks,
    which is what separates "chemistry is uninformative here" from "the walk
    schema wasted it".
    """
    p = {**DEFAULTS, **dict(ctx.params)}
    n = load_ii_graph(ctx.graph).n_vocab
    A = load_chem_graph().incidence(n)
    # Down-weight ubiquitous compounds; "contains water-soluble things" is not
    # a signal that distinguishes any two ingredients.
    idf = np.log(1.0 + n / np.maximum(A.sum(0, keepdims=True), 1.0))
    return TrainResult(embedding=svd_embed(A * idf, int(p["d_model"])),
                       metadata=dict(p))
