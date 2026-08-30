#!/usr/bin/env python3
"""H2 baselines: can a closed-form factorisation match SGNS here?

At 1,790 ingredients the co-occurrence matrix is 1790x1790 -- small enough to
factor exactly in seconds. SGNS is an implicit factorisation of shifted PPMI
(Levy & Goldberg 2014), so if the corpus signal is what matters rather than the
optimiser, these should land in the same place. They also cannot collapse,
which makes them a useful control on H1.

    python tools/train_factorization.py --method svd-ppmi
    python tools/train_factorization.py --method glove
    python tools/train_factorization.py --method chem-svd
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DERIVED = Path(os.environ.get("EPICURE_DERIVED", ROOT / "data" / "derived"))
OUT_DIR = Path(os.environ.get("EPICURE_OUT", ROOT / "models"))
II_GRAPH = os.environ.get("EPICURE_II_GRAPH", "ii_graph.npz")
DIM = 300
SEED = 20260805


def cooc_matrix() -> tuple[np.ndarray, int]:
    z = np.load(DERIVED / II_GRAPH, allow_pickle=True)
    n = len(z["itos"])
    C = np.zeros((n, n), np.float64)
    # the graph is stored one-directional; symmetrise to match training
    np.add.at(C, (z["src"].astype(int), z["dst"].astype(int)), z["count"].astype(float))
    np.add.at(C, (z["dst"].astype(int), z["src"].astype(int)), z["count"].astype(float))
    return C, n


def ppmi(C: np.ndarray, shift: float = 1.0) -> np.ndarray:
    """Shifted positive PMI -- the matrix SGNS implicitly factors."""
    tot = C.sum()
    if tot <= 0:
        return C
    row = C.sum(1, keepdims=True)
    col = C.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        M = np.log((C * tot) / np.maximum(row * col, 1e-12)) - np.log(shift)
    return np.nan_to_num(np.maximum(M, 0.0), nan=0.0, posinf=0.0, neginf=0.0)


def svd_embed(M: np.ndarray, d: int) -> np.ndarray:
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    d = min(d, len(s))
    # sqrt of the singular values is the symmetric factorisation, and is what
    # makes SVD-PPMI comparable to SGNS rather than systematically sharper
    return U[:, :d] * np.sqrt(s[:d])


def glove(C: np.ndarray, d: int, epochs: int = 200, lr: float = 0.05,
          x_max: float = 100.0, alpha: float = 0.75,
          chunk: int = 40_000) -> np.ndarray:
    """GloVe by full-batch AdaGrad on the non-zero entries.

    Gradients are accumulated over chunks of non-zero pairs and applied once at
    the end of each epoch, so this is still exact full-batch AdaGrad -- the
    chunking changes memory, not arithmetic. It is not optional: at 366k
    non-zeros and d=300 a single (nnz, d) temporary is 879 MB, and the update
    needs four of them live at once, which SIGKILLed a 4 GB node after epoch 1.
    """
    rng = np.random.default_rng(SEED)
    n = C.shape[0]
    i, j = np.nonzero(C)
    x = C[i, j]
    w = np.minimum(1.0, (x / x_max) ** alpha)
    logx = np.log(x)

    W = (rng.random((n, d)) - 0.5) / d
    Wc = (rng.random((n, d)) - 0.5) / d
    b = np.zeros(n)
    bc = np.zeros(n)
    gW, gWc = np.ones_like(W), np.ones_like(Wc)
    gb, gbc = np.ones(n), np.ones(n)

    for ep in range(epochs):
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
        for acc, P, G in ((aW, W, gW), (aWc, Wc, gWc),
                          (ab, b, gb), (abc, bc, gbc)):
            G += acc ** 2
            P -= lr * acc / np.sqrt(G)
        if ep % 50 == 0 or ep == epochs - 1:
            print(f"  glove epoch {ep+1}/{epochs} loss {loss:,.1f}", flush=True)
    # both matrices, as the reference implementation recommends
    return W + Wc


def chem_svd(n: int, d: int) -> np.ndarray:
    """Factor the ingredient x compound incidence matrix directly.

    This is the fair chemistry-only baseline for H5: it uses exactly the
    information Chem has, without random walks, so it separates "chemistry is
    uninformative" from "the walk schema wasted it".
    """
    z = np.load(DERIVED / "flavor_graph.npz", allow_pickle=True)
    s, dd = z["src"].astype(int), z["dst"].astype(int)
    n_comp = int(dd.max()) + 1
    A = np.zeros((n, n_comp), np.float64)
    A[s, dd] = 1.0
    # down-weight ubiquitous compounds; water-soluble everything is not a signal
    idf = np.log(1.0 + n / np.maximum(A.sum(0, keepdims=True), 1.0))
    return svd_embed(A * idf, d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["svd-ppmi", "glove", "chem-svd"])
    ap.add_argument("--dim", type=int, default=DIM)
    a = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"method={a.method} graph={II_GRAPH} dim={a.dim}", flush=True)
    C, n = cooc_matrix()
    print(f"  cooc matrix {C.shape}, {int((C > 0).sum()):,} non-zero", flush=True)

    if a.method == "svd-ppmi":
        W = svd_embed(ppmi(C), a.dim)
    elif a.method == "glove":
        W = glove(C, a.dim)
    else:
        W = chem_svd(n, a.dim)

    name = f"epicure_{a.method}"
    np.save(OUT_DIR / f"{name}.npy", W)
    (OUT_DIR / f"{name}.json").write_text(json.dumps(
        {"method": a.method, "dim": int(W.shape[1]), "vocab_size": int(W.shape[0]),
         "graph": II_GRAPH}, indent=2))
    print(f"  saved {OUT_DIR}/{name}.npy {W.shape}", flush=True)


if __name__ == "__main__":
    main()
