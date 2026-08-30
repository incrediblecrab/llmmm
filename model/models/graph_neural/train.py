"""LightGCN and SGC propagation."""
from __future__ import annotations

import time

import numpy as np

from ingredient_model.data import load_ii_graph
from ingredient_model.registry import register
from ingredient_model.spec import TrainContext, TrainResult

LIGHTGCN_DEFAULTS = dict(d_model=300, layers=3, epochs=60, lr=0.05,
                         reg=1e-5, batch_size=65536, negative_samples=1,
                         weight="npmi")
SGC_DEFAULTS = dict(d_model=300, layers=2, weight="npmi", self_loops=True)


def normalised_adjacency(g, weight: str = "npmi", self_loops: bool = True
                         ) -> np.ndarray:
    """Symmetrically normalised adjacency ``D^-1/2 (A + I) D^-1/2``.

    Symmetric normalisation, not row normalisation: it keeps the operator
    symmetric so its spectrum is real and bounded, which is what stops repeated
    propagation from exploding towards the highest-degree nodes. Without it, a
    three-layer stack simply returns salt.
    """
    n = g.n_vocab
    A = np.zeros((n, n), np.float64)
    src, dst = np.asarray(g.src, int), np.asarray(g.dst, int)
    if weight == "npmi":
        # NPMI is signed; shift into positive territory so the operator stays a
        # valid diffusion rather than mixing in reflections.
        w = np.asarray(g.npmi, float)
        w = w - w.min() + 1e-3
    else:
        w = np.log1p(np.asarray(g.count, float))
    A[src, dst] = w
    A[dst, src] = w
    if self_loops:
        np.fill_diagonal(A, A.diagonal() + 1.0)
    d = np.maximum(A.sum(1), 1e-12) ** -0.5
    return A * d[:, None] * d[None, :]


@register(name="sgc", family="graph_neural", cost_hint="cheap",
          defaults=SGC_DEFAULTS, tags=("propagation", "closed-form"),
          requires=("ii_graph_train",),
          description="Simple graph convolution — k-step propagation of a spectral base")
def train_sgc(ctx: TrainContext) -> TrainResult:
    """Propagate a spectral base ``k`` times.

    SGC's claim is that the depth of a graph network buys almost nothing beyond
    the smoothing that its propagation performs, so removing every non-linearity
    leaves the accuracy intact. Here that makes it a control on ``lightgcn``: if
    the trained model does not beat this closed-form one, its parameters are not
    earning their cost.
    """
    p = {**SGC_DEFAULTS, **dict(ctx.params)}
    g = load_ii_graph(ctx.graph)
    S = normalised_adjacency(g, str(p["weight"]), bool(p["self_loops"]))
    vals, vecs = np.linalg.eigh(S)
    order = np.argsort(-np.abs(vals))
    d = min(int(p["d_model"]), len(vals))
    base = vecs[:, order[:d]] * np.abs(vals[order[:d]]) ** 0.5

    W = base
    for _ in range(int(p["layers"])):
        W = S @ W
    return TrainResult(embedding=W, metadata=dict(p))


@register(name="lightgcn", family="graph_neural", cost_hint="moderate",
          defaults=LIGHTGCN_DEFAULTS, tags=("propagation", "bpr"),
          requires=("ii_graph_train",),
          description="LightGCN propagation trained with a BPR ranking loss")
def train_lightgcn(ctx: TrainContext) -> TrainResult:
    """Layer-averaged linear propagation, fitted with BPR.

    The output is the *propagated* representation rather than the free parameter
    table, because that is what the model actually scores with — exporting the
    base table would ship something the model never uses.
    """
    import torch

    p = {**LIGHTGCN_DEFAULTS, **dict(ctx.params)}
    g = load_ii_graph(ctx.graph)
    n, d = g.n_vocab, int(p["d_model"])
    torch.manual_seed(ctx.seed)
    rng = np.random.default_rng(ctx.seed)

    S = torch.tensor(normalised_adjacency(g, str(p["weight"])),
                     dtype=torch.float32, device=ctx.device)
    src = torch.tensor(np.asarray(g.src, np.int64), device=ctx.device)
    dst = torch.tensor(np.asarray(g.dst, np.int64), device=ctx.device)
    base = torch.nn.Parameter(
        torch.randn(n, d, device=ctx.device) / d ** 0.5)
    opt = torch.optim.Adam([base], lr=float(p["lr"]))
    n_layers, K = int(p["layers"]), int(p["negative_samples"])
    B = int(p["batch_size"])

    def propagate() -> torch.Tensor:
        # Layer averaging, not the last layer alone: each depth captures a
        # different neighbourhood radius and averaging is what keeps a deep
        # stack from over-smoothing into the graph's dominant eigenvector.
        h, acc = base, base
        for _ in range(n_layers):
            h = S @ h
            acc = acc + h
        return acc / (n_layers + 1)

    history, t0 = [], time.time()
    for ep in range(int(p["epochs"])):
        perm = torch.tensor(rng.permutation(len(src)), device=ctx.device)
        tot, nb = 0.0, 0
        for i in range(0, len(perm), B):
            sel = perm[i:i + B]
            E = propagate()
            u, v = src[sel], dst[sel]
            neg = torch.randint(0, n, (len(sel), K), device=ctx.device)
            pos_s = (E[u] * E[v]).sum(1, keepdim=True)
            neg_s = torch.bmm(E[neg], E[u].unsqueeze(2)).squeeze(2)
            loss = (-torch.nn.functional.logsigmoid(pos_s - neg_s).mean()
                    + float(p["reg"]) * base.pow(2).sum() / len(sel))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        history.append(tot / max(nb, 1))
        if ep % 10 == 0 or ep == int(p["epochs"]) - 1:
            print(f"  epoch {ep + 1:>3}/{p['epochs']}  loss {history[-1]:.4f}  "
                  f"{time.time() - t0:.0f}s", flush=True)

    with torch.no_grad():
        W = propagate().cpu().numpy()
    return TrainResult(embedding=W, metadata={"loss_history": history, **p},
                       extra_arrays={"base_table":
                                     base.detach().cpu().numpy()})
