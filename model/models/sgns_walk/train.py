"""Skip-gram with negative sampling over random walks.

Four implementation details below are load-bearing — each one, done the obvious
way instead, produces a model that still returns plausible nearest neighbours
and so survives a spot check while being geometrically degenerate:

1. **Two embedding tables.** Sharing one makes ``centre · context`` a
   self-similarity whose trivial minimiser is "point every vector the same way".
2. **Init scale ``1/sqrt(d)``.** word2vec's ``U(-0.5/d, 0.5/d)`` with a zeroed
   output table sits at a saddle; singular directions then grow one at a time
   and 20 epochs leaves the space effectively rank-3.
3. **SparseAdam, not Adam.** A dense optimiser steps *every* row on *every*
   iteration, because a row that received no gradient still has a non-zero
   momentum buffer from earlier batches. Over thousands of steps that drags the
   whole table along one accumulated common direction.
4. **Negatives drawn per centre.** One shared draw repels every centre in the
   batch from the same handful of points, manufacturing exactly the common
   direction the other three details exist to avoid.
"""
from __future__ import annotations

import time

import numpy as np

from ingredient_model.data import load_chem_graph, load_ii_graph
from ingredient_model.registry import register
from ingredient_model.spec import TrainContext, TrainResult

from .walks import walks_chem, walks_cooc, walks_core

DEFAULTS = dict(d_model=300, walks_per_node=100, walk_length=50, context_size=7,
                negative_samples=5, batch_size=32768, lr=0.0025, epochs=20,
                init=1.0)


def _sgns(walk_fn, ctx: TrainContext, n_vocab: int, n_extra: int,
          unigram: np.ndarray) -> TrainResult:
    import torch
    import torch.nn.functional as F

    p = {**DEFAULTS, **dict(ctx.params)}
    rng = np.random.default_rng(ctx.seed)
    torch.manual_seed(ctx.seed)
    device = ctx.device
    n_nodes = n_vocab + n_extra
    d = int(p["d_model"])

    emb = torch.nn.Embedding(n_nodes, d, sparse=True).to(device)
    out = torch.nn.Embedding(n_nodes, d, sparse=True).to(device)
    sc = float(p["init"]) / d ** 0.5
    torch.nn.init.uniform_(emb.weight, -sc, sc)
    torch.nn.init.uniform_(out.weight, -sc, sc)
    opt = torch.optim.SparseAdam(
        list(emb.parameters()) + list(out.parameters()), lr=float(p["lr"]))

    # Unigram^0.75 — the word2vec noise distribution. Uniform sampling would
    # make a rare compound as likely a negative as salt, which under-penalises
    # exactly the hubs that need separating.
    freq = np.ones(n_nodes, np.float64)
    freq[:n_vocab] += unigram
    noise = torch.tensor((freq ** 0.75) / (freq ** 0.75).sum(),
                         dtype=torch.float, device=device)

    C, K, B = int(p["context_size"]), int(p["negative_samples"]), int(p["batch_size"])
    history, t0 = [], time.time()
    for ep in range(int(p["epochs"])):
        rw = walk_fn(rng)
        # Windows are addressed, never materialised. Concatenating every sliding
        # window builds a ~600 MB int64 copy per epoch; instead permute
        # (walk, offset) ids and gather each batch straight out of `rw`.
        nw = rw.shape[1] - C + 1
        n_win = rw.shape[0] * nw
        perm = rng.permutation(n_win)
        off = np.arange(C)

        tot, nb = 0.0, 0
        for i in range(0, n_win, B):
            sel = perm[i:i + B]
            win = rw[(sel // nw)[:, None], (sel % nw)[:, None] + off]
            b = torch.from_numpy(win.astype(np.int64)).to(device)
            m = b.shape[0]
            centre = emb(b[:, 0])
            pos = torch.bmm(out(b[:, 1:]), centre.unsqueeze(2)).squeeze(2).reshape(-1)
            neg_ix = torch.multinomial(noise, m * K, replacement=True).view(m, K)
            neg = torch.bmm(out(neg_ix), centre.unsqueeze(2)).squeeze(2).reshape(-1)
            # Weighting the negative term by C-1 gives the same expected
            # gradient as (C-1)*K distinct draws at a fraction of the memory,
            # and keeps the SGNS ratio at 1 positive : K negatives.
            loss = (F.binary_cross_entropy_with_logits(
                        pos, torch.ones_like(pos), reduction="sum")
                    + (C - 1) * F.binary_cross_entropy_with_logits(
                        neg, torch.zeros_like(neg), reduction="sum")) / pos.numel()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        history.append(tot / max(nb, 1))
        print(f"  epoch {ep + 1:>2}/{p['epochs']}  loss {history[-1]:.4f}  "
              f"walks {len(rw):,}  windows {n_win:,}  "
              f"{time.time() - t0:.0f}s", flush=True)

    W = emb.weight.detach().cpu().numpy()[:n_vocab]
    return TrainResult(embedding=W,
                       metadata={"loss_history": history, **p},
                       extra_arrays={"context_table":
                                     out.weight.detach().cpu().numpy()[:n_vocab]})


@register(name="sgns-cooc", family="sgns_walk", cost_hint="moderate",
          defaults=DEFAULTS, tags=("baseline", "recipe-context"),
          requires=("ii_graph_train",),
          description="SGNS over the ingredient-ingredient NPMI graph")
def train_cooc(ctx: TrainContext) -> TrainResult:
    ii = load_ii_graph(ctx.graph)
    p = {**DEFAULTS, **dict(ctx.params)}
    return _sgns(
        lambda rng: walks_cooc(ii, rng, int(p["walks_per_node"]),
                               int(p["walk_length"])),
        ctx, ii.n_vocab, 0, np.asarray(ii.unigram))


@register(name="sgns-chem", family="sgns_walk", cost_hint="moderate",
          defaults=DEFAULTS, tags=("chemistry",),
          requires=("ii_graph_train", "chem_graph"),
          description="SGNS over typed ingredient->compound->ingredient metapaths")
def train_chem(ctx: TrainContext) -> TrainResult:
    ii, chem = load_ii_graph(ctx.graph), load_chem_graph()
    p = {**DEFAULTS, **dict(ctx.params)}
    return _sgns(
        lambda rng: walks_chem(ii, chem, rng, int(p["walks_per_node"]),
                               int(p["walk_length"])),
        ctx, ii.n_vocab, chem.n_compounds, np.asarray(ii.unigram))


@register(name="sgns-core", family="sgns_walk", cost_hint="moderate",
          defaults={**DEFAULTS, "ii_repeat": 10.0},
          tags=("hybrid", "chemistry", "recipe-context"),
          requires=("ii_graph_train", "chem_graph"),
          description="SGNS over the union graph; ii_repeat sets the mixing")
def train_core(ctx: TrainContext) -> TrainResult:
    ii, chem = load_ii_graph(ctx.graph), load_chem_graph()
    p = {**DEFAULTS, "ii_repeat": 10.0, **dict(ctx.params)}
    return _sgns(
        lambda rng: walks_core(ii, chem, rng, int(p["walks_per_node"]),
                               int(p["walk_length"]), float(p["ii_repeat"])),
        ctx, ii.n_vocab, chem.n_compounds, np.asarray(ii.unigram))
