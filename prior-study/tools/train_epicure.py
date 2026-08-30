#!/usr/bin/env python3
"""Train the three Epicure-equivalent embeddings: Cooc, Chem and Core.

All three share architecture and hyperparameters and differ only in the random
walk schema, which is the whole point of the paper -- it places each model at a
different spot on the chemistry-vs-recipe-context spectrum, so any difference in
the resulting geometry is attributable to the walk and nothing else.

    Cooc  walks the NPMI ingredient-ingredient graph only.
    Chem  walks typed ingredient-compound metapaths only (I -> C -> I), with
          each walk pinned to one of the 15 compound categories so the metapath
          is genuinely typed rather than an untyped bipartite stroll.
    Core  walks the union, with ingredient-ingredient edges up-weighted by
          `ii_repeat` so co-occurrence is injected at controlled mixing.

Hyperparameters come from data/raw/epicure-core/config.json and are not tuned:
d_model 300, walks_per_node 100, walk_length 50, context_size 7, 5 negatives,
batch 32768, lr 0.0025, 20 epochs, SGNS.

Objective follows the standard metapath2vec/PyG formulation: each walk is cut
into sliding windows of `context_size`, the first node of a window is the
centre and the rest are positive contexts. Negatives come from the unigram^0.75
noise distribution, drawn per centre, and centre/context live in separate tables
(standard SGNS) -- both details turn out to be load-bearing for the geometry.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
# Azure ML mounts inputs/outputs at arbitrary paths, so both ends are overridable.
DERIVED = Path(os.environ.get("EPICURE_DERIVED", ROOT / "data" / "derived"))
OUT_DIR = Path(os.environ.get("EPICURE_OUT", ROOT / "models"))
# Held-out study runs train on ii_graph_train.npz so link prediction measures
# generalisation rather than memorisation; default keeps legacy behaviour.
II_GRAPH = os.environ.get("EPICURE_II_GRAPH", "ii_graph.npz")

CFG = dict(d_model=300, walks_per_node=100, walk_length=50, context_size=7,
           negative_samples=5, batch_size=32768, lr=0.0025, epochs=20, init=1.0)


# --------------------------------------------------------------------- graphs
class CSR:
    """Weighted adjacency in CSR form with a global cumulative-weight array.

    The cumulative array is what makes weighted walking vectorisable: because
    each node's edges are contiguous and the cumsum is globally monotonic, one
    `searchsorted` over all edges samples a neighbour for every walker at once.
    Per-node loops would be ~1000x slower at this walk volume.
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
        """One weighted hop for every walker. Dead ends stay put (caller masks)."""
        lo, hi = self.indptr[cur], self.indptr[cur + 1]
        clo, chi = self.cum[lo], self.cum[hi]
        target = clo + rng.random(len(cur)) * (chi - clo)
        j = np.searchsorted(self.cum, target, side="right") - 1
        j = np.clip(j, lo, np.maximum(hi - 1, lo))
        nxt = self.indices[j]
        dead = self.deg[cur] == 0
        nxt[dead] = cur[dead]
        return nxt


def load_ii(symmetric: bool = True):
    z = np.load(DERIVED / II_GRAPH, allow_pickle=True)
    s, d, w = z["src"], z["dst"], z["npmi"].astype(np.float64)
    # NPMI can be negative; shift into a positive sampling weight while keeping
    # the ordering, so a weakly-associated pair is rare rather than impossible.
    w = w - w.min() + 1e-3
    if symmetric:
        s, d, w = (np.concatenate([s, d]), np.concatenate([d, s]),
                   np.concatenate([w, w]))
    return s, d, w, z["itos"]


def load_ic():
    z = np.load(DERIVED / "flavor_graph.npz", allow_pickle=True)
    return z["src"], z["dst"], z["ctype"], z["itos"]


# ---------------------------------------------------------------------- walks
def walks_cooc(n_vocab, rng, wpn, wlen):
    s, d, w, _ = load_ii()
    g = CSR(n_vocab, s, d, w)
    start = np.repeat(np.arange(n_vocab)[g.deg > 0], wpn)
    rng.shuffle(start)
    out = np.empty((len(start), wlen + 1), np.int32)
    out[:, 0] = start
    cur = start.copy()
    for t in range(wlen):
        cur = g.step(cur, rng)
        out[:, t + 1] = cur
    return out


def walks_chem(n_vocab, rng, wpn, wlen):
    """Typed I -> C -> I metapaths, one compound category per walk.

    Pinning the category is what makes the metapath typed: a walk confined to
    organosulfur compounds connects garlic/onion/leek, while one confined to
    lactones connects dairy and stone fruit. An untyped bipartite walk would
    average those channels together and lose exactly the structure Chem exists
    to capture. Compound nodes are offset by n_vocab so ingredients and
    compounds share one embedding table but never collide.
    """
    s, d, ctype, _ = load_ic()
    walks = []
    n_cat = int(ctype.max()) + 1
    per_cat = max(wpn // n_cat, 1)
    for c in range(n_cat):
        m = ctype[d] == c
        if m.sum() < 2:
            continue
        cs, cd = s[m].astype(np.int64), d[m].astype(np.int64) + n_vocab
        fwd = CSR(n_vocab + len(ctype), cs, cd, np.ones(m.sum()))
        rev = CSR(n_vocab + len(ctype), cd, cs, np.ones(m.sum()))
        live = np.arange(n_vocab)[fwd.deg[:n_vocab] > 0]
        if not len(live):
            continue
        start = np.repeat(live, per_cat)
        rng.shuffle(start)
        out = np.empty((len(start), wlen + 1), np.int32)
        out[:, 0] = start
        cur = start.copy()
        for t in range(wlen):
            cur = (fwd if t % 2 == 0 else rev).step(cur, rng)
            out[:, t + 1] = cur
        walks.append(out)
    return np.concatenate(walks)


def walks_core(n_vocab, rng, wpn, wlen, ii_repeat=10):
    """Union walk: I-C edges plus I-I edges up-weighted by `ii_repeat`.

    This is the "injected ingredient-ingredient walks at controlled mixing" the
    config describes. Both edge families live in one graph, so a single walk can
    hop garlic -> allicin -> onion (chemistry) and then onion -> tomato (recipe
    context), which is precisely the blend Core is meant to sit at.
    """
    s2, d2, w2, _ = load_ii()
    s1, d1, ctype, _ = load_ic()
    n_all = n_vocab + len(ctype)
    ic_s = np.concatenate([s1.astype(np.int64), d1.astype(np.int64) + n_vocab])
    ic_d = np.concatenate([d1.astype(np.int64) + n_vocab, s1.astype(np.int64)])
    ic_w = np.ones(len(ic_s))
    w2 = w2 / w2.mean() * ii_repeat
    g = CSR(n_all, np.concatenate([ic_s, s2]), np.concatenate([ic_d, d2]),
            np.concatenate([ic_w, w2]))
    live = np.arange(n_vocab)[g.deg[:n_vocab] > 0]
    start = np.repeat(live, wpn)
    rng.shuffle(start)
    out = np.empty((len(start), wlen + 1), np.int32)
    out[:, 0] = start
    cur = start.copy()
    for t in range(wlen):
        cur = g.step(cur, rng)
        out[:, t + 1] = cur
    return out


# -------------------------------------------------------------------- training
def train(variant: str, device: str, cfg: dict, out_dir: Path, seed: int = 0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    itos = np.load(DERIVED / II_GRAPH, allow_pickle=True)["itos"]
    n_vocab = len(itos)
    n_comp = len(np.load(DERIVED / "flavor_graph.npz", allow_pickle=True)["ctype"])
    n_nodes = n_vocab + n_comp

    # Two tables, as in standard SGNS. A single shared table makes centre.context
    # a self-similarity whose trivial minimiser is "point every vector the same
    # way", which collapses the participation ratio to ~1 while still producing
    # plausible nearest neighbours -- so it survives a spot check.
    emb = torch.nn.Embedding(n_nodes, cfg["d_model"], sparse=True).to(device)
    out = torch.nn.Embedding(n_nodes, cfg["d_model"], sparse=True).to(device)
    # Init scale is load-bearing. word2vec's U(-0.5/d, 0.5/d) with a zero output
    # table sits at a saddle: gradient descent then grows singular directions one
    # at a time, so the effective rank climbs only logarithmically and 20 epochs
    # leaves the space essentially rank-3. Scaling init up breaks the saddle and
    # lets all 300 directions grow together.
    sc = cfg["init"] / cfg["d_model"] ** 0.5
    torch.nn.init.uniform_(emb.weight, -sc, sc)
    torch.nn.init.uniform_(out.weight, -sc, sc)

    # SparseAdam, not Adam. A dense optimiser over an embedding table applies a
    # step to *every* row on *every* iteration, because the momentum buffer of a
    # row that received no gradient is still non-zero from earlier batches. Over
    # thousands of steps that drags all 3,578 rows along one accumulated common
    # direction -- which is precisely the degenerate geometry we measured (top PC
    # 76% of variance, avg pairwise cosine +0.73). SparseAdam only touches rows
    # that actually appear in the batch.
    opt = torch.optim.SparseAdam(list(emb.parameters()) + list(out.parameters()),
                                 lr=cfg["lr"])

    # Negatives are drawn from the unigram distribution raised to 3/4, the
    # word2vec noise distribution: uniform sampling over 3,578 nodes would make
    # a rare compound as likely a negative as salt, which under-penalises the
    # hubs that actually need separating.
    freq = np.ones(n_nodes, dtype=np.float64)
    ii = np.load(DERIVED / II_GRAPH, allow_pickle=True)
    freq[:n_vocab] += ii["uni"].astype(np.float64)
    noise = torch.tensor((freq ** 0.75) / (freq ** 0.75).sum(),
                         dtype=torch.float, device=device)

    C, K, B = cfg["context_size"], cfg["negative_samples"], cfg["batch_size"]
    gen = {"cooc": walks_cooc, "chem": walks_chem, "core": walks_core}[variant.split("-")[0]]
    # H1 sweeps the ingredient-ingredient mixing weight. At 0 `core` degenerates
    # to Chem's pure I->C->I schema; at large values it approaches Cooc, so one
    # knob traces the whole collapse curve.
    if "ii_repeat" in cfg and variant.split("-")[0] == "core":
        gen = functools.partial(gen, ii_repeat=cfg["ii_repeat"])
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        rw = gen(n_vocab, rng, cfg["walks_per_node"], cfg["walk_length"])
        # Windows are addressed, never materialised. Concatenating every
        # sliding window builds a ~600MB int64 copy per epoch, which on an
        # 18GB shared machine pushes the process into swap and stalls it at
        # ~1% CPU. Instead permute (walk, offset) ids and gather each batch
        # straight out of `rw`, which keeps the epoch footprint at ~60MB.
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
            centre = emb(b[:, 0])                       # (m, d)
            ctx = out(b[:, 1:])                         # (m, C-1, d)
            p = torch.bmm(ctx, centre.unsqueeze(2)).squeeze(2).reshape(-1)
            # Negatives must be drawn *per centre*. Sharing one draw across the
            # whole batch repels all m centres from the same handful of points,
            # which manufactures exactly the common direction we are trying to
            # remove. Drawing (m, K) and weighting the term by C-1 has the same
            # expected gradient as (m, (C-1)*K) distinct draws at a fraction of
            # the memory, and keeps the SGNS ratio at 1 positive : K negatives.
            neg = out(torch.multinomial(noise, m * K, replacement=True).view(m, K))
            n = torch.bmm(neg, centre.unsqueeze(2)).squeeze(2).reshape(-1)
            loss = (F.binary_cross_entropy_with_logits(
                        p, torch.ones_like(p), reduction="sum") +
                    (C - 1) * F.binary_cross_entropy_with_logits(
                        n, torch.zeros_like(n), reduction="sum")) / p.numel()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        print(f"  [{variant}] epoch {ep+1:>2}/{cfg['epochs']} "
              f"loss {tot/max(nb,1):.4f}  walks {len(rw):,}  "
              f"windows {n_win:,}  {time.time()-t0:.0f}s", flush=True)

    W = emb.weight.detach().cpu().numpy()[:n_vocab]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"epicure_{variant}.npy", W)
    (out_dir / f"epicure_{variant}.json").write_text(json.dumps(
        {"variant": variant, "vocab_size": int(n_vocab), **cfg}, indent=2))
    print(f"  saved {out_dir}/epicure_{variant}.npy {W.shape}", flush=True)
    return W


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="*", default=["cooc", "chem", "core"])
    ap.add_argument("--epochs", type=int, default=CFG["epochs"])
    ap.add_argument("--lr", type=float, default=CFG["lr"])
    ap.add_argument("--batch", type=int, default=CFG["batch_size"])
    ap.add_argument("--init", type=float, default=CFG["init"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0,
                    help="replicate seed; differences smaller than seed "
                         "variance are not real")
    ap.add_argument("--ii-repeat", type=float, default=None,
                    help="core only: I-I edge weight multiplier (H1 sweep)")
    ap.add_argument("--d-model", type=int, default=CFG["d_model"],
                    help="embedding width; the app ships this table, so the "
                         "smallest width that holds accuracy is a product "
                         "decision, not just a modelling one")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available()
                    else "cpu")
    a = ap.parse_args()
    cfg = {**CFG, "epochs": a.epochs, "lr": a.lr, "batch_size": a.batch,
           "init": a.init, "d_model": a.d_model}
    if a.ii_repeat is not None:
        cfg["ii_repeat"] = a.ii_repeat
    print(f"device={a.device} graph={II_GRAPH} cfg={cfg}\n")
    for v in a.variants:
        train(v + a.tag, a.device, cfg, OUT_DIR, seed=a.seed)


if __name__ == "__main__":
    main()
