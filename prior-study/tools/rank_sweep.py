#!/usr/bin/env python3
"""Rank sweep: does compressing the association matrix destroy the tail first?

Levy & Goldberg (2014) showed SGNS implicitly factorises a shifted PMI matrix,
so the embedding and the NPMI graph are two estimates of the *same* quantity:
the graph stores it exactly but sparsely, the embedding approximates it at
rank d but densely. Eckart-Young says a rank-d truncation keeps the top-d
singular directions, and in a PMI-like matrix those are dominated by
high-frequency structure. Staples live in the head of the spectrum; everything
interesting lives in the tail that truncation throws away.

That predicts something specific and falsifiable, stated before running this:

  P1. Non-staple recall@10 rises monotonically with rank r.
  P2. Aggregate recall@10 saturates at a much lower rank than non-staple does,
      because the head is captured by the first few directions.
  P3. On pairs the graph has never observed, the embedding beats the graph at
      every rank -- that is the only place it can earn its beta.

P1/P2 failing would mean our explanation for the A5 result is wrong, even
though A5 itself would stand. P3 failing would mean beta should be 0.

Truncating a trained d=300 embedding is not identical to training at d=r --
it isolates rank while holding training fixed, which is the point, but it
cannot capture how a lower-d model would reallocate capacity during training.
Read it as a statement about representational capacity, not training dynamics.

Usage (cwd must be tools/):
    python rank_sweep.py --emb ../models/epicure_cooc.npy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from backtest import DERIVED, IDS, SEED, rank_stats

RANKS = (2, 4, 8, 16, 32, 64, 128, 300)


def load_emb(path: Path, n: int) -> np.ndarray:
    """Accept either a .npy or the raw .f32 blob the app ships."""
    p = Path(path)
    if p.suffix == ".npy":
        W = np.load(p)[:n]
    else:
        raw = np.frombuffer(p.read_bytes(), "<f4")
        assert raw.size % n == 0, f"{p} not divisible by {n} rows"
        W = raw.reshape(n, raw.size // n).astype(np.float32)
    return W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)


def truncate(W: np.ndarray, r: int) -> np.ndarray:
    """Best rank-r approximation (Eckart-Young), rows re-normalised for cosine."""
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    Wr = (U[:, :r] * S[:r]) @ Vt[:r]
    return Wr / (np.linalg.norm(Wr, axis=1, keepdims=True) + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="../data/derived/app_bundle/cooc.emb.f32")
    ap.add_argument("--graph", default="ii_graph_recipe_train.npz")
    ap.add_argument("--n-staples", type=int, default=50)
    ap.add_argument("--out-json", default="../results/rank_sweep.json")
    args = ap.parse_args()

    z = np.load(IDS, allow_pickle=True)
    flat, offs, itos = z["flat"].astype(np.int64), z["offsets"], z["itos"]
    n_vocab = len(itos)
    bt = np.load(DERIVED / "recipe_backtest.npz")
    test, uni = bt["test"], bt["uni_train"].astype(np.float64)

    W0 = load_emb(Path(args.emb), n_vocab)
    ranks = tuple(r for r in RANKS if r <= W0.shape[1])

    g = np.load(DERIVED / args.graph, allow_pickle=True)
    A = np.zeros((n_vocab, n_vocab), np.float32)
    gsrc, gdst, gw = g["src"], g["dst"], g["npmi"].astype(np.float32)
    A[gsrc, gdst] = gw
    A[gdst, gsrc] = gw
    seen = A != 0

    U, S, Vt = np.linalg.svd(W0, full_matrices=False)
    energy = np.cumsum(S ** 2) / (S ** 2).sum()

    Ws = {r: truncate(W0, r) for r in ranks}

    # Replay the identical problems backtest.py scored: same seed, same order,
    # so the hidden ingredient is the same one in every recipe.
    rng = np.random.default_rng(SEED)
    held_all, graph_rank = [], []
    emb_rank: dict[int, list] = {r: [] for r in ranks}
    nocover = []
    B = 4096
    for s in range(0, len(test), B):
        rows = test[s:s + B]
        ctx = np.zeros((len(rows), n_vocab), bool)
        held = np.empty(len(rows), np.int64)
        for k, r in enumerate(rows):
            ing = flat[offs[r]:offs[r + 1]]
            h = rng.integers(len(ing))
            held[k] = ing[h]
            ctx[k, np.delete(ing, h)] = True

        gsc = ctx @ A
        graph_rank.append(rank_stats(gsc, held, ctx))
        # True where the graph has no edge at all between the hidden ingredient
        # and anything in the basket -- the graph is structurally silent here.
        nocover.append(~(seen[held] & ctx).any(1))

        n = ctx.sum(1, keepdims=True)
        for r, W in Ws.items():
            q = (ctx @ W) / n
            q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-9
            emb_rank[r].append(rank_stats(q @ W.T, held, ctx))
        held_all.append(held)

    held_all = np.concatenate(held_all)
    graph_rank = np.concatenate(graph_rank)
    nocover = np.concatenate(nocover)
    staples = np.argsort(uni)[-args.n_staples:]
    nonstaple = ~np.isin(held_all, staples)

    def r10(x, m=None):
        x = x if m is None else x[m]
        return round(float((x <= 10).mean()), 4)

    out = {
        "emb": Path(args.emb).stem,
        "n_test": int(len(held_all)),
        "slices": {"all": int(len(held_all)),
                   "non_staple": int(nonstaple.sum()),
                   "graph_uncovered": int(nocover.sum()),
                   "graph_uncovered_nonstaple": int((nocover & nonstaple).sum())},
        "spectrum": {f"rank_{r}": round(float(energy[r - 1]), 4)
                     for r in ranks},
        "graph_npmi": {"all": r10(graph_rank),
                       "non_staple": r10(graph_rank, nonstaple),
                       "graph_uncovered": r10(graph_rank, nocover)},
        "by_rank": {},
    }
    for r in ranks:
        e = np.concatenate(emb_rank[r])
        out["by_rank"][str(r)] = {
            "all": r10(e),
            "non_staple": r10(e, nonstaple),
            "graph_uncovered": r10(e, nocover),
            "variance_explained": round(float(energy[r - 1]), 4),
        }
    print(json.dumps(out, indent=2))
    Path(args.out_json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
