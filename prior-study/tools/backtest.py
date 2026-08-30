#!/usr/bin/env python3
"""Recipe-level backtest: can the model finish a real recipe?

M2-M5 all score the *graph* -- held-out edges, substitution triplets. That is
the right way to test the embedding, but it is not what the product does. The
app is handed a partial recipe and asked what else belongs, so this evaluates
exactly that: hide one ingredient from a real held-out recipe, rank all 1,790
candidates from the ones that remain, and see where the hidden ingredient lands.

Leakage matters more here than anywhere else in the study. The edge-level split
used elsewhere still let every test recipe contribute to the co-occurrence
counts, so scoring recipe completion against it would be optimistic. `build`
therefore holds out whole *recipes* and rebuilds the graph without them, at the
same fixed NPMI threshold and min_count the main graph uses -- the only
difference between this graph and the study's is which recipes were counted.

The popularity baseline is not a formality. Recommending onion, salt and butter
to everyone scores respectably on this task, and a model that has learned only
frequency will look competent until it is compared against one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_ii_graph import cooccurrence

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
IDS = DERIVED / "recipe_ids.npz"
SEED = 20260806
N_TEST = 100_000
THRESHOLD = -0.15   # the value the main graph settled on
MIN_COUNT = 2
LAMBDAS = (0.02, 0.05, 0.1, 0.2)
BETAS = (0.1, 0.5, 2.0)


def build(args) -> None:
    z = np.load(IDS, allow_pickle=True)
    flat, offs, itos = z["flat"].astype(np.int64), z["offsets"], z["itos"]
    n_vocab, n_rec = len(itos), len(offs) - 1
    lens = np.diff(offs)

    rng = np.random.default_rng(SEED)
    # Only recipes with >=3 ingredients are usable: hiding one must still leave
    # at least two to query from.
    eligible = np.flatnonzero(lens >= 3)
    test = np.sort(rng.choice(eligible, min(args.n_test, len(eligible)),
                              replace=False))
    print(f"{n_rec:,} recipes, {len(eligible):,} eligible, "
          f"{len(test):,} held out")

    keep = np.ones(n_rec, bool)
    keep[test] = False
    kf, ko = [], [0]
    for r in np.flatnonzero(keep):
        kf.append(flat[offs[r]:offs[r + 1]])
        ko.append(ko[-1] + int(lens[r]))
    kflat = np.concatenate(kf)
    koffs = np.array(ko, np.int64)
    print(f"rebuilding graph from {keep.sum():,} recipes")

    src, dst, cnt, uni = cooccurrence(kflat, koffs, n_vocab)
    cnt = cnt.astype(np.float64)
    m = cnt >= MIN_COUNT
    src, dst, cnt = src[m], dst[m], cnt[m]
    n_train_rec = int(keep.sum())
    pij = cnt / n_train_rec
    npmi = (np.log(pij / ((uni[src] / n_train_rec) * (uni[dst] / n_train_rec)))
            / -np.log(pij))
    sel = npmi >= THRESHOLD
    src, dst, cnt, npmi = src[sel], dst[sel], cnt[sel], npmi[sel]
    deg = np.bincount(np.concatenate([src, dst]), minlength=n_vocab)
    print(f"edges {len(src):,} | isolated {int((deg == 0).sum()):,}")

    np.savez_compressed(DERIVED / args.out, src=src.astype(np.int32),
                        dst=dst.astype(np.int32), npmi=npmi.astype(np.float32),
                        count=cnt.astype(np.int32), uni=uni,
                        n_recipes=n_train_rec, threshold=THRESHOLD,
                        min_count=MIN_COUNT, itos=itos)
    np.savez_compressed(DERIVED / "recipe_backtest.npz", test=test,
                        uni_train=uni)
    print(f"wrote {args.out} and recipe_backtest.npz")


def rank_stats(scores: np.ndarray, target: np.ndarray,
               banned: np.ndarray) -> np.ndarray:
    """Rank of each target among candidates, excluding what the user already has."""
    scores = scores.copy()
    scores[banned] = -np.inf
    tgt = scores[np.arange(len(target)), target]
    return (scores > tgt[:, None]).sum(1) + 1


def evaluate(args) -> None:
    z = np.load(IDS, allow_pickle=True)
    flat, offs, itos = z["flat"].astype(np.int64), z["offsets"], z["itos"]
    n_vocab = len(itos)
    bt = np.load(DERIVED / "recipe_backtest.npz")
    test, uni = bt["test"], bt["uni_train"].astype(np.float64)

    W = np.load(args.emb)[:n_vocab]
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)

    # Dense NPMI adjacency for the graph baseline. If a plain lookup in the
    # graph the embedding was trained on wins, the embedding is not earning its
    # place in the product for this task.
    g = np.load(DERIVED / args.graph, allow_pickle=True)
    A = np.zeros((n_vocab, n_vocab), np.float32)
    gs, gd, gw = g["src"], g["dst"], g["npmi"].astype(np.float32)
    A[gs, gd] = gw
    A[gd, gs] = gw

    rng = np.random.default_rng(SEED)
    sims = W @ W.T
    logfreq = np.log1p(uni)
    zfreq = (logfreq - logfreq.mean()) / logfreq.std()
    nov: dict[str, list[np.ndarray]] = {}
    allheld: list[np.ndarray] = []
    acc: dict[str, list[np.ndarray]] = {k: [] for k in
                                        ("embed_mean", "embed_max",
                                         "graph_npmi", "popularity")}
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

        # Query vector is the mean of what the user already has, which is what
        # the app can actually compute.
        q = (ctx @ W) / ctx.sum(1, keepdims=True)
        q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-9
        acc["embed_mean"].append(rank_stats(q @ W.T, held, ctx))
        # Max similarity to any single held ingredient: rewards a candidate
        # that strongly matches one item rather than the recipe's average.
        acc["embed_max"].append(rank_stats(
            np.stack([sims[c].max(0) for c in ctx]), held, ctx))
        acc["graph_npmi"].append(rank_stats(ctx @ A, held, ctx))
        # Popularity ignores the query entirely -- the control.
        acc["popularity"].append(rank_stats(np.tile(uni, (len(rows), 1)),
                                            held, ctx))
        # Recall rewards predicting salt, but an app that only suggests salt is
        # useless. Subtracting a multiple of the popularity prior traces how
        # much recall a more adventurous ranking actually costs.
        base = q @ W.T
        for lam in LAMBDAS:
            s = base - lam * zfreq[None, :]
            key = f"embed_lam{lam}"
            acc.setdefault(key, []).append(rank_stats(s, held, ctx))
            top = np.argpartition(np.where(ctx, -np.inf, s), -10, 1)[:, -10:]
            nov.setdefault(key, []).append(logfreq[top].mean(1))
        top = np.argpartition(np.where(ctx, -np.inf, base), -10, 1)[:, -10:]
        nov.setdefault("embed_mean", []).append(logfreq[top].mean(1))

        # The shippable ranker. The graph is authoritative where it has
        # support. beta controls how far the embedding may override real
        # evidence: at 0.1 it only breaks ties, at 2.0 it dominates.
        # NOTE: this comment used to claim the embedding's job was to order
        # candidates the graph is silent about. rank_sweep.py falsified that --
        # the graph is silent in 26 of 100,000 cases (0.026%), and on those the
        # embedding also scores 0.000. beta is retained because it improves
        # aggregate recall (0.3632 -> 0.3909) at no measurable non-staple cost,
        # which is a narrower claim. See docs/PREREGISTRATION.md A7.
        # A8 re-ran this against an embedding trained without the held-out
        # recipes (cooc-recipeholdout): beta=0.5 moved by +0.0002, so the number
        # above carries no embedding-side leakage. beta=2.0 scores higher in
        # aggregate (0.4249) but loses 0.017 on the non-staple slice, which is
        # the slice the product exists for -- do not raise beta on the
        # aggregate number alone.
        gs = ctx @ A
        for beta in BETAS:
            acc.setdefault(f"hybrid_b{beta}", []).append(
                rank_stats(gs + beta * base, held, ctx))
        allheld.append(held)

    def summarise(r):
        return {"recall@1": round(float((r <= 1).mean()), 4),
                "recall@10": round(float((r <= 10).mean()), 4),
                "recall@50": round(float((r <= 50).mean()), 4),
                "mrr": round(float((1 / r).mean()), 4),
                "median_rank": float(np.median(r))}

    out = {"model": Path(args.emb).stem, "n_test": int(len(test)),
           "graph": args.graph,
           "scorers": {k: summarise(np.concatenate(v)) for k, v in acc.items()},
           "top10_mean_log_freq": {k: round(float(np.concatenate(v).mean()), 3)
                                   for k, v in nov.items()}}

    # The staples are in almost every recipe, so predicting them inflates every
    # scorer including the control. Restricting to recipes whose missing
    # ingredient is *not* a staple is the metric the product lives or dies on.
    held_all = np.concatenate(allheld)
    staples = np.argsort(uni)[-args.n_staples:]
    interesting = ~np.isin(held_all, staples)
    out["non_staple_slice"] = {
        "n": int(interesting.sum()),
        "frac_of_test": round(float(interesting.mean()), 3),
        "scorers": {k: summarise(np.concatenate(v)[interesting])
                    for k, v in acc.items()
                    if not k.startswith(("embed_lam", "embed_max"))}}
    print(json.dumps(out, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--n-test", type=int, default=N_TEST)
    b.add_argument("--out", default="ii_graph_recipe_train.npz")
    b.set_defaults(fn=build)
    e = sub.add_parser("eval")
    e.add_argument("--emb", required=True)
    e.add_argument("--graph", default="ii_graph_recipe_train.npz")
    e.add_argument("--n-staples", type=int, default=50)
    e.add_argument("--out-json", default="")
    e.set_defaults(fn=evaluate)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
