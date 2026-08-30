"""item2vec — skip-gram with negative sampling over recipe baskets.

Identical objective to the random-walk family, with the graph removed. A recipe
*is* the context window: every ingredient in it is a positive context for every
other, with no window ordering, because a recipe has no ordering to respect.

The comparison this enables is precise. ``sgns-cooc`` walks an NPMI-weighted
graph, so it sees pairs re-weighted by an association measure that has already
divided popularity out. ``item2vec`` sees the empirical basket distribution
directly. Any difference between them is attributable to that re-weighting and
to nothing else, which is the cleanest available test of whether the NPMI
transform earns its place in the pipeline.
"""
from __future__ import annotations

import time

import numpy as np

from ingredient_model.data import load_recipes
from ingredient_model.data.recipes import RECIPE_IDS
from ingredient_model.registry import register
from ingredient_model.spec import TrainContext, TrainResult

DEFAULTS = dict(d_model=300, epochs=5, lr=0.0025, batch_size=32768,
                negative_samples=5, init=1.0, max_recipes=0,
                subsample=1e-4)


def _pair_stream(corpus, rng, subsample: float, unigram: np.ndarray):
    """All within-recipe ingredient pairs, as one shuffled array per epoch.

    Pairs are generated per length bucket by ``triu_indices``, which keeps the
    whole thing in NumPy. With frequent-word subsampling applied first, the pair
    count stays bounded even though the raw corpus would produce billions.
    """
    lens = corpus.sizes
    keep_p = np.ones(len(unigram))
    if subsample > 0:
        f = unigram / max(unigram.sum(), 1.0)
        # word2vec's subsampling rule: discard frequent tokens with probability
        # rising in their frequency. Salt is in a third of all recipes and
        # contributes almost no information about any particular pairing.
        with np.errstate(divide="ignore", invalid="ignore"):
            keep_p = np.minimum(1.0, (np.sqrt(f / subsample) + 1) * (subsample / f))
        keep_p = np.nan_to_num(keep_p, nan=1.0, posinf=1.0)

    out = []
    for k in range(2, int(lens.max()) + 1):
        rows = np.nonzero(lens == k)[0]
        if not len(rows):
            continue
        iu, ju = np.triu_indices(k, 1)
        step = max(int(4_000_000 / max(len(iu), 1)), 1)
        for s in range(0, len(rows), step):
            r = rows[s:s + step]
            ids = corpus.flat[corpus.offsets[r][:, None] + np.arange(k)].astype(np.int64)
            if subsample > 0:
                ids = np.where(rng.random(ids.shape) < keep_p[ids], ids, -1)
            a, b = ids[:, iu].ravel(), ids[:, ju].ravel()
            ok = (a >= 0) & (b >= 0)
            out.append(np.stack([a[ok], b[ok]], 1).astype(np.int32))
    pairs = np.concatenate(out)
    # `triu_indices` only ever emits i<j, and recipes are stored in ascending
    # id order, so column 0 is always the lower id. Feeding that straight into
    # an asymmetric skip-gram makes centre-exposure a function of the
    # vocabulary index: measured on the unfixed build, the alphabetically-first
    # decile was the centre 96% of the time and the last decile 4%
    # (corr(centre-share, id) = -0.980). The input matrix for late-alphabet
    # ingredients was therefore barely trained, its nearest neighbours were
    # alphabetical rather than semantic, and M6 recall collapsed to 0.0066.
    # Randomising the direction restores ~0.50 exposure for every ingredient at
    # no memory cost; over many epochs it is equivalent to emitting both
    # directions, which would double an already-large array.
    swap = rng.random(len(pairs)) < 0.5
    pairs[swap] = pairs[swap][:, ::-1]
    rng.shuffle(pairs)
    return pairs


@register(name="item2vec", family="recipe_basket", cost_hint="heavy",
          defaults=DEFAULTS, tags=("recipes", "sgns", "no-graph"),
          requires=("recipes",),
          description="SGNS over recipe baskets, with no graph intermediary")
def train_item2vec(ctx: TrainContext) -> TrainResult:
    import torch
    import torch.nn.functional as F

    p = {**DEFAULTS, **dict(ctx.params)}
    corpus = load_recipes(ctx.corpus or RECIPE_IDS)
    max_r = int(p["max_recipes"])
    if max_r and corpus.n_recipes > max_r:
        rng0 = np.random.default_rng(ctx.seed)
        corpus = corpus.select(
            np.sort(rng0.choice(corpus.n_recipes, max_r, replace=False)))

    rng = np.random.default_rng(ctx.seed)
    torch.manual_seed(ctx.seed)
    n, d, device = corpus.n_vocab, int(p["d_model"]), ctx.device
    unigram = corpus.unigram()

    emb = torch.nn.Embedding(n, d, sparse=True).to(device)
    out = torch.nn.Embedding(n, d, sparse=True).to(device)
    sc = float(p["init"]) / d ** 0.5
    torch.nn.init.uniform_(emb.weight, -sc, sc)
    torch.nn.init.uniform_(out.weight, -sc, sc)
    opt = torch.optim.SparseAdam(
        list(emb.parameters()) + list(out.parameters()), lr=float(p["lr"]))

    noise_np = (unigram + 1.0) ** 0.75
    noise = torch.tensor(noise_np / noise_np.sum(), dtype=torch.float, device=device)
    K, B = int(p["negative_samples"]), int(p["batch_size"])
    history, t0 = [], time.time()

    for ep in range(int(p["epochs"])):
        pairs = _pair_stream(corpus, rng, float(p["subsample"]), unigram)
        tot, nb = 0.0, 0
        for i in range(0, len(pairs), B):
            blk = torch.from_numpy(pairs[i:i + B].astype(np.int64)).to(device)
            m = blk.shape[0]
            centre = emb(blk[:, 0])
            pos = (out(blk[:, 1]) * centre).sum(1)
            neg_ix = torch.multinomial(noise, m * K, replacement=True).view(m, K)
            neg = torch.bmm(out(neg_ix), centre.unsqueeze(2)).squeeze(2).reshape(-1)
            loss = (F.binary_cross_entropy_with_logits(
                        pos, torch.ones_like(pos), reduction="sum")
                    + F.binary_cross_entropy_with_logits(
                        neg, torch.zeros_like(neg), reduction="sum")) / m
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        history.append(tot / max(nb, 1))
        print(f"  epoch {ep + 1:>2}/{p['epochs']}  loss {history[-1]:.4f}  "
              f"pairs {len(pairs):,}  {time.time() - t0:.0f}s", flush=True)

    return TrainResult(
        embedding=emb.weight.detach().cpu().numpy(),
        metadata={"loss_history": history, "n_recipes": corpus.n_recipes, **p})
