"""M6 — recipe completion.

M1-M5 score the *geometry*: held-out edges, substitution triplets. That is the
right way to test an embedding, but it is not what the product does. The product
is handed a partial recipe and asked what else belongs, so this evaluates
exactly that: hide one ingredient from a real held-out recipe, rank all 1,790
candidates from the ones that remain, and record where the hidden one lands.

Two properties make this the most trustworthy number in the workspace:

* **It is leak-free for every family.** The test recipes were never in any
  training input, so graph models and recipe models are on identical footing —
  unlike M4, which is only sound for one of them.
* **It has a real baseline.** Recommending onion, salt and butter to everyone
  scores respectably here. A model that has learned nothing but frequency looks
  competent until it is put beside that baseline, which is why the popularity
  control is computed alongside rather than optionally.

One thing this metric does *not* control for on its own, recorded because it
reordered the leaderboard once it was measured. Context is pooled by summing
unit vectors and candidates are ranked by cosine against that sum. Re-scoring
with the embedding mean removed changes the result substantially for some
families and not at all for others:

===============  ==========  ==========
family           raw @10     centred
===============  ==========  ==========
glove            0.2509      **0.4919**
item2vec-fixed   0.0195      0.1756
item2vec         0.0070      0.0645
sgc              0.0135      0.0539
ease             0.1653      0.1880
sgns-cooc        0.4020      0.3967
svd-ppmi         0.4462      0.4443
===============  ==========  ==========

Two things make this worth reporting rather than discarding. Centring never
meaningfully *hurts* — the two families that lose anything lose 0.005 and
0.002, inside sampling noise. And it changes the ranking: ``glove`` goes from
fourth to first, ahead of ``svd-ppmi``.

**The mechanism is not established.** Three plausible explanations were tested
and all three fail, so none of them is quoted here as the reason:

* *Translation gauge.* The obvious story is that cosine against a summed query
  is not translation-invariant, so an off-centre space is scored on an
  arbitrary choice. But translating a synthetic embedding by a large constant
  barely moves raw M6 at all (0.8395 -> 0.8415). Pinned in
  ``tests/test_completion_centring.py`` so the explanation cannot quietly
  return.
* *Popularity in the shared direction.* Correlation between alignment with the
  mean direction and log frequency is **+0.646 for ``sgns-cooc``, which gains
  nothing**, and **-0.009 for ``item2vec-fixed``, which gains +0.156**. The
  sign of the relationship is backwards from the hypothesis.
* *Centroid size alone.* ``glove`` (0.433) and ``sgns-cooc`` (0.414) have
  almost identical mean pairwise cosine and gain +0.241 and -0.005.

So ``M6_centred_*`` is reported *alongside* the raw number, never instead of
it. Read the raw column as "how this embedding behaves if served as-is" and the
centred column as "how it behaves after a standard, cost-free post-processing".
The gap between them is a reproducible property of a family that is not yet
explained, which is a reason to keep measuring it, not a reason to pick one.
"""
from __future__ import annotations

import numpy as np

from ..config import SEED
from .metrics import unit


def _rank_of_target(scores: np.ndarray, target: np.ndarray,
                    forbid: np.ndarray) -> np.ndarray:
    """Rank of each target among all candidates, with the visible context
    ingredients excluded — recommending an ingredient the recipe already lists
    is not a prediction.

    Ties are broken by *midrank* (the average of the positions the tied group
    occupies), not by counting only strictly-better candidates. This matters:
    the optimistic rule silently rewards degenerate models. A rank-1 collapsed
    embedding normalises to just two distinct rows, so every candidate takes
    one of two scores and ~900 of them tie at the recall@10 cut; under the
    optimistic rule that scored 0.211 — thirty-eight times chance, and better
    than several genuinely trained models — purely as an artefact.
    Representation collapse is a real failure mode of the transformer and
    over-smoothed graph families, so the metric must not pay for it.

    For a model with no ties this is arithmetically identical to the optimistic
    rule (``equal`` is 1, the target itself), so no tie-free result changes.
    """
    scores = np.array(scores, dtype=float, copy=True)
    scores[forbid] = -np.inf
    tgt = scores[np.arange(len(target)), target]
    greater = (scores > tgt[:, None]).sum(1)
    equal = (scores == tgt[:, None]).sum(1)  # includes the target itself
    return greater + 1.0 + (equal - 1) / 2.0


def recipe_completion(W: np.ndarray, corpus, *, n_test: int = 20_000,
                      seed: int = SEED, unigram: np.ndarray | None = None,
                      scorer=None) -> dict:
    """Hide one ingredient per recipe and rank all candidates from the rest.

    Context is pooled by summing the unit vectors of the visible ingredients.
    Summing rather than averaging is deliberate: the candidate ranking is
    invariant to the scale of the pooled vector, so the two are identical here,
    and summing avoids a division that would differ across recipe lengths for no
    effect.

    ``scorer`` is an optional native conditional model. When given, its ranking
    is reported alongside the embedding one rather than instead of it — the
    embedding number is what makes families comparable, the native number is
    what the model would actually serve.
    """
    rng = np.random.default_rng(seed)
    lens = corpus.sizes
    # A recipe needs 3 ingredients so that hiding one still leaves two to query.
    eligible = np.flatnonzero(lens >= 3)
    if not len(eligible):
        return {"M6_n": 0}
    pick = rng.choice(eligible, min(n_test, len(eligible)), replace=False)

    U = unit(W)
    n_vocab = W.shape[0]
    ranks, pop_ranks, native_ranks = [], [], []
    logf = None if unigram is None else np.log1p(unigram)

    # Bucket by length so context pooling is one matrix op per bucket.
    for k in np.unique(lens[pick]):
        rows = pick[lens[pick] == k]
        ids = corpus.flat[corpus.offsets[rows][:, None] + np.arange(k)].astype(np.int64)
        hide = rng.integers(0, k, len(rows))
        target = ids[np.arange(len(rows)), hide]
        mask = np.ones_like(ids, bool)
        mask[np.arange(len(rows)), hide] = False
        ctx = ids[mask].reshape(len(rows), k - 1)

        pooled = U[ctx].sum(1)
        forbid = np.zeros((len(rows), n_vocab), bool)
        forbid[np.arange(len(rows))[:, None], ctx] = True
        ranks.append(_rank_of_target(pooled @ U.T, target, forbid))
        if logf is not None:
            pop = np.repeat(logf[None], len(rows), 0)
            pop_ranks.append(_rank_of_target(pop, target, forbid))
        if scorer is not None:
            native_ranks.append(
                _rank_of_target(np.asarray(scorer(ctx), float), target, forbid))

    r = np.concatenate(ranks)
    out = {
        "M6_n": int(len(r)),
        "M6_recall_at_10": float((r <= 10).mean()),
        "M6_recall_at_50": float((r <= 50).mean()),
        "M6_mrr": float((1.0 / r).mean()),
        "M6_median_rank": float(np.median(r)),
    }
    if pop_ranks:
        p = np.concatenate(pop_ranks)
        out.update({
            "M6_popularity_recall_at_10": float((p <= 10).mean()),
            "M6_popularity_mrr": float((1.0 / p).mean()),
            "M6_lift_over_popularity":
                float((r <= 10).mean() - (p <= 10).mean()),
        })
    if native_ranks:
        nr = np.concatenate(native_ranks)
        out.update({
            "M6_native_recall_at_10": float((nr <= 10).mean()),
            "M6_native_recall_at_50": float((nr <= 50).mean()),
            "M6_native_mrr": float((1.0 / nr).mean()),
            "M6_native_median_rank": float(np.median(nr)),
        })
        if pop_ranks:
            out["M6_native_lift_over_popularity"] = float(
                (nr <= 10).mean() - (np.concatenate(pop_ranks) <= 10).mean())
    return out
