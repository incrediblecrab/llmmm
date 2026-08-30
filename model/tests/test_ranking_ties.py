"""Ranking must not pay for representation collapse.

M6 originally ranked the target by counting only strictly-better candidates.
That is the optimistic rank under ties, and it is exploitable: a model whose
scores are degenerate gets a large tied block, and every member of that block is
credited with the best position in it.

This is not hypothetical. Representation collapse is the characteristic failure
of both families in this workspace most likely to suffer it — an over-smoothed
graph network converges to the dominant eigenvector, and a transformer trained
without warmup collapses to a single token direction. Either would have scored
~0.21 on M6, beating `sgc` (0.014) and `residual` (0.149), while having learnt
nothing at all.
"""
from __future__ import annotations

import numpy as np

from ingredient_model.eval.completion import _rank_of_target, recipe_completion
from ingredient_model.data.splits import held_out_recipes

N_VOCAB = 1790


def _corpus():
    return held_out_recipes("recipe-holdout", limit=4000)


def test_midrank_matches_optimistic_rank_when_there_are_no_ties():
    """The fix must not silently restate every previously measured result."""
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(64, 200))
    target = rng.integers(0, 200, 64)
    forbid = np.zeros((64, 200), bool)

    got = _rank_of_target(scores, target, forbid)
    tgt = scores[np.arange(64), target]
    optimistic = (scores > tgt[:, None]).sum(1) + 1
    assert np.array_equal(got, optimistic.astype(float))


def test_fully_tied_scores_give_the_middle_rank():
    """All candidates equal means the target is, in expectation, in the middle.
    The optimistic rule would call this rank 1."""
    scores = np.zeros((8, 100))
    target = np.arange(8)
    forbid = np.zeros((8, 100), bool)

    got = _rank_of_target(scores, target, forbid)
    assert np.allclose(got, 50.5), got


def test_excluded_context_does_not_join_the_tied_block():
    """Forbidden candidates are set to -inf; they must not be counted as tied
    with a target that also happens to score -inf."""
    scores = np.full((4, 50), -np.inf)
    scores[:, 10] = 1.0
    target = np.full(4, 10)
    forbid = np.zeros((4, 50), bool)
    forbid[:, :5] = True

    got = _rank_of_target(scores, target, forbid)
    assert np.allclose(got, 1.0), got


def test_collapsed_embedding_scores_at_or_below_chance():
    """The end-to-end guarantee: a rank-1 space earns nothing.

    Normalising a rank-1 matrix yields exactly two distinct rows, so the whole
    candidate set takes two scores and hundreds tie at the recall@10 cut.
    """
    corpus = _corpus()
    rng = np.random.default_rng(7)
    collapsed = rng.normal(size=(N_VOCAB, 1)) @ rng.normal(size=(1, 64))

    m6 = recipe_completion(collapsed, corpus, n_test=4000)["M6_recall_at_10"]
    chance = 10 / N_VOCAB
    assert m6 <= chance, f"collapsed space scored {m6:.4f} > chance {chance:.4f}"


def test_random_embedding_scores_near_chance():
    """A metric that gives credit to noise cannot support any claim."""
    corpus = _corpus()
    W = np.random.default_rng(3).normal(size=(N_VOCAB, 64))

    m6 = recipe_completion(W, corpus, n_test=4000)["M6_recall_at_10"]
    assert m6 < 3 * (10 / N_VOCAB), m6
