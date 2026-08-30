"""Centred M6, and the explanation that turned out to be wrong.

Re-scoring M6 with the embedding mean removed changes ``glove`` from 0.2509 to
0.4919 — fourth place to first — while moving ``sgns-cooc`` and ``svd-ppmi`` by
less than sampling noise. That is a reproducible fact about the recorded runs.

The *reason* is not known. The natural explanation is that cosine against a
summed context vector is not invariant to translating the embedding space, so an
off-centre family is being scored on an arbitrary gauge. These tests exist
because that explanation is **false**, and it is convincing enough to be worth
blocking permanently: a large pure translation leaves raw M6 essentially
unchanged. Popularity-in-the-mean and centroid size were also tested against the
real runs and also fail to predict which families move.

What *is* guaranteed, and is worth pinning: the centred variant is exactly
invariant to translation, so whatever else it measures, it does not measure
where a family put its centroid.
"""
from __future__ import annotations

import numpy as np
import pytest

from ingredient_model.eval.completion import recipe_completion


class _Corpus:
    def __init__(self, recipes):
        self.sizes = np.array([len(r) for r in recipes], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)[:-1]]).astype(np.int64)
        self.flat = np.concatenate(recipes).astype(np.uint16)
        self.n_recipes = len(recipes)

    def recipe(self, i):
        return self.flat[self.offsets[i]:self.offsets[i] + self.sizes[i]]


@pytest.fixture
def setup():
    """Ingredients in latent clusters; recipes drawn from within one cluster."""
    rng = np.random.default_rng(0)
    n_items, d, n_clusters = 120, 16, 8
    centres = rng.normal(size=(n_clusters, d))
    member = np.repeat(np.arange(n_clusters), n_items // n_clusters)
    W = centres[member] * 3.0 + rng.normal(scale=0.4, size=(n_items, d))

    recipes = []
    for _ in range(4000):
        c = rng.integers(n_clusters)
        pool = np.flatnonzero(member == c)
        recipes.append(np.sort(rng.choice(pool, size=4, replace=False)))
    return W.astype(np.float32), _Corpus(recipes)


def _r10(W, corpus):
    return recipe_completion(W, corpus, n_test=2000)["M6_recall_at_10"]


def _centred(W):
    return W - W.mean(0, keepdims=True)


def test_centred_m6_is_exactly_invariant_to_translation(setup):
    """The one property the centred variant is guaranteed to have."""
    W, corpus = setup
    base = _r10(_centred(W), corpus)
    assert base > 0.2, "fixture should be learnable"
    for shift in (1.0, 30.0, -75.0):
        moved = W + shift * np.ones(W.shape[1], dtype=np.float32)
        assert abs(_r10(_centred(moved), corpus) - base) < 1e-9


def test_translation_does_not_explain_the_centring_gain(setup):
    """Blocks the tempting-but-false 'raw M6 is gauge-dependent' story.

    If a future change makes a pure translation actually wreck raw M6, this
    fails, and the explanation rejected in eval/completion.py deserves another
    look. Until then it stays rejected.
    """
    W, corpus = setup
    base = _r10(W, corpus)
    shifted = _r10(W + 30.0 * np.ones(W.shape[1], dtype=np.float32), corpus)
    assert abs(shifted - base) < 0.05, (
        f"a pure translation moved raw M6 from {base:.4f} to {shifted:.4f}. "
        "The gauge explanation for the centring gain may be right after all — "
        "re-open it in eval/completion.py rather than leaving it listed as "
        "rejected."
    )


def test_centring_is_idempotent(setup):
    W, corpus = setup
    Wc = _centred(W)
    assert abs(_r10(Wc, corpus) - _r10(_centred(Wc), corpus)) < 1e-9


def test_harness_reports_centred_alongside_raw():
    """Both numbers must be recorded; the centred one must not replace M6."""
    import inspect

    from ingredient_model.eval import harness

    src = inspect.getsource(harness.evaluate)
    assert "M6_centred_" in src, "harness no longer records the centred variant"
    assert src.count("recipe_completion(") == 2, (
        "expected exactly two completion calls, raw and centred"
    )


def test_centred_keys_do_not_collide_with_raw_keys(setup):
    """A naming slip that overwrote M6_* would silently change the leaderboard."""
    W, corpus = setup
    raw = recipe_completion(W, corpus, n_test=500)
    centred = {f"M6_centred_{k[3:]}": v for k, v in
               recipe_completion(_centred(W), corpus, n_test=500).items()
               if k.startswith("M6_") and not k.startswith("M6_popularity")
               and k != "M6_n"}
    assert centred, "centred variant produced no keys"
    assert not (set(raw) & set(centred))
