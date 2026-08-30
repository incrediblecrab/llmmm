"""Centre/context exposure in item2vec must not depend on the vocabulary index.

Recipes are stored in ascending id order and the vocabulary is sorted
alphabetically, so ``np.triu_indices`` emits pairs whose first column is always
the lower id. Skip-gram is asymmetric — column 0 trains the input matrix, which
is the matrix that gets exported as the embedding — so an unrandomised pair
stream trains each ingredient in proportion to how early its name sorts.

On the build that shipped this defect the effect was not marginal:
corr(centre-share, vocabulary index) = -0.980, the first decile appearing as
centre 96% of the time and the last decile 4%. Downstream, item2vec's nearest
neighbours were alphabetical (``butter`` -> ``buttermilk``, ``cantal_cheese``,
``cookie_butter``; ``yeast`` -> ``worcestershire_sauce``, ``yam``,
``whole_wheat_flour``) and M6 recall@10 was 0.0066, at chance, while M4 link-AUC
was the best on the leaderboard at 0.7179. That gap is what made the defect
survive: the metric everyone quoted looked excellent.

These tests assert the property, not the implementation, so they stay valid if
the fix later changes from a random swap to emitting both directions.
"""
from __future__ import annotations

import numpy as np
import pytest

from models.recipe_basket.item2vec import _pair_stream


class _Corpus:
    """A corpus whose recipes are sorted, exactly as the real loader stores them."""

    def __init__(self, recipes):
        assert all(list(r) == sorted(r) for r in recipes), "fixture must be sorted"
        self.sizes = np.array([len(r) for r in recipes], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)[:-1]]).astype(np.int64)
        self.flat = np.concatenate(recipes).astype(np.uint16)


@pytest.fixture
def corpus():
    rng = np.random.default_rng(0)
    n_items = 60
    recipes = [np.sort(rng.choice(n_items, size=int(rng.integers(2, 9)), replace=False))
               for _ in range(4000)]
    return _Corpus(recipes), n_items


def _centre_share(pairs, n_items):
    a = np.bincount(pairs[:, 0], minlength=n_items).astype(float)
    b = np.bincount(pairs[:, 1], minlength=n_items).astype(float)
    total = a + b
    seen = total > 0
    return a[seen] / total[seen], np.nonzero(seen)[0]


def test_centre_share_is_independent_of_vocabulary_index(corpus):
    c, n_items = corpus
    pairs = _pair_stream(c, np.random.default_rng(1), 0.0, np.ones(n_items))
    share, ids = _centre_share(pairs, n_items)
    r = np.corrcoef(ids, share)[0, 1]
    assert abs(r) < 0.2, (
        f"centre-share still tracks the vocabulary index (corr={r:+.3f}); "
        "the input matrix is being trained in alphabetical order"
    )


def test_every_ingredient_is_centre_about_half_the_time(corpus):
    c, n_items = corpus
    pairs = _pair_stream(c, np.random.default_rng(2), 0.0, np.ones(n_items))
    share, _ = _centre_share(pairs, n_items)
    assert share.min() > 0.35 and share.max() < 0.65, (
        f"centre-share spans {share.min():.2f}-{share.max():.2f}; every "
        "ingredient must serve as centre and as context roughly equally"
    )


def test_the_unfixed_ordering_would_be_caught(corpus):
    """The guard above fails on the original pair stream, so it is not vacuous."""
    c, n_items = corpus
    pairs = _pair_stream(c, np.random.default_rng(3), 0.0, np.ones(n_items))
    broken = np.sort(pairs, axis=1)          # what triu_indices produced before
    share, ids = _centre_share(broken, n_items)
    assert abs(np.corrcoef(ids, share)[0, 1]) > 0.8


def test_pairs_are_preserved_as_unordered_pairs(corpus):
    """Randomising direction must not add, drop or alter any pair."""
    c, n_items = corpus
    pairs = _pair_stream(c, np.random.default_rng(4), 0.0, np.ones(n_items))
    got = np.sort(pairs, axis=1)

    expected = []
    for k in range(2, int(c.sizes.max()) + 1):
        rows = np.nonzero(c.sizes == k)[0]
        if not len(rows):
            continue
        iu, ju = np.triu_indices(k, 1)
        ids = c.flat[c.offsets[rows][:, None] + np.arange(k)].astype(np.int64)
        expected.append(np.stack([ids[:, iu].ravel(), ids[:, ju].ravel()], 1))
    expected = np.sort(np.concatenate(expected), axis=1)

    assert len(got) == len(expected)
    key = lambda p: np.sort(p[:, 0].astype(np.int64) * 100_000 + p[:, 1])
    np.testing.assert_array_equal(key(got), key(expected))


def test_no_self_pairs(corpus):
    c, n_items = corpus
    pairs = _pair_stream(c, np.random.default_rng(5), 0.0, np.ones(n_items))
    assert not (pairs[:, 0] == pairs[:, 1]).any()
