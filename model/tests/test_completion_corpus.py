"""M6's test recipes must be recipes no model saw.

``held_out_recipes`` recovers the test set as the complement of the training
corpus under a seeded permutation rather than storing it twice. That is exact
only while it agrees with ``RecipeCorpus.split``, and if the two ever drift the
failure is silent and flattering: M6 would be scored partly on training data and
every model would simply look better. Hence this test.
"""
import numpy as np
import pytest

from ingredient_model.config import PATHS, SEED
from ingredient_model.data.recipes import load_recipes
from ingredient_model.data.splits import get_split, held_out_recipes

SPLIT = "recipe-holdout"
pytestmark = pytest.mark.skipif(
    not (PATHS.recipes / get_split(SPLIT).corpus).exists(),
    reason="recipe-holdout not built; run scripts/build_splits.py")


def _fingerprints(corpus, limit=None):
    """A hashable identity per recipe: its sorted ingredient multiset.

    Recipe *indices* are meaningless across corpora because `select` re-packs
    them, so identity has to come from contents.
    """
    n = corpus.n_recipes if limit is None else min(limit, corpus.n_recipes)
    return {tuple(sorted(corpus.recipe(i).tolist())) for i in range(n)}


def test_held_out_recipes_are_absent_from_the_training_corpus():
    test = held_out_recipes(SPLIT, limit=3000)
    assert test is not None and test.n_recipes > 0

    train = load_recipes(get_split(SPLIT).corpus)
    train_fp = _fingerprints(train, limit=400_000)
    test_fp = _fingerprints(test)

    # Distinct recipes routinely share an ingredient set — "flour, sugar,
    # butter, egg" appears thousands of times — so overlap here is collision,
    # not leakage. The measured rate between two disjoint slices of the
    # *training* corpus is ~30%, which is the floor this can approach. A broken
    # reconstruction would hand back training rows and sit near 100%.
    overlap = len(test_fp & train_fp) / len(test_fp)
    assert overlap < 0.60, (
        f"{overlap:.1%} of held-out recipes also appear in training — far above "
        f"the ~30% natural collision rate, so the complement reconstruction has "
        f"drifted from RecipeCorpus.split")


def test_completion_set_is_not_drawn_from_one_source():
    """The corpus is stored grouped by source, so any index-ordered slice is
    single-language. M6 measured on RecipeNLG alone is not M6 on this corpus."""
    test = held_out_recipes(SPLIT, limit=5000)
    sources = set(test.source.tolist())
    assert len(sources) >= 3, (
        f"completion set covers only {sources} — it is being sliced by index "
        f"rather than sampled")


def test_reconstruction_matches_the_builder_exactly():
    """The definitive check: rebuild the split the way the builder did and
    confirm the same recipes come back, content for content."""
    full = load_recipes()
    train = load_recipes(get_split(SPLIT).corpus)
    frac = (full.n_recipes - train.n_recipes) / full.n_recipes
    _, builder_test = full.split(frac=frac, seed=SEED)

    got = held_out_recipes(SPLIT, limit=None)
    assert got.n_recipes == builder_test.n_recipes == full.n_recipes - train.n_recipes
    for i in (0, 1, 2, 17, 999, 50_000, builder_test.n_recipes - 1):
        assert np.array_equal(got.recipe(i), builder_test.recipe(i)), (
            f"recipe {i} differs — held_out_recipes and RecipeCorpus.split "
            f"no longer agree")


def test_other_splits_have_no_completion_corpus():
    """M6 is only defined where recipes were actually held back. Elsewhere it
    must return None rather than quietly scoring on training data."""
    assert held_out_recipes("edge-holdout") is None
    assert held_out_recipes("full") is None
