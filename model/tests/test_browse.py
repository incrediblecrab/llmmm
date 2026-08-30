"""The corpus must stay readable.

Token ids are not inspectable — a normalisation bug, a misaligned vocabulary and
a truncated source all look identical once everything is an integer. These tests
assert the properties that make the corpus browsable at all, over all 4.6M rows
rather than a sample, because "all recipes are viewable" is a claim about all of
them.
"""
from __future__ import annotations

import numpy as np
import pytest

from ingredient_model.data.browse import breakdown, coverage, sample, view
from ingredient_model.data.recipes import load_recipes


def test_every_recipe_is_reachable_and_decodes():
    c = coverage()
    assert c["all_viewable"], c
    assert c["offsets_contiguous"]
    assert c["offsets_cover_all_tokens"]
    assert c["all_ids_in_vocabulary"]
    assert c["all_vocabulary_entries_named"]
    assert c["n_recipes"] > 4_000_000


def test_recipes_are_stored_sorted_and_deduplicated():
    """Set-based models would double-count a repeated ingredient, and the
    hidden-target sampler assumes one row is one set."""
    assert coverage()["recipes_sorted_and_deduplicated"]


def test_first_and_last_recipes_render():
    c = load_recipes()
    for i in (0, c.n_recipes - 1):
        v = view(i, c)
        assert v.ingredients and all(isinstance(x, str) for x in v.ingredients)
        assert v.source and v.lang


def test_index_out_of_range_is_rejected():
    c = load_recipes()
    with pytest.raises(IndexError):
        view(c.n_recipes, c)


def test_sampling_is_not_biased_to_one_source():
    """The corpus is stored grouped by source, so a prefix-based sample returns
    only RecipeNLG — the exact bias that once restricted the M6 test set to
    English recipes."""
    rows = sample(400, seed=0)
    assert len({r.source for r in rows}) >= 3, {r.source for r in rows}


def test_filters_actually_filter():
    assert all(r.lang == "zh" for r in sample(20, lang="zh"))
    assert all("saffron" in r.ingredients for r in sample(10, contains="saffron"))
    assert all(len(r.ingredients) >= 12 for r in sample(10, min_size=12))


def test_unknown_ingredient_is_reported_clearly():
    with pytest.raises(KeyError):
        sample(5, contains="definitely_not_an_ingredient")


def test_breakdown_accounts_for_every_recipe():
    c = load_recipes()
    rows = breakdown(c)
    assert sum(n for _s, _l, n, _m in rows) == c.n_recipes
    assert len(rows) == len(np.unique(c.source))
