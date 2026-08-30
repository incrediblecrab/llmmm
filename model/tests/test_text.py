"""The recipe-text join.

The risk this file guards against is not that the join is missing text — a
blank title is obvious and harmless. It is that the join is *shifted*: text
that belongs to recipe n attached to recipe n+1. That failure is silent,
plausible and corrupts every downstream use, so the checks here are about
position, not content.
"""
from __future__ import annotations

import numpy as np
import pytest

from ingredient_model.config import PATHS
from ingredient_model.data import text as text_mod
from ingredient_model.data.recipes import load_recipes

pytestmark = pytest.mark.skipif(
    not (PATHS.recipes / text_mod.TEXT_FILE).exists(),
    reason="text index not built (make text)",
)


@pytest.fixture(scope="module")
def idx():
    return text_mod.load_text()


@pytest.fixture(scope="module")
def corp():
    return load_recipes()


def test_one_row_per_recipe(idx, corp):
    assert len(idx) == corp.n_recipes


def test_index_is_positional(idx):
    """idx must be 0..n-1 in order. A gap or a repeat means some source
    yielded a different number of records on the text pass than on the
    corpus pass, which shifts every row after it."""
    a = idx["idx"].to_numpy()
    assert a[0] == 0
    assert np.array_equal(a, np.arange(len(a)))


def test_title_coverage_is_high(idx):
    have = (idx["title"].str.len() > 0).mean()
    assert have > 0.95, f"title coverage fell to {have:.1%}"


def test_blank_sources_are_aligned_not_dropped(idx, corp):
    """Sources with no text reader must still occupy their rows. If they were
    skipped instead, the rows after them would carry someone else's text."""
    src = np.asarray(corp.source)
    seen = False
    for name in ("07-indian", "hebrew-9.7k", "08-indonesian"):
        rows = np.flatnonzero(src == name)
        if not len(rows):
            continue
        seen = True
        assert (idx["idx"].to_numpy()[rows] == rows).all()
        assert (idx["title"].to_numpy()[rows] == "").all()
    assert seen, "none of the blank-text sources were found in the corpus"


@pytest.mark.parametrize("i", [0, 1, 1_000_000, 1_994_537, 4_647_846])
def test_text_belongs_to_the_right_recipe(idx, corp, i):
    """A sampled row's decoded tokens should appear in its own raw text.

    Only meaningful for English sources: the normaliser translates, so a
    Chinese recipe carries tokens like "peanut" over text reading 花生 and no
    substring of one occurs in the other. Non-English rows are checked for
    alignment only, which is what this file is really about.
    """
    row = idx.iloc[i]
    assert int(row["idx"]) == i
    if corp.lang[i] != "en":
        pytest.skip("cross-lingual: tokens are translations, not substrings")
    blob = (str(row["raw_ingredients"]) + " " + str(row["title"])).lower()
    toks = [corp.itos[t] for t in corp.recipe(i)]
    if not toks or not blob.strip():
        pytest.skip("no text or no tokens for this row")
    hits = sum(1 for t in toks if t.split("_")[0] in blob)
    assert hits > 0, f"row {i}: no token of {toks} appears in its own text"


def test_normalisation_is_cross_lingual(idx, corp):
    """The counterpart to the skip above, stated as a positive claim: Chinese
    recipes really do carry English tokens. This is the normaliser doing
    translation, and it is why the corpus is comparable across languages."""
    zh = np.flatnonzero(np.asarray(corp.lang) == "zh")
    if not len(zh):
        pytest.skip("no Chinese recipes")
    i = int(zh[len(zh) // 2])
    toks = [corp.itos[t] for t in corp.recipe(i)]
    assert toks, "a Chinese recipe decoded to no tokens"
    assert all(t.isascii() for t in toks), f"expected English tokens, got {toks}"
