"""The leakage rule is the workspace's central claim, so it is asserted here.

If these tests ever pass vacuously — because a split was renamed, or the guard
was softened to a warning — the leaderboard silently starts mixing numbers that
are not comparable. That failure is invisible in the output, which is why it is
tested rather than documented.
"""
import pytest

from ingredient_model.data.splits import (DEFAULT_SPLIT, SPLITS, LeakageError,
                                          check_leakage, get_split)


def test_every_split_declares_what_it_is_valid_for():
    for name, split in SPLITS.items():
        assert split.name == name, "the key and the name must not drift apart"
        assert split.description, f"{name} has no description"
        assert split.graph, f"{name} has no training graph"


def test_recipe_model_on_edge_split_is_refused():
    with pytest.raises(LeakageError) as e:
        check_leakage(get_split("edge-holdout"), ("recipes",))
    assert "recipe" in str(e.value).lower()


def test_recipe_model_on_recipe_split_is_allowed():
    check_leakage(get_split("recipe-holdout"), ("recipes",))


def test_graph_model_is_allowed_on_either_holdout():
    for name in ("edge-holdout", "recipe-holdout"):
        check_leakage(get_split(name), ("ii_graph_train",))


def test_the_default_split_is_the_honest_one():
    """The default has to be the split that is valid for every family. If the
    convenient default is the unsound one, the unsound number is what gets
    quoted."""
    check_leakage(get_split(DEFAULT_SPLIT), ("recipes",))


def test_unknown_split_names_fail_loudly():
    with pytest.raises(KeyError):
        get_split("no-such-split")
