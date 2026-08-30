"""Evaluation splits, and the leakage rule that makes them comparable.

A model may only be scored against labels it could not have seen. That sounds
obvious and is easy to violate here, because the same corpus reaches models
through two different doors:

* **Graph models** read the ingredient-ingredient graph. Removing 10% of its
  edges hides them.
* **Recipe models** read recipes. Removing an *edge* from the graph hides
  nothing from them — the recipes that produced that edge are still in the
  corpus, so the "held-out" pair is fully observable.

So there are two protocols, and which one is valid depends on what the model
reads:

``edge-holdout``    10% of graph edges removed. Valid for graph models only.
``recipe-holdout``  30% of *recipes* removed and the graph rebuilt from the
                    remainder. Valid for every model, and therefore the only
                    protocol under which the two families may be compared.

:func:`check_leakage` enforces this. It refuses rather than warns, because an
optimistic number that is merely flagged still ends up quoted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .graphs import GRAPH_HELDOUT, GRAPH_TRAIN


@dataclass(frozen=True)
class Split:
    name: str
    graph: str
    """Training graph — the one a graph model may walk or factor."""

    heldout: str
    """Held-out edges — labels for M4. Never a training input."""

    corpus: str | None
    """Training corpus for recipe models. ``None`` means the protocol does not
    provide a leak-free corpus, so recipe models cannot use it."""

    description: str = ""

    @property
    def safe_for_recipes(self) -> bool:
        return self.corpus is not None


SPLITS: dict[str, Split] = {
    "edge-holdout": Split(
        name="edge-holdout", graph=GRAPH_TRAIN, heldout=GRAPH_HELDOUT,
        corpus=None,
        description="10% of graph edges removed; graph models only"),
    "recipe-holdout": Split(
        name="recipe-holdout", graph="ii_graph_rh_train.npz",
        heldout="ii_graph_rh_heldout.npz", corpus="recipe_ids_rh_train.npz",
        description="30% of recipes removed and the graph rebuilt; all models"),
    "full": Split(
        name="full", graph="ii_graph.npz", heldout=GRAPH_HELDOUT,
        corpus="recipe_ids.npz",
        description="every edge and every recipe — production artefacts only, "
                    "M4 is meaningless here"),
}

#: The default has to be the protocol that is valid for *every* family, not the
#: one that reproduces the most prior numbers. A default that is unsound for
#: recipe models is a default that quietly produces uncomparable leaderboards
#: whenever someone omits the flag — and the inflated number is the one that
#: gets quoted. Graph models can still opt into `edge-holdout` explicitly to
#: compare against the earlier study.
DEFAULT_SPLIT = "recipe-holdout"


class LeakageError(RuntimeError):
    pass


def check_leakage(split: Split, requires: tuple[str, ...],
                  strict: bool = True) -> str | None:
    """Refuse to score a recipe model against edge-level labels.

    Returns a warning string when the combination is unsound and ``strict`` is
    off; raises otherwise.
    """
    if "recipes" not in requires:
        return None
    if split.name == "full":
        msg = (f"split 'full' holds nothing out — M4 measures memorisation, "
               f"not generalisation")
    elif not split.safe_for_recipes:
        msg = (f"a recipe-reading model cannot be scored on split "
               f"{split.name!r}: removing graph edges does not remove the "
               f"recipes that produced them, so the held-out pairs are fully "
               f"visible in training. Use --split recipe-holdout "
               f"(build it with: python scripts/build_splits.py)")
    else:
        return None
    if strict:
        raise LeakageError(msg)
    return msg


def get_split(name: str) -> Split:
    if name not in SPLITS:
        raise KeyError(f"unknown split {name!r}. Available: "
                       f"{', '.join(sorted(SPLITS))}")
    return SPLITS[name]


def held_out_recipes(split_name: str, limit: int | None = 80_000):
    """The recipes no model saw, for M6. ``None`` when the split has none.

    Only the recipe-level protocol holds recipes back, so M6 is reported there
    and omitted elsewhere rather than quietly computed against training data.

    ``limit=None`` returns the whole complement in corpus order; any integer
    draws a uniform sample of that size.

    The test set is recovered as the complement of the training corpus under the
    same seeded permutation the builder used, rather than being stored a second
    time. That is exact, but it couples this function to
    ``RecipeCorpus.split`` — ``tests/test_completion_corpus.py`` asserts the two
    still agree, because if they silently drift M6 would be scored on training
    recipes and would simply look better.
    """
    from ..config import SEED
    from .recipes import load_recipes

    split = get_split(split_name)
    if split.name != "recipe-holdout" or not split.corpus:
        return None
    from ..config import PATHS
    if not (PATHS.recipes / split.corpus).exists():
        return None

    full = load_recipes()
    train = load_recipes(split.corpus)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(full.n_recipes)
    cut = full.n_recipes - train.n_recipes
    test_idx = perm[:cut]
    if limit and limit < len(test_idx):
        # Sample, don't take the head. The corpus is stored grouped by source —
        # the first two million rows are all RecipeNLG — so slicing the lowest
        # indices yields an English-only test set and M6 silently stops
        # measuring the 31% of the corpus that is Chinese.
        test_idx = rng.choice(test_idx, limit, replace=False)
    return full.select(np.sort(test_idx))
