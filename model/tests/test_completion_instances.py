from __future__ import annotations

import numpy as np

from ingredient_model.data.recipes import RecipeCorpus
from ingredient_model.eval.completion import _rank_of_target, completion_ranks
from ingredient_model.eval.metrics import unit


def test_returned_instances_preserve_the_draw_and_rank_order():
    sizes = np.array([3, 4, 3, 5])
    corpus = RecipeCorpus(
        flat=np.array([0, 2, 4, 0, 1, 3, 5, 1, 2, 3, 0, 1, 2, 3, 4]),
        offsets=np.r_[0, np.cumsum(sizes)],
        lang=np.array(["en"] * 4), source=np.array(["fixture"] * 4),
        itos=[str(i) for i in range(6)])
    matrix = np.random.default_rng(1).normal(size=(6, 4))
    plain = completion_ranks(matrix, corpus, n_test=4, seed=2)
    detailed = completion_ranks(matrix, corpus, n_test=4, seed=2,
                                include_instances=True)
    np.testing.assert_array_equal(plain["embedding"], detailed["embedding"])
    vectors = unit(matrix)
    for i, (row, target) in enumerate(zip(detailed["recipe_row"], detailed["target"])):
        recipe = corpus.recipe(row)
        assert target in recipe
        assert detailed["recipe_size"][i] == len(recipe)
        context = recipe[recipe != target]
        scores = (vectors[context].sum(0) @ vectors.T)[None]
        forbid = np.zeros((1, corpus.n_vocab), dtype=bool)
        forbid[0, context] = True
        expected = _rank_of_target(scores, np.array([target]), forbid)[0]
        assert expected == detailed["embedding"][i]
