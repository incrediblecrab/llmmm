from __future__ import annotations

import numpy as np
import pytest

from ingredient_model.data.recipes import RecipeCorpus
from ingredient_model.spec import TrainContext


def fixture_corpus():
    sizes = np.array([3, 4, 2, 5])
    return RecipeCorpus(
        flat=np.array([0, 1, 2, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3, 4]),
        offsets=np.r_[0, np.cumsum(sizes)],
        lang=np.array(["en"] * 4), source=np.array(["fixture"] * 4),
        itos=[str(i) for i in range(5)])


def test_training_records_actual_examples_not_the_sampling_cap(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from models.set_transformer import train

    monkeypatch.setattr(train, "load_recipes", lambda _: fixture_corpus())
    params = dict(d_model=8, n_heads=2, n_layers=1, epochs=2, batch_size=2,
                  max_len=4, max_recipes=0, dropout=0.0, warmup=0)
    context = TrainContext(graph="fixture", seed=1, out_dir=tmp_path, params=params)
    result = train.train_masked_set(context)
    assert result.metadata["n_recipes"] == 4
    assert result.metadata["n_eligible_recipes"] == 2
    assert result.metadata["n_examples_seen"] == 4
    assert result.metadata["n_optimizer_steps"] == 2


def test_all_filtered_recipes_cannot_produce_success_shaped_training(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from models.set_transformer import train

    monkeypatch.setattr(train, "load_recipes", lambda _: fixture_corpus())
    context = TrainContext(
        graph="fixture", seed=1, out_dir=tmp_path, params={"max_len": 2})
    with pytest.raises(ValueError, match="no recipes satisfy"):
        train.train_masked_set(context)


def all_lengths_corpus():
    sizes = np.array([1, 2, 3, 98])
    return RecipeCorpus(
        flat=np.concatenate([np.arange(size) for size in sizes]),
        offsets=np.r_[0, np.cumsum(sizes)],
        lang=np.array(["en"] * 4), source=np.array(["fixture"] * 4),
        itos=[str(i) for i in range(98)])


def test_indexed_batches_keep_singletons_pairs_and_all_long_recipe_items():
    corpus = all_lengths_corpus()
    visited = []
    for ids, mask, rows in corpus.indexed_batches(2, min_size=1, max_len=None):
        for i, row in enumerate(rows):
            np.testing.assert_array_equal(ids[i, mask[i]], corpus.recipe(row))
        visited.extend(rows.tolist())
    assert sorted(visited) == list(range(corpus.n_recipes))


def test_production_training_uses_every_row_once_per_epoch(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from models.set_transformer import train

    monkeypatch.setattr(train, "load_recipes", lambda _: all_lengths_corpus())
    params = dict(d_model=8, n_heads=2, n_layers=1, epochs=2, batch_size=2,
                  min_len=1, max_len=None, max_recipes=0, expected_recipes=4,
                  dropout=0.0, warmup=0)
    context = TrainContext(graph="fixture", seed=1, out_dir=tmp_path,
                           params=params, split="full")
    result = train.train_masked_set(context)
    assert result.metadata["n_eligible_recipes"] == 4
    assert result.metadata["n_examples_seen"] == 8
    assert result.metadata["epoch_coverage"] == [
        {"epoch": epoch, "unique_recipes": 4, "examples_seen": 4,
         "ingredient_slots_seen": 104,
         "every_eligible_recipe_once": True}
        for epoch in (1, 2)
    ]
    assert result.metadata["n_ingredient_slots_seen"] == 208
    assert result.metadata["observed_min_len"] == 1
    assert result.metadata["observed_max_len"] == 98


def test_row_counts_alone_cannot_hide_a_duplicate_and_a_missing_recipe(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from models.set_transformer import train

    original = RecipeCorpus.indexed_batches

    def repeated(self, *args, **kwargs):
        for ids, mask, rows in original(self, *args, **kwargs):
            changed = rows.copy()
            if 0 in rows:
                changed[:] = 0
            yield ids, mask, changed

    monkeypatch.setattr(RecipeCorpus, "indexed_batches", repeated)
    monkeypatch.setattr(train, "load_recipes", lambda _: all_lengths_corpus())
    params = dict(d_model=8, n_heads=2, n_layers=1, epochs=1, batch_size=2,
                  min_len=1, max_len=None, max_recipes=0, expected_recipes=4,
                  dropout=0.0, warmup=0)
    context = TrainContext(graph="fixture", seed=1, out_dir=tmp_path, params=params)
    with pytest.raises(RuntimeError, match="exactly once"):
        train.train_masked_set(context)


@pytest.mark.parametrize("restriction", [
    {"max_recipes": 2},
    {"min_len": 3},
    {"max_len": 32},
])
def test_required_full_count_rejects_any_sampling_or_length_exclusion(
        tmp_path, monkeypatch, restriction):
    pytest.importorskip("torch")
    from models.set_transformer import train

    monkeypatch.setattr(train, "load_recipes", lambda _: all_lengths_corpus())
    params = {"max_recipes": 0, "min_len": 1, "max_len": None,
              "expected_recipes": 4, **restriction}
    context = TrainContext(graph="fixture", seed=1, out_dir=tmp_path, params=params)
    with pytest.raises(ValueError, match="expected all"):
        train.train_masked_set(context)
