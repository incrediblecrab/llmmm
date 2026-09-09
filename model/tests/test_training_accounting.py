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
