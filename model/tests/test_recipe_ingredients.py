from __future__ import annotations

import numpy as np
import pytest

from ingredient_model._hashing import file_sha256
from ingredient_model.recipe_ingredients import CanonicalIngredientIndex


def index_for(path, rows, *, offset_dtype=np.int64):
    flat = np.asarray([value for row in rows for value in row], dtype=np.uint16)
    offsets = np.asarray([0, *np.cumsum([len(row) for row in rows])], dtype=offset_dtype)
    np.savez(path, flat=flat, offsets=offsets)
    return CanonicalIngredientIndex.load(
        path, corpus_sha256=file_sha256(path), n_recipes=len(rows),
        n_slots=len(flat), n_vocab=6)


@pytest.mark.parametrize("offset_dtype", [np.int64, np.uint64])
def test_vector_filter_matches_set_difference_and_preserves_order(tmp_path, offset_dtype):
    rows = [[0, 1, 2], [0, 2], [1], [2, 3, 4, 5], [4]]
    index = index_for(tmp_path / "corpus.npz", rows, offset_dtype=offset_dtype)
    order = [4, 3, 0, 2, 1] * 150
    for available in (frozenset({0, 1, 2}), frozenset({4}), frozenset(range(6))):
        for missing in (0, 1, 3):
            expected = [recipe_id for recipe_id in order
                        if len(set(rows[recipe_id]) - available) <= missing]
            assert index.filter_ids(order, available, missing) == expected
    assert index.filter_ids([], frozenset({0}), 0) == []
    assert not index.flat.flags.writeable
    assert not index.offsets.flags.writeable


def test_corpus_hash_mismatch_is_not_a_silent_fallback(tmp_path):
    path = tmp_path / "corpus.npz"
    index_for(path, [[0, 1], [2]])
    with pytest.raises(ValueError, match="checksum"):
        CanonicalIngredientIndex.load(
            path, corpus_sha256="0" * 64, n_recipes=2, n_slots=3, n_vocab=6)


@pytest.mark.parametrize("flat,offsets", [
    ([0, 0], [0, 2]),
    ([1, 0], [0, 2]),
    ([0, 6], [0, 2]),
    ([0, 1], [0, 0, 2]),
    ([0, 1], [0, 3, 2]),
])
def test_malformed_canonical_arrays_fail_closed(tmp_path, flat, offsets):
    path = tmp_path / "invalid.npz"
    np.savez(path, flat=np.asarray(flat, dtype=np.int64),
             offsets=np.asarray(offsets, dtype=np.int64))
    with pytest.raises(ValueError, match="canonical ingredient"):
        CanonicalIngredientIndex.load(
            path, corpus_sha256=file_sha256(path), n_recipes=len(offsets) - 1,
            n_slots=len(flat), n_vocab=6)


def test_changed_corpus_and_out_of_range_rows_are_explicit_errors(tmp_path):
    path = tmp_path / "corpus.npz"
    index = index_for(path, [[0, 1], [2]])
    with pytest.raises(ValueError, match="candidate ID"):
        index.filter_ids([2], frozenset({0}), 0)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after initialization"):
        index.check_unchanged()
