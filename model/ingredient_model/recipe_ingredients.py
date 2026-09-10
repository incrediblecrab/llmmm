"""Read-only canonical ingredient arrays for bounded, vectorized recipe filtering."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

from ._hashing import file_sha256


def _stamp(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@lru_cache(maxsize=1)
def _arrays(path: Path, stamp: tuple[int, int, int], expected_hash: str,
            n_recipes: int, n_slots: int, n_vocab: int) -> tuple[np.ndarray, np.ndarray]:
    if file_sha256(path) != expected_hash:
        raise ValueError("canonical ingredient corpus checksum differs from the catalog")
    with np.load(path, allow_pickle=False) as archive:
        flat, offsets = archive["flat"], archive["offsets"]
    if (flat.ndim != 1 or offsets.ndim != 1 or flat.dtype.kind not in "iu"
            or offsets.dtype.kind not in "iu" or len(flat) != n_slots
            or len(offsets) != n_recipes + 1 or offsets[0] != 0
            or offsets[-1] != n_slots or (offsets[1:] <= offsets[:-1]).any()
            or (np.diff(offsets) > n_vocab).any()
            or (flat < 0).any() or (flat >= n_vocab).any()):
        raise ValueError("canonical ingredient arrays have invalid shapes, offsets or vocabulary IDs")
    invalid_order = flat[1:] <= flat[:-1]
    invalid_order[offsets[1:-1] - 1] = False
    if invalid_order.any():
        raise ValueError("canonical ingredient rows must be sorted unique sets")
    if _stamp(path) != stamp:
        raise ValueError("canonical ingredient corpus changed while loading")
    flat.flags.writeable = offsets.flags.writeable = False
    return flat, offsets


@dataclass(frozen=True)
class CanonicalIngredientIndex:
    path: Path
    stamp: tuple[int, int, int]
    flat: np.ndarray
    offsets: np.ndarray
    n_vocab: int

    @classmethod
    def load(cls, path: Path, *, corpus_sha256: str, n_recipes: int,
             n_slots: int, n_vocab: int) -> "CanonicalIngredientIndex":
        path = path.resolve()
        stamp = _stamp(path)
        for name, value in (("n_recipes", n_recipes), ("n_slots", n_slots), ("n_vocab", n_vocab)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        flat, offsets = _arrays(path, stamp, corpus_sha256, n_recipes, n_slots, n_vocab)
        return cls(path, stamp, flat, offsets, n_vocab)

    def check_unchanged(self) -> None:
        if _stamp(self.path) != self.stamp:
            raise ValueError("canonical ingredient corpus changed after initialization")

    def filter_ids(self, ids: Sequence[int], available: frozenset[int],
                   max_missing: int) -> list[int]:
        membership = np.zeros(self.n_vocab, dtype=bool)
        membership[list(available)] = True
        kept = []
        for start in range(0, len(ids), 512):
            batch = np.asarray(ids[start:start + 512], dtype=np.int64)
            if ((batch < 0) | (batch >= len(self.offsets) - 1)).any():
                raise ValueError("candidate ID is outside the canonical ingredient corpus")
            begins = self.offsets[batch].astype(np.int64)
            lengths = self.offsets[batch + 1].astype(np.int64) - begins
            columns = np.arange(int(lengths.max()))
            valid = columns[None, :] < lengths[:, None]
            positions = begins[:, None] + columns[None, :]
            positions[~valid] = 0
            overlap = (membership[self.flat[positions]] & valid).sum(axis=1)
            kept.extend(batch[lengths - overlap <= max_missing].tolist())
        return kept

    def recipe_ids(self, recipe_id: int) -> frozenset[int]:
        return frozenset(int(value) for value in self.flat[
            self.offsets[recipe_id]:self.offsets[recipe_id + 1]])
