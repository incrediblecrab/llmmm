"""The tokenised recipe corpus.

4.6M recipes across 13 languages, each stored as a **sorted set** of vocabulary
ids: an ingredient is counted once per recipe however many times it is listed,
because "2 tbsp butter, plus butter for greasing" is one ingredient, not two.

Layout is a flat id array plus offsets (CSR over recipes) rather than a list of
lists. 4.6M Python lists cost several GB and make every pass a Python loop; the
flat form is 71 MB and lets slicing and counting stay in NumPy.

This is the substrate for recipe-level models. The ingredient-ingredient graph
is a *derived summary* of it — training directly on recipes keeps set-level
structure (a recipe is a co-selected basket) that pairwise NPMI throws away.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import PATHS, SEED

RECIPE_IDS = "recipe_ids.npz"


@dataclass(frozen=True)
class RecipeCorpus:
    flat: np.ndarray
    offsets: np.ndarray
    lang: np.ndarray
    source: np.ndarray
    itos: list[str]

    @property
    def n_recipes(self) -> int:
        return len(self.offsets) - 1

    @property
    def n_vocab(self) -> int:
        return len(self.itos)

    @property
    def sizes(self) -> np.ndarray:
        return np.diff(self.offsets)

    def recipe(self, i: int) -> np.ndarray:
        return self.flat[self.offsets[i]:self.offsets[i + 1]]

    def unigram(self) -> np.ndarray:
        """Recipes containing each ingredient. Recipes, not occurrences —
        ingredient sets are deduplicated, so these coincide by construction."""
        return np.bincount(self.flat.astype(np.int64),
                           minlength=self.n_vocab).astype(np.float64)

    def select(self, rows: np.ndarray) -> "RecipeCorpus":
        """A sub-corpus over the given recipe indices, re-packed.

        Used for cuisine stratification and for recipe-level held-out splits.
        The vocabulary is preserved verbatim so embeddings trained on any
        stratum stay row-comparable with every other.
        """
        rows = np.asarray(rows, np.int64)
        lens = self.sizes[rows]
        new_off = np.zeros(len(rows) + 1, np.int64)
        np.cumsum(lens, out=new_off[1:])
        flat = np.empty(int(new_off[-1]), self.flat.dtype)
        for j, r in enumerate(rows):
            flat[new_off[j]:new_off[j + 1]] = self.recipe(int(r))
        return RecipeCorpus(flat=flat, offsets=new_off, lang=self.lang[rows],
                            source=self.source[rows], itos=self.itos)

    def by_language(self, code: str) -> "RecipeCorpus":
        return self.select(np.nonzero(self.lang == code)[0])

    def by_source(self, name: str) -> "RecipeCorpus":
        return self.select(np.nonzero(self.source == name)[0])

    def split(self, frac: float = 0.1, seed: int = SEED
              ) -> tuple["RecipeCorpus", "RecipeCorpus"]:
        """Hold out whole recipes, not individual edges.

        Removing edges leaves the recipes that produced them in the corpus, so a
        set-level model can reconstruct a "held-out" pair from the basket it was
        drawn from. Holding out entire recipes is the only split that is honest
        for models which read recipes directly.
        """
        rng = np.random.default_rng(seed)
        perm = rng.permutation(self.n_recipes)
        cut = int(self.n_recipes * frac)
        return self.select(np.sort(perm[cut:])), self.select(np.sort(perm[:cut]))

    def csr_matrix(self):
        """Recipe x ingredient sparse indicator, for factorisation models."""
        from scipy.sparse import csr_matrix
        return csr_matrix(
            (np.ones(len(self.flat), np.float32),
             self.flat.astype(np.int32), self.offsets.astype(np.int64)),
            shape=(self.n_recipes, self.n_vocab))

    def batches(self, batch_size: int, *, min_size: int = 2, shuffle: bool = True,
                seed: int = SEED, max_len: int | None = None):
        """Yield padded ``(ids, mask)`` batches of recipes.

        Recipes are bucketed by length before batching so a batch is mostly one
        length and padding stays near zero. With sizes ranging 1..60, naive
        batching pads to the longest member and wastes most of the compute.
        """
        lens = self.sizes
        keep = np.nonzero(lens >= min_size)[0]
        if max_len is not None:
            keep = keep[lens[keep] <= max_len]
        order = keep[np.argsort(lens[keep], kind="stable")]
        rng = np.random.default_rng(seed)
        blocks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
        if shuffle:
            rng.shuffle(blocks)
        for block in blocks:
            width = int(lens[block].max())
            ids = np.zeros((len(block), width), np.int64)
            mask = np.zeros((len(block), width), bool)
            for j, r in enumerate(block):
                r_ids = self.recipe(int(r))
                ids[j, :len(r_ids)] = r_ids
                mask[j, :len(r_ids)] = True
            yield ids, mask


@functools.lru_cache(maxsize=1)
def load_recipes(name: str = RECIPE_IDS) -> RecipeCorpus:
    path = PATHS.recipes / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Populate the workspace first:\n"
            f"    python scripts/import_data.py --from <llmmm-checkout>")
    z = np.load(path, allow_pickle=True)
    return RecipeCorpus(
        flat=z["flat"], offsets=z["offsets"].astype(np.int64),
        lang=z["lang"], source=z["source"], itos=[str(x) for x in z["itos"]])
