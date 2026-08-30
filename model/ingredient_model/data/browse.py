"""Browse the corpus in human-readable form.

The corpus is stored as `uint16` token ids because that is what the models
consume, but the ids are not inspectable: a bug in normalisation, a misaligned
vocabulary or a truncated source all look identical once everything is an
integer. Every claim this workspace makes rests on those ids meaning what we
think they mean, so they have to be readable.

Two things are deliberately *not* recoverable and it is better to say so than to
imply otherwise:

* **Ingredient order and duplicates.** The builder stores ``sorted(set(ids))``,
  so "flour, butter, flour" and "butter, flour" are the same row. This is
  correct for the co-occurrence models — a recipe is a set — but it means the
  corpus cannot be used for anything order-sensitive without rebuilding it.
* **Titles, quantities and instructions.** The readers only ever yielded
  ingredient lists, so the corpus never contained them. Recovering them means
  going back to the source files; :func:`source_records` does that for one
  source at a time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .recipes import RecipeCorpus, load_recipes


@dataclass(frozen=True)
class RecipeView:
    """One recipe, rendered."""

    index: int
    source: str
    lang: str
    ingredients: tuple[str, ...]

    def __str__(self) -> str:
        return (f"#{self.index:<9,} {self.source:<22} {self.lang:<3} "
                f"({len(self.ingredients):2d})  "
                f"{', '.join(self.ingredients)}")


def view(index: int, corpus: RecipeCorpus | None = None) -> RecipeView:
    """Render a single recipe by corpus index."""
    c = corpus or load_recipes()
    if not 0 <= index < c.n_recipes:
        raise IndexError(
            f"recipe {index} out of range — corpus holds {c.n_recipes:,}")
    return RecipeView(index=index, source=str(c.source[index]),
                      lang=str(c.lang[index]),
                      ingredients=tuple(c.itos[i] for i in c.recipe(index)))


def sample(n: int = 10, *, source: str | None = None, lang: str | None = None,
           contains: str | None = None, min_size: int = 0, seed: int = 0,
           corpus: RecipeCorpus | None = None) -> list[RecipeView]:
    """A random sample, optionally filtered.

    Sampling is uniform over the *matching* rows rather than over a prefix. The
    corpus is stored grouped by source, so taking the first n rows of a filter
    returns one source and one language — the same bias that was silently
    restricting the M6 test set to English recipes.
    """
    c = corpus or load_recipes()
    keep = np.ones(c.n_recipes, bool)
    if source is not None:
        keep &= np.char.find(c.source.astype(str), source) >= 0
    if lang is not None:
        keep &= c.lang.astype(str) == lang
    if min_size:
        keep &= c.sizes >= min_size
    if contains is not None:
        try:
            tok = c.itos.index(contains)
        except ValueError:
            raise KeyError(f"{contains!r} is not in the vocabulary") from None
        has = np.zeros(c.n_recipes, bool)
        rows = np.repeat(np.arange(c.n_recipes), c.sizes)
        has[np.unique(rows[c.flat == tok])] = True
        keep &= has

    idx = np.flatnonzero(keep)
    if len(idx) == 0:
        return []
    rng = np.random.default_rng(seed)
    pick = rng.choice(idx, min(n, len(idx)), replace=False)
    return [view(int(i), c) for i in np.sort(pick)]


def coverage(corpus: RecipeCorpus | None = None) -> dict:
    """Prove every recipe is reachable and decodes to real names.

    "All recipes are viewable" is a claim about all 4.6M of them, so it is
    checked over all 4.6M rather than over a sample.
    """
    c = corpus or load_recipes()
    sizes = c.sizes
    lo, hi = c.offsets[:-1], c.offsets[1:]

    contiguous = bool(np.array_equal(hi[:-1], lo[1:]))
    covers_all = bool(lo[0] == 0 and hi[-1] == len(c.flat))
    in_vocab = bool(c.flat.max() < len(c.itos))
    named = all(isinstance(s, str) and s for s in c.itos)
    # Within a recipe ids are stored sorted and de-duplicated; if that ever
    # stopped holding, set-based models would double-count an ingredient.
    step = np.diff(c.flat.astype(np.int64))
    boundary = np.zeros(len(step), bool)
    boundary[hi[:-1] - 1] = True
    strictly_sorted = bool(np.all(step[~boundary] > 0))

    return {
        "n_recipes": int(c.n_recipes),
        "n_tokens": int(len(c.flat)),
        "vocabulary": int(len(c.itos)),
        "offsets_contiguous": contiguous,
        "offsets_cover_all_tokens": covers_all,
        "all_ids_in_vocabulary": in_vocab,
        "all_vocabulary_entries_named": named,
        "recipes_sorted_and_deduplicated": strictly_sorted,
        "min_size": int(sizes.min()),
        "max_size": int(sizes.max()),
        "mean_size": float(sizes.mean()),
        "n_sources": int(len(np.unique(c.source))),
        "n_languages": int(len(np.unique(c.lang))),
        "all_viewable": bool(contiguous and covers_all and in_vocab
                             and named and strictly_sorted),
    }


def breakdown(corpus: RecipeCorpus | None = None) -> list[tuple[str, str, int, float]]:
    """Per-source recipe counts and mean size, largest first."""
    c = corpus or load_recipes()
    src = c.source.astype(str)
    sizes = c.sizes
    out = []
    for s in np.unique(src):
        m = src == s
        langs = np.unique(c.lang.astype(str)[m])
        out.append((s, "/".join(langs), int(m.sum()), float(sizes[m].mean())))
    return sorted(out, key=lambda r: -r[2])


def source_records(key: str, limit: int = 5) -> list[tuple[str, list[str]]]:
    """Read raw ingredient lines straight from the original source files.

    This is the only check that can catch a normalisation error. Everything
    else in the workspace compares derived artefacts against other derived
    artefacts, which cannot detect a reader that silently drops a column or
    mis-splits a delimiter — that failure is perfectly self-consistent
    downstream.

    Requires the prior study's readers (``prior-study/tools``); returns an empty
    list if they are not present, since the derived artefacts are self-sufficient
    for training.
    """
    import sys

    from ..config import PATHS

    root = PATHS.prior_study
    if not (root / "tools" / "corpus.py").exists():
        return []
    # Left on the path: the readers import sibling modules lazily, so removing
    # it after the initial import breaks them mid-iteration.
    if str(root / "tools") not in sys.path:
        sys.path.insert(0, str(root / "tools"))
    try:
        import corpus as raw  # type: ignore
    except Exception:
        return []

    out = []
    for k, _lang, items in raw.iter_all(None):
        if k != key:
            continue
        out.append((k, [str(i) for i in items]))
        if len(out) >= limit:
            break
    return out
