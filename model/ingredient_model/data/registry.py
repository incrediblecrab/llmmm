"""Declared datasets and their availability.

A model compartment names its data dependencies in ``ModelSpec.requires``. The
CLI checks them before running and the Azure layer checks them before
submitting, so a missing file fails in a second locally rather than after a job
has queued and started on the cluster.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import PATHS
from .graphs import CHEM_GRAPH, GRAPH_FULL, GRAPH_HELDOUT, GRAPH_TRAIN
from .labels import SUBSTITUTIONS
from .recipes import RECIPE_IDS


@dataclass(frozen=True)
class Dataset:
    key: str
    path: Path
    description: str
    role: str
    """``train`` (a model may read it) or ``label`` (evaluation only)."""

    approx_mb: float = 0.0
    counts: tuple[str, ...] = ()
    """Keys to read from the artefact and substitute into ``description``.

    A description that states a size in prose is a copy of a number that lives
    somewhere else, and it goes stale when the artefact is rebuilt — this one
    claimed 203,504 edges for a graph that had been regenerated with more, and
    printed the old figure in ``im list``. Naming the keys here and formatting
    them at display time means the sentence cannot disagree with the file it
    describes.
    """

    @property
    def available(self) -> bool:
        return self.path.exists()

    def describe(self) -> str:
        """The description with any live counts substituted in.

        Falls back to the unformatted template when the artefact is missing or
        does not carry the named keys, because ``im list`` exists precisely to
        be run before the data is there.
        """
        if not self.counts or not self.available:
            return self.description
        try:
            vals = self._read_counts()
        except Exception:
            return self.description
        try:
            return self.description.format(**vals)
        except (KeyError, IndexError):
            return self.description

    def _read_counts(self) -> dict[str, str]:
        """Length of each named key, formatted with thousands separators.

        Two artefact formats are in use and the caller should not have to care
        which, so the dispatch is on the suffix and the meaning of a "count" is
        the same in both: how many rows the named column has.
        """
        if self.path.suffix == ".parquet":
            import pyarrow.parquet as pq

            n = pq.ParquetFile(self.path).metadata.num_rows
            return {k: f"{n:,}" for k in self.counts}

        import numpy as np

        with np.load(self.path, allow_pickle=False) as z:
            # A zero-dimensional entry is a stored scalar and has no length,
            # so it formats as its value. Everything else is an array and
            # formats as how many rows it has. One rule, and which one applies
            # is a property of the artefact rather than of this table.
            def fmt(v):
                if v.ndim:
                    return f"{len(v):,}"
                x = v.item()
                return f"{x:g}" if isinstance(x, float) else f"{x:,}"

            return {k: fmt(z[k]) for k in self.counts if k in z}


DATASETS: dict[str, Dataset] = {
    d.key: d for d in [
        Dataset("ii_graph", PATHS.graphs / GRAPH_FULL,
                "Ingredient-ingredient NPMI graph, all {src} edges "
                "over {n_recipes} recipes",
                "train", 2.4, counts=("src", "n_recipes")),
        Dataset("ii_graph_train", PATHS.graphs / GRAPH_TRAIN,
                "Edge-split partner of ii_graph_heldout, {src} edges over "
                "{n_recipes} recipes — compare that recipe count against "
                "ii_graph before using it",
                "train", 2.2, counts=("src", "n_recipes")),
        Dataset("ii_graph_heldout", PATHS.graphs / GRAPH_HELDOUT,
                "Edge-split labels for M4: {src} edges, fraction {fraction}, "
                "never a training input",
                "label", 0.05, counts=("src", "fraction")),
        Dataset("chem_graph", PATHS.graphs / CHEM_GRAPH,
                "Ingredient-compound incidence, {src} links across 15 "
                "compound classes",
                "train", 0.08, counts=("src",)),
        Dataset("recipes", PATHS.recipes / RECIPE_IDS,
                "{lang} tokenised recipes with language and source metadata",
                "train", 40.7, counts=("lang",)),
        Dataset("substitutions", PATHS.catalog / SUBSTITUTIONS,
                "{rows} human-voted substitution pairs — labels for M2/M3",
                "label", 1.9, counts=("rows",)),
    ]
}


def check_available(keys) -> list[str]:
    """Return the keys that are declared but missing."""
    missing = []
    for k in keys:
        if k not in DATASETS:
            raise KeyError(f"undeclared dataset {k!r}; "
                           f"add it to data/registry.py")
        if not DATASETS[k].available:
            missing.append(k)
    return missing


def describe() -> str:
    rows = ["  key                 role    size      status  description"]
    for d in DATASETS.values():
        status = "ok" if d.available else "MISSING"
        rows.append(f"  {d.key:<20}{d.role:<8}{d.approx_mb:>6.1f}MB  "
                    f"{status:<8}{d.describe()}")
    return "\n".join(rows)
