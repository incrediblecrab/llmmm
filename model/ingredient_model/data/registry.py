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

    @property
    def available(self) -> bool:
        return self.path.exists()


DATASETS: dict[str, Dataset] = {
    d.key: d for d in [
        Dataset("ii_graph", PATHS.graphs / GRAPH_FULL,
                "Ingredient-ingredient NPMI graph, all 203,504 edges",
                "train", 2.4),
        Dataset("ii_graph_train", PATHS.graphs / GRAPH_TRAIN,
                "The same graph with 10% of edges removed for link prediction",
                "train", 2.2),
        Dataset("ii_graph_heldout", PATHS.graphs / GRAPH_HELDOUT,
                "The removed 10% — labels for M4, never a training input",
                "label", 0.05),
        Dataset("chem_graph", PATHS.graphs / CHEM_GRAPH,
                "Ingredient-compound incidence with 15 compound classes",
                "train", 0.08),
        Dataset("recipes", PATHS.recipes / RECIPE_IDS,
                "4.6M tokenised recipes with language and source metadata",
                "train", 40.7),
        Dataset("substitutions", PATHS.catalog / SUBSTITUTIONS,
                "210,612 human-voted substitution pairs — labels for M2/M3",
                "label", 1.9),
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
                    f"{status:<8}{d.description}")
    return "\n".join(rows)
