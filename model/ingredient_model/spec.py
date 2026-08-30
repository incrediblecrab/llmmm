"""The contract between the core library and a model compartment.

A compartment supplies a :class:`ModelSpec`; the core library supplies loading,
evaluation, artefact storage and Azure submission. Neither side imports the
other's internals, so a new model type is added by writing one folder and
touching nothing else.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


@dataclass
class TrainContext:
    """Everything a trainer is allowed to depend on.

    Passing a context rather than letting trainers read globals is what keeps a
    run reproducible: the split, the seed and the output directory are all
    recorded in the manifest because they had to travel through here.
    """

    graph: str
    """Ingredient-ingredient graph to train on — always a training split, never
    the full graph, or link prediction scores memorisation."""

    seed: int
    out_dir: Path
    device: str = "cpu"
    params: Mapping[str, Any] = field(default_factory=dict)
    corpus: str | None = None
    """Recipe corpus to train on, for models that read recipes. ``None`` means
    the full corpus, which is only sound when nothing is scored against
    held-out edges."""

    split: str = "edge-holdout"

    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)


@dataclass
class TrainResult:
    """What every trainer returns, regardless of family.

    The embedding matrix is the common currency: a random-walk model, a matrix
    factorisation and a transformer all reduce to one ``(n_vocab, d)`` array, so
    a single evaluation harness scores all of them on identical terms.
    """

    embedding: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    extra_arrays: dict[str, np.ndarray] = field(default_factory=dict)
    """Family-specific artefacts (item biases, context tables, attention
    weights). Kept beside the embedding but never required by the harness."""

    scorer: "CompletionScorer | None" = None
    """Optional native completion scorer, ``(contexts) -> (m, n_vocab)``.

    Some families are not *only* an embedding. A masked-set transformer scores a
    candidate against an encoded context; an item-item autoencoder scores it
    against a full weight matrix. Ranking those by cosine between an exported
    table and a summed context measures a shadow of the model rather than the
    model, and would make a conditional model look worse than the popularity
    baseline for reasons that have nothing to do with what it learned.

    When present, M6 is reported twice: once through the embedding, which keeps
    every family on identical terms, and once natively, which is what the model
    would actually serve. Neither number replaces the other.
    """

    def __post_init__(self) -> None:
        if self.embedding.ndim != 2:
            raise ValueError(f"embedding must be 2-D, got {self.embedding.shape}")
        if not np.isfinite(self.embedding).all():
            raise ValueError("embedding contains NaN or inf")


TrainFn = Callable[[TrainContext], TrainResult]
CompletionScorer = Callable[[np.ndarray], np.ndarray]
"""``(m, k) int64 context ids -> (m, n_vocab) float scores``, higher is better."""


@dataclass(frozen=True)
class ModelSpec:
    """A registered, runnable model type."""

    name: str
    family: str
    """The compartment it lives in — the folder name under ``models/``."""

    train: TrainFn
    description: str = ""
    defaults: Mapping[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    """Data dependencies, checked before a job is submitted rather than after
    it has queued for ten minutes. Values are keys of
    ``ingredient_model.data.registry.DATASETS``."""

    cost_hint: str = "cheap"
    """Rough compute class — ``cheap`` (seconds), ``moderate`` (minutes),
    ``heavy`` (hours). Drives Azure fan-out and budget estimates."""

    def resolved_params(self, overrides: Mapping[str, Any] | None = None) -> dict:
        return {**self.defaults, **(overrides or {})}

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "family": self.family,
            "description": self.description,
            "defaults": dict(self.defaults),
            "tags": list(self.tags),
            "requires": list(self.requires),
            "cost_hint": self.cost_hint,
        }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_default))


def _default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")
