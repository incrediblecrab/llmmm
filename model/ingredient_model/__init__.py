"""Ingredient representation learning — core library.

The package supplies data access, evaluation, artefact storage and Azure
submission. Model types live in their own compartments under ``models/`` and
plug in through :func:`ingredient_model.registry.register`.
"""
from importlib import import_module
from typing import TYPE_CHECKING

from .config import PATHS, SEED

if TYPE_CHECKING:
    from .registry import all_specs, families, get, register
    from .spec import ModelSpec, TrainContext, TrainResult

__version__ = "0.1.0"

__all__ = [
    "PATHS", "SEED", "ModelSpec", "TrainContext", "TrainResult",
    "register", "get", "all_specs", "families", "__version__",
]


def __getattr__(name: str) -> object:
    # Recovery must import before NumPy or the training dependencies exist.
    if name in {"register", "get", "all_specs", "families"}:
        module = ".registry"
    elif name in {"ModelSpec", "TrainContext", "TrainResult"}:
        module = ".spec"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
