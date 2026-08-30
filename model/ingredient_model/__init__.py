"""Ingredient representation learning — core library.

The package supplies data access, evaluation, artefact storage and Azure
submission. Model types live in their own compartments under ``models/`` and
plug in through :func:`ingredient_model.registry.register`.
"""
from .config import PATHS, SEED
from .registry import all_specs, families, get, register
from .spec import ModelSpec, TrainContext, TrainResult

__version__ = "0.1.0"

__all__ = [
    "PATHS", "SEED", "ModelSpec", "TrainContext", "TrainResult",
    "register", "get", "all_specs", "families", "__version__",
]
