"""Model registry and compartment discovery.

Compartments are found by walking ``models/`` and importing each package, so a
new model type becomes available by creating a folder — there is no central list
to update and therefore no way for the list to drift from what exists on disk.

    from ingredient_model.registry import register

    @register(name="sgns-cooc", family="sgns_walk", cost_hint="moderate")
    def train(ctx: TrainContext) -> TrainResult:
        ...
"""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from .spec import ModelSpec, TrainFn

_REGISTRY: dict[str, ModelSpec] = {}
_DISCOVERED = False

COMPARTMENTS_PACKAGE = "models"


def register(
    name: str,
    family: str,
    description: str = "",
    defaults: Mapping[str, Any] | None = None,
    tags: tuple[str, ...] = (),
    requires: tuple[str, ...] = (),
    cost_hint: str = "cheap",
) -> Callable[[TrainFn], TrainFn]:
    """Decorate a training function to make it runnable by name.

    The function is returned unchanged, so a compartment stays directly
    importable and testable without going through the registry.
    """

    def deco(fn: TrainFn) -> TrainFn:
        if name in _REGISTRY:
            raise ValueError(
                f"duplicate model name {name!r} "
                f"(already registered by family {_REGISTRY[name].family!r})")
        _REGISTRY[name] = ModelSpec(
            name=name, family=family, train=fn,
            description=description or (fn.__doc__ or "").strip().split("\n")[0],
            defaults=dict(defaults or {}), tags=tuple(tags),
            requires=tuple(requires), cost_hint=cost_hint)
        return fn

    return deco


def discover(force: bool = False) -> None:
    """Import every compartment package under ``models/``.

    Import errors are deliberately not swallowed. A compartment whose
    dependencies are missing should fail loudly at discovery, not silently
    vanish from the registry and turn into a confusing "unknown model" later.
    """
    global _DISCOVERED
    if _DISCOVERED and not force:
        return
    pkg = importlib.import_module(COMPARTMENTS_PACKAGE)
    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.ispkg and not mod.name.startswith("_"):
            importlib.import_module(f"{COMPARTMENTS_PACKAGE}.{mod.name}")
    _DISCOVERED = True


def get(name: str) -> ModelSpec:
    discover()
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown model {name!r}. Available: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[name]


def all_specs() -> list[ModelSpec]:
    discover()
    return sorted(_REGISTRY.values(), key=lambda s: (s.family, s.name))


def families() -> dict[str, list[ModelSpec]]:
    out: dict[str, list[ModelSpec]] = {}
    for spec in all_specs():
        out.setdefault(spec.family, []).append(spec)
    return out


def iter_names() -> Iterator[str]:
    discover()
    yield from sorted(_REGISTRY)


def compartment_dir(family: str) -> Path:
    pkg = importlib.import_module(f"{COMPARTMENTS_PACKAGE}.{family}")
    return Path(pkg.__file__).resolve().parent
