"""Run artefacts: what a training run leaves behind, and how it is read back.

One directory per run, holding the embedding, a manifest and (once scored) its
metrics. The manifest records the graph split, seed, resolved parameters and
library versions, because a number without the conditions that produced it
cannot be compared against anything.
"""
from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import PATHS
from .spec import ModelSpec, TrainResult, write_json

EMBEDDING = "embedding.npy"
MANIFEST = "manifest.json"
METRICS = "metrics.json"


@dataclass
class Manifest:
    run_id: str
    model: str
    family: str
    graph: str
    seed: int
    params: dict[str, Any]
    created: str
    duration_s: float
    shape: tuple[int, int]
    environment: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def load(run_dir: Path) -> "Manifest":
        d = json.loads((run_dir / MANIFEST).read_text())
        d["shape"] = tuple(d["shape"])
        return Manifest(**d)


def _environment() -> dict[str, str]:
    env = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "scipy", "torch"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        env["git"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        pass
    # Which corpus produced this run. Two leaderboards built on different
    # corpus generations are not comparable, and without this stamp they are
    # indistinguishable once the terminal scrollback is gone.
    try:
        from .config import corpus_generation
        env["corpus_generation"] = str(corpus_generation().get("generation"))
    except Exception:
        pass
    return env


def resolve_run(ref: str, *, root: Path | None = None) -> Path:
    """Find a run directory by id, path, or sweep-relative id.

    Sweeps group their runs under ``runs/<experiment>/<run-id>``, but a model
    that consumes another model's output — ``concat``, ``residual``,
    ``text-aligned`` — is given a bare run id. Looking only in ``runs/<id>``
    means those models can never see a sibling produced by the same sweep, which
    fails as a confusing FileNotFoundError at the end of a long batch rather
    than in the pre-flight check.

    Search order is most-specific first: an explicit path, then the sweep's own
    directory, then the top level, then anywhere below ``runs/``.
    """
    p = Path(ref)
    if p.exists() and (p / MANIFEST).exists():
        return p
    candidates = []
    if root is not None:
        candidates.append(Path(root) / ref)
    candidates.append(PATHS.run_dir(ref))
    for c in candidates:
        if (c / MANIFEST).exists():
            return c
    matches = sorted(m.parent for m in PATHS.runs.rglob(f"{ref}/{MANIFEST}"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(
            f"run {ref!r} is ambiguous — {len(matches)} matches: "
            + ", ".join(str(m.relative_to(PATHS.runs)) for m in matches[:5]))
    raise FileNotFoundError(f"no run {ref!r} under {PATHS.runs}")


def save_run(run_id: str, spec: ModelSpec, result: TrainResult, *, graph: str,
             seed: int, params: dict, duration_s: float,
             out_dir: Path | None = None) -> Path:
    d = out_dir or PATHS.run_dir(run_id)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / EMBEDDING, result.embedding.astype(np.float32))
    for key, arr in result.extra_arrays.items():
        np.save(d / f"{key}.npy", arr)
    write_json(d / MANIFEST, asdict(Manifest(
        run_id=run_id, model=spec.name, family=spec.family, graph=graph,
        seed=seed, params=params,
        created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), duration_s=duration_s,
        shape=tuple(result.embedding.shape), environment=_environment(),
        metadata=result.metadata)))
    return d


def load_embedding(run_dir: Path) -> np.ndarray:
    return np.load(run_dir / EMBEDDING)


def save_metrics(run_dir: Path, metrics: dict) -> None:
    write_json(run_dir / METRICS, metrics)


def load_metrics(run_dir: Path) -> dict | None:
    p = run_dir / METRICS
    return json.loads(p.read_text()) if p.exists() else None


def iter_runs(root: Path | None = None):
    """Every completed run, newest last. A directory without a manifest is a
    partial or crashed run and is skipped rather than half-reported.

    The walk is recursive because sweeps group their runs under a named
    subdirectory. A flat scan silently omitted every swept run, so a sweep could
    report six successes and leave the leaderboard unchanged."""
    base = root or PATHS.runs
    if not base.exists():
        return
    for manifest in sorted(base.rglob(MANIFEST)):
        d = manifest.parent
        if (d / EMBEDDING).exists():
            yield d
