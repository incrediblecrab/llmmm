"""Run artefacts: what a training run leaves behind, and how it is read back.

One directory per run, holding the embedding, a manifest and (once scored) its
metrics. The manifest records the canonical corpus digest, graph split, seed,
resolved parameters and library versions, because a number without the
conditions that produced it cannot be compared against anything.
"""
from __future__ import annotations

import json
import platform
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from numpy.typing import NDArray

from .config import PATHS
from .spec import CompletionScorer, ModelSpec, TrainResult, write_json

EMBEDDING = "embedding.npy"
MANIFEST = "manifest.json"
METRICS = "metrics.json"


def unevaluated_metrics(split: str) -> dict:
    """Completion marker for a trained checkpoint without evaluation scores."""
    return {"split": split, "evaluation_status": "not_run"}


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
    # A generation label can survive a rebuild. Record the promoted digest too;
    # training preflight verifies the bytes, so saving need not hash them again.
    from .config import corpus_generation

    corpus = corpus_generation()
    env["corpus_generation"] = str(corpus.get("generation"))
    if "sha256" in corpus:
        digest = corpus["sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(
                f"{PATHS.generation_file}: invalid canonical corpus SHA-256")
        env["corpus_sha256"] = digest
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


def _load_native_array(path: Path) -> NDArray[np.floating]:
    try:
        array = np.load(path, allow_pickle=False)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"native scorer requires {path}") from e
    except (OSError, ValueError, EOFError) as e:
        raise ValueError(f"cannot load native scorer asset {path}: {e}") from e
    if isinstance(array, np.lib.npyio.NpzFile):
        array.close()
        raise ValueError(f"native scorer asset {path} must be a single array")
    if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise ValueError(
            f"native scorer asset {path} must contain finite floating-point values")
    return array


def load_native_scorer(run_dir: Path, n_vocab: int) -> CompletionScorer | None:
    """Restore the full predictor recorded by a run, not its embedding proxy.

    Model identity comes from the manifest, not the presence of optional files:
    a missing or corrupt native asset must fail rather than silently change the
    evaluated model. ``n_vocab`` is the caller's vocabulary size and must agree
    with the run. Only masked-set restoration imports torch; EASE and
    embedding-only runs do not require it.
    """
    man = Manifest.load(run_dir)
    if len(man.shape) != 2 or min(man.shape) <= 0:
        raise ValueError(
            f"invalid embedding shape {man.shape} in {run_dir / MANIFEST}")
    if man.shape[0] != n_vocab:
        raise ValueError(
            f"run {man.run_id!r} has {man.shape[0]} vocabulary rows, expected {n_vocab}")

    if man.model == "ease":
        path = run_dir / "item_scores.npy"
        B = _load_native_array(path)
        if B.shape != (n_vocab, n_vocab):
            raise ValueError(
                f"native scorer asset {path} has shape {B.shape}, "
                f"expected {(n_vocab, n_vocab)}")

        def scorer(context_ids: NDArray[np.int64]) -> NDArray[np.floating]:
            return B[context_ids].sum(axis=1)

        return scorer

    if man.model == "masked-set":
        token_path = run_dir / "state__tok__weight.npy"
        tokens = _load_native_array(token_path)
        expected = (n_vocab + 1, man.shape[1])
        if tokens.shape != expected:
            raise ValueError(
                f"native scorer asset {token_path} has shape {tokens.shape}, "
                f"expected {expected}")
        for path in sorted(run_dir.glob("state__*.npy")):
            if path != token_path:
                _load_native_array(path)

        from models.set_transformer.train import restore

        try:
            return restore(run_dir, n_vocab)
        except RuntimeError as e:
            raise ValueError(
                f"cannot restore native scorer for {man.run_id!r} "
                f"in {run_dir}: {e}") from e

    return None


def save_metrics(run_dir: Path, metrics: dict) -> None:
    write_json(run_dir / METRICS, metrics)


def load_metrics(run_dir: Path) -> dict | None:
    p = run_dir / METRICS
    return json.loads(p.read_text()) if p.exists() else None


def iter_runs(root: Path | None = None, *,
              require_embedding: bool = True) -> Iterator[Path]:
    """Manifest-backed runs in stable path order, including nested sweeps.

    Weight consumers require an embedding by default. Reports and listings may
    opt into metadata-only runs, as on a git-only checkout. Directories without
    a manifest are partial or crashed runs and are always skipped.
    """
    base = root or PATHS.runs
    if not base.exists():
        return
    for manifest in sorted(base.rglob(MANIFEST)):
        d = manifest.parent
        if not require_embedding or (d / EMBEDDING).exists():
            yield d
