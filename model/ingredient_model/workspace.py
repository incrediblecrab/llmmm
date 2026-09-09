"""Explicit workspace choices and a verified entry point for current training."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .config import REPO


@dataclass(frozen=True)
class Workspace:
    benchmark_sweep: str
    training_sweep: str
    training_experiment: Path
    default_embedding_run: str


def load_workspace(model_root: Path = REPO) -> Workspace:
    path = model_root / "workspace.json"
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or type(data.get("version")) is not int:
        raise ValueError(f"{path}: missing integer workspace version")
    if data["version"] != 1:
        raise ValueError(f"{path}: unsupported workspace version {data['version']}")

    def relative(key: str) -> str:
        value = data.get(key)
        if (not isinstance(value, str) or not value
                or "\\" in value or ":" in value
                or Path(value).is_absolute()
                or any(part in (".", "..") for part in value.split("/"))):
            raise ValueError(f"{path}: {key} must be a relative workspace path")
        return value

    return Workspace(
        benchmark_sweep=relative("benchmark_sweep"),
        training_sweep=relative("training_sweep"),
        training_experiment=model_root / relative("training_experiment"),
        default_embedding_run=relative("default_embedding_run"),
    )


def verify_training_corpus(data_root: Path, expected_generation: str) -> dict:
    marker_path = data_root / "GENERATION.json"
    marker = json.loads(marker_path.read_text())
    if not isinstance(marker, dict) or marker.get("generation") != expected_generation:
        raise ValueError(
            f"{marker_path}: training requires generation {expected_generation!r}")
    if marker.get("corpus") != "recipe_ids.npz":
        raise ValueError(f"{marker_path}: expected canonical recipe_ids.npz")
    expected = marker.get("sha256")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"{marker_path}: missing canonical corpus SHA-256")
    corpus = data_root / "recipes" / "recipe_ids.npz"
    digest = hashlib.sha256()
    with corpus.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(
            f"{corpus}: checksum differs from GENERATION.json; restore the "
            "declared data before training")
    return marker


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from .experiments import _load, run_experiment

    workspace = load_workspace()
    experiment = _load(workspace.training_experiment)
    if experiment.get("name") != workspace.training_sweep:
        raise ValueError("workspace.json and the training experiment name disagree")
    generation = experiment.get("corpus_generation")
    if not isinstance(generation, str) or not generation:
        raise ValueError("the current training experiment must declare corpus_generation")
    return run_experiment(workspace.training_experiment, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
