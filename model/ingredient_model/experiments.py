"""Declarative sweeps — the experiment is a file, not a shell command.

A sweep typed into a terminal leaves no record of what was run. Six weeks later
the results directory holds forty runs and the only account of how they were
produced is shell history on one machine. Declaring the sweep in a file makes
the experiment an artefact that is versioned, diffable and re-runnable, and it
means the *plan* can be reviewed before the compute is spent — which matters
when compute is a fixed $150 rather than a tap.

Every sweep is checked against the leakage rule before anything runs. Finding
out that half a grid was invalid after paying for it is the specific failure
this is designed to prevent.
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifacts import save_metrics, save_run
from .config import PATHS
from .data.splits import (DEFAULT_SPLIT, LeakageError, check_leakage,
                          get_split, held_out_recipes)
from .eval.harness import build_context, evaluate
from .eval.report import leaderboard, render_one
from .registry import discover, get
from .spec import TrainContext


def _load(path: Path) -> dict:
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ModuleNotFoundError as e:
            raise SystemExit(
                "PyYAML is needed for .yaml sweeps — `pip install pyyaml`, "
                "or write the sweep as .json") from e
        return yaml.safe_load(text)
    return json.loads(text)


@dataclass
class Trial:
    model: str
    split: str
    seed: int
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        bits = "-".join(f"{k}{v}" for k, v in sorted(self.params.items()))
        return "-".join(x for x in (self.model, self.split, bits,
                                    f"s{self.seed}") if x)


def expand(spec: dict) -> list[Trial]:
    """Turn a sweep declaration into the list of runs it implies.

    ``grid`` is a cross product; ``params`` are fixed for every trial. Seeds are
    a list rather than a count because a sweep must name the seeds it used —
    "three seeds" is not reproducible, ``[0, 1, 2]`` is.
    """
    models = spec.get("models") or [spec["model"]]
    seeds = spec.get("seeds", [0])
    splits = spec.get("splits") or [spec.get("split", DEFAULT_SPLIT)]
    fixed = dict(spec.get("params", {}))
    grid = spec.get("grid", {})
    keys = sorted(grid)
    combos = [dict(zip(keys, vals))
              for vals in itertools.product(*(grid[k] for k in keys))] or [{}]

    return [Trial(model=m, split=s, seed=int(sd), params={**fixed, **combo})
            for m in models for s in splits for sd in seeds for combo in combos]


def check(trials: list[Trial]) -> list[str]:
    """Every reason the sweep would be refused, found before anything runs."""
    problems = []
    for t in trials:
        try:
            spec = get(t.model)
        except KeyError:
            problems.append(f"{t.run_id}: unknown model {t.model!r}")
            continue
        try:
            check_leakage(get_split(t.split), spec.requires)
        except (LeakageError, KeyError) as e:
            problems.append(f"{t.run_id}: {e}")
    return problems


def run_experiment(path: Path, dry_run: bool = False) -> int:
    discover()
    spec_doc = _load(Path(path))
    name = spec_doc.get("name", Path(path).stem)
    trials = expand(spec_doc)
    out_root = PATHS.runs / name if spec_doc.get("group", True) else PATHS.runs

    print(f"experiment {name}  —  {len(trials)} trials")
    if desc := spec_doc.get("description"):
        print(f"  {desc}")
    print()

    problems = check(trials)
    if problems:
        print("refusing to run — the plan is invalid:")
        for p in problems:
            print(f"  {p}")
        return 2

    for t in trials:
        print(f"  {t.run_id}")
    if dry_run:
        print(f"\ndry run — nothing executed. Drop --dry-run to spend the compute.")
        return 0

    print()
    done, failed = [], []
    # Loaded once for the whole sweep: it costs a full-corpus read, and every
    # trial on a given split must see the identical test recipes or the
    # leaderboard is comparing runs against different exams.
    completion_cache: dict[str, object] = {}
    for i, t in enumerate(trials, 1):
        mspec = get(t.model)
        split = get_split(t.split)
        if t.split not in completion_cache:
            completion_cache[t.split] = held_out_recipes(t.split)
        completion_corpus = completion_cache[t.split]
        print(f"[{i}/{len(trials)}] {t.run_id}", flush=True)
        try:
            t0 = time.time()
            out_dir = out_root / t.run_id
            ctx = TrainContext(params=mspec.resolved_params(t.params),
                               seed=t.seed, graph=split.graph, out_dir=out_dir,
                               corpus=split.corpus, split=split.name)
            result = mspec.train(ctx)
            d = save_run(t.run_id, mspec, result, graph=split.graph,
                         seed=t.seed, params={**t.params, "split": split.name},
                         duration_s=time.time() - t0, out_dir=out_dir)
            metrics = evaluate(result.embedding, build_context(split.name),
                               completion_corpus=completion_corpus,
                               scorer=result.scorer)
            save_metrics(d, metrics)
            print(render_one(t.run_id, metrics))
            done.append(t.run_id)
        except Exception as e:  # one bad trial must not void the rest
            # A sweep is a batch: aborting on the first failure throws away the
            # trials that already succeeded and the compute they cost.
            print(f"  FAILED: {type(e).__name__}: {e}")
            failed.append((t.run_id, f"{type(e).__name__}: {e}"))
        print()

    print(f"{len(done)} succeeded, {len(failed)} failed")
    for rid, err in failed:
        print(f"  {rid}: {err}")
    if done:
        print()
        print(leaderboard(root=out_root))
    return 1 if failed else 0
