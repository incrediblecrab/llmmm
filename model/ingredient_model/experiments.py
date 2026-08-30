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
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifacts import METRICS, save_metrics, save_run
from .config import PATHS, corpus_generation
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
    label: str | None = None
    """Optional short name, for trials whose parameters are run ids.

    A blend's inputs are its parameters, so the derived id would embed two other
    full run ids and become unreadable. The split and seed stay in the name
    either way — a run id that does not say which protocol produced it is the
    thing that makes two leaderboards impossible to tell apart later.
    """

    @property
    def run_id(self) -> str:
        if self.label:
            return f"{self.label}-{self.split}-s{self.seed}"
        bits = "-".join(f"{k}{v}" for k, v in sorted(self.params.items()))
        return "-".join(x for x in (self.model, self.split, bits,
                                    f"s{self.seed}") if x)


def expand(spec: dict) -> list[Trial]:
    """Turn a sweep declaration into the list of runs it implies.

    ``grid`` is a cross product; ``params`` are fixed for every trial. Seeds are
    a list rather than a count because a sweep must name the seeds it used —
    "three seeds" is not reproducible, ``[0, 1, 2]`` is.

    A model entry may be a bare name, or a mapping carrying parameters that
    apply to that model alone::

        models:
          - svd-ppmi
          - name: residual
            id: residual-svdppmi-ease
            params: {base: svd-ppmi-..., correction: ease-...}

    Per-model parameters exist because the blends cannot share the sweep-wide
    block: each one names *different* inputs. Without this, running every model
    means one file per blend and no single command that reproduces the
    leaderboard.
    """
    models = spec.get("models") or [spec["model"]]
    seeds = spec.get("seeds", [0])
    splits = spec.get("splits") or [spec.get("split", DEFAULT_SPLIT)]
    fixed = dict(spec.get("params", {}))
    grid = spec.get("grid", {})
    keys = sorted(grid)
    combos = [dict(zip(keys, vals))
              for vals in itertools.product(*(grid[k] for k in keys))] or [{}]

    out = []
    for m in models:
        if isinstance(m, dict):
            mname, mparams = m["name"], dict(m.get("params", {}))
            label = m.get("id")
        else:
            mname, mparams, label = m, {}, None
        for s in splits:
            for sd in seeds:
                for combo in combos:
                    out.append(Trial(model=mname, split=s, seed=int(sd),
                                     params={**fixed, **mparams, **combo},
                                     label=label))
    return out


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


def order(trials: list[Trial]) -> tuple[list[Trial], dict[str, set[str]]]:
    """Sort so a trial runs after any sibling whose output it consumes.

    Dependencies are *derived*, not declared. A blend names its inputs by run id
    in its own parameters — ``base=svd-ppmi-recipe-holdout-s0`` — so the sweep
    file already contains the edge and a separate ``needs:`` list would be a
    second copy of it that can go stale. Scanning the parameters means the
    ordering cannot disagree with what the model will actually load.

    A parameter naming a run that is *not* in this sweep is left alone: it is
    presumed to exist already, and ``resolve_run`` will say so if it does not.

    Declaration order is preserved among independent trials, so a sweep that
    needs no ordering runs exactly as written.
    """
    ids = {t.run_id for t in trials}
    deps: dict[str, set[str]] = {}
    for t in trials:
        d = set()
        for v in t.params.values():
            for token in str(v).split(","):
                token = token.strip()
                if token in ids and token != t.run_id:
                    d.add(token)
        deps[t.run_id] = d

    by_id = {t.run_id: t for t in trials}
    out: list[Trial] = []
    done: set[str] = set()
    remaining = [t.run_id for t in trials]
    while remaining:
        ready = [r for r in remaining if deps[r] <= done]
        if not ready:
            # A cycle. Run them in declaration order and let the loading model
            # produce the error — refusing here would be a worse message than
            # the FileNotFoundError naming the specific run that is missing.
            ready = remaining
        for r in ready:
            out.append(by_id[r])
            done.add(r)
        remaining = [r for r in remaining if r not in done]
    return out, deps


class _Timeout(Exception):
    pass


class _deadline:
    """Abort a trial that has stopped making progress.

    A sweep is unattended: without this, one model that fails to converge holds
    the remaining trials hostage for as long as the terminal stays open. SIGALRM
    only interrupts between bytecodes, so a trial wedged inside a single long
    BLAS call will overrun its deadline — this bounds the common case, not
    every case, and the journal is what makes the uncommon one recoverable.
    """

    def __init__(self, seconds: int | None):
        self.seconds = int(seconds or 0)
        self.armed = False

    def __enter__(self):
        if self.seconds > 0 and hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, self._fire)
            signal.alarm(self.seconds)
            self.armed = True
        return self

    def _fire(self, *_):
        raise _Timeout(f"exceeded {self.seconds}s")

    def __exit__(self, *_):
        if self.armed:
            signal.alarm(0)
        return False


def _journal(path: Path, event: dict) -> None:
    """Append one line describing what just happened.

    stdout is not a record — it dies with the terminal, and a sweep long enough
    to need resuming is exactly the kind that gets left running overnight in a
    window that later gets closed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                             **event}) + "\n")


def run_experiment(path: Path, dry_run: bool = False, *, resume: bool = True,
                   timeout_s: int | None = None,
                   only: tuple[str, ...] = ()) -> int:
    discover()
    spec_doc = _load(Path(path))
    name = spec_doc.get("name", Path(path).stem)
    trials = expand(spec_doc)
    if only:
        trials = [t for t in trials if t.model in only]
    out_root = PATHS.runs / name if spec_doc.get("group", True) else PATHS.runs
    timeout_s = timeout_s if timeout_s is not None else spec_doc.get("timeout_s")

    gen = corpus_generation().get("generation", "unknown")
    print(f"experiment {name}  —  {len(trials)} trials  —  corpus {gen}")
    if desc := spec_doc.get("description"):
        print(f"  {desc}")
    print()

    problems = check(trials)
    if problems:
        print("refusing to run — the plan is invalid:")
        for p in problems:
            print(f"  {p}")
        return 2

    trials, deps = order(trials)

    # Resume is the default because the alternative is worse: re-running a
    # completed sweep silently spends hours reproducing numbers that are already
    # on disk, and the failure it is meant to recover from — one trial in twenty
    # crashing — is the common one.
    skipped = []
    if resume:
        pending = []
        for t in trials:
            if (out_root / t.run_id / METRICS).exists():
                skipped.append(t.run_id)
            else:
                pending.append(t)
        trials = pending

    for t in trials:
        d = sorted(deps.get(t.run_id, ()))
        print(f"  {t.run_id}" + (f"   after {', '.join(d)}" if d else ""))
    if skipped:
        print(f"\n  {len(skipped)} already complete, skipping "
              f"(--force to rerun): {', '.join(skipped[:4])}"
              + (" …" if len(skipped) > 4 else ""))
    if dry_run:
        print(f"\ndry run — nothing executed. Drop --dry-run to spend the compute.")
        return 0
    if not trials:
        print("\nnothing to do.")
        print(leaderboard(root=out_root))
        return 0

    print()
    jpath = out_root / "journal.jsonl"
    _journal(jpath, {"event": "sweep_start", "experiment": name,
                     "corpus_generation": gen, "trials": len(trials),
                     "skipped": len(skipped)})

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
        t0 = time.time()
        try:
            out_dir = out_root / t.run_id
            ctx = TrainContext(params=mspec.resolved_params(t.params),
                               seed=t.seed, graph=split.graph, out_dir=out_dir,
                               corpus=split.corpus, split=split.name)
            with _deadline(timeout_s):
                result = mspec.train(ctx)
                d = save_run(t.run_id, mspec, result, graph=split.graph,
                             seed=t.seed,
                             params={**t.params, "split": split.name},
                             duration_s=time.time() - t0, out_dir=out_dir)
                metrics = evaluate(result.embedding, build_context(split.name),
                                   completion_corpus=completion_corpus,
                                   scorer=result.scorer)
            save_metrics(d, metrics)
            print(render_one(t.run_id, metrics))
            done.append(t.run_id)
            _journal(jpath, {"event": "trial_ok", "run_id": t.run_id,
                             "model": t.model, "split": t.split,
                             "seconds": round(time.time() - t0, 1)})
        except Exception as e:  # one bad trial must not void the rest
            # A sweep is a batch: aborting on the first failure throws away the
            # trials that already succeeded and the compute they cost.
            err = f"{type(e).__name__}: {e}"
            print(f"  FAILED: {err}")
            failed.append((t.run_id, err))
            _journal(jpath, {"event": "trial_failed", "run_id": t.run_id,
                             "model": t.model, "split": t.split,
                             "error": err,
                             "seconds": round(time.time() - t0, 1)})
        print()

    print(f"{len(done)} succeeded, {len(failed)} failed"
          + (f", {len(skipped)} already complete" if skipped else ""))
    for rid, err in failed:
        print(f"  {rid}: {err}")
    _journal(jpath, {"event": "sweep_end", "ok": len(done),
                     "failed": len(failed), "skipped": len(skipped)})
    if done or skipped:
        print()
        print(leaderboard(root=out_root))
    return 1 if failed else 0
