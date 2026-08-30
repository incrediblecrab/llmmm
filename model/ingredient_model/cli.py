"""Command line interface.

    im list                             registered models, datasets, splits
    im gate                             Phase 0 negative control — run first
    im train ease --set reg=500         train, score and record one model
    im eval <run-id|path>               re-score an existing embedding
    im report                           leaderboard across every scored run
    im explain tomato basil             the reasoning layer
    im sweep experiments/baseline.yaml  run a declared experiment

Training always scores the model and writes the embedding and its metrics into
one run directory. Keeping them together is deliberate: a metric detached from
the artefact and conditions that produced it cannot be checked later.

``--split`` selects the evaluation protocol and, with it, the training inputs.
Recipe-reading models are *refused* on the edge-level split, because removing
graph edges does not remove the recipes that produced them.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .artifacts import (Manifest, iter_runs, load_embedding, load_metrics,
                        save_metrics, save_run)
from .config import PATHS
from .data.registry import check_available, describe
from .data.splits import (DEFAULT_SPLIT, SPLITS, check_leakage, get_split,
                          held_out_recipes)
from .eval import build_context, control_gate, evaluate, render_one
from .eval.report import collect, report
from .registry import families, get
from .spec import TrainContext


def _parse_set(items: list[str] | None) -> dict:
    """``--set key=value`` with values typed by literal parsing.

    A hyperparameter that silently arrives as a string produces a model that
    trains and is wrong, so the conversion is explicit rather than best-effort.
    """
    out: dict = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = {"true": True, "false": False}.get(v.lower(), v)
    return out


def _resolve_run(ref: str) -> Path:
    p = Path(ref)
    if p.exists():
        return p
    p = PATHS.run_dir(ref)
    if p.exists():
        return p
    raise SystemExit(f"no such run: {ref}")


# ----------------------------------------------------------------- commands
def cmd_list(a) -> int:
    print("MODELS")
    for family, specs in families().items():
        print(f"\n  {family}")
        for s in specs:
            missing = check_available(s.requires)
            flag = "" if not missing else f"   [missing: {', '.join(missing)}]"
            print(f"    {s.name:<14}{s.cost_hint:<10}{s.description}{flag}")
    print(f"\nDATASETS\n{describe()}")
    print("\nSPLITS")
    for s in SPLITS.values():
        ok = "ok" if (PATHS.graphs / s.heldout).exists() else "NOT BUILT"
        print(f"  {s.name:<16}{ok:<12}{s.description}")
    return 0


def cmd_gate(a) -> int:
    g = control_gate(build_context(a.split))
    for k, v in g["results"].items():
        print(render_one(k, v))
    print(f"\n  GATE: {'PASS' if g['passed'] else 'FAIL ' + str(g['failed'])}"
          f"  — random vectors must score ~0.50 on M2 and M4")
    return 0 if g["passed"] else 1


def cmd_train(a) -> int:
    spec = get(a.model)
    split = get_split(a.split)
    missing = check_available(spec.requires)
    if missing:
        raise SystemExit(f"{spec.name} needs missing datasets: {', '.join(missing)}\n"
                         f"  python scripts/import_data.py --from <llmmm-checkout>")
    if not a.allow_leakage:
        check_leakage(split, spec.requires, strict=True)
    else:
        warn = check_leakage(split, spec.requires, strict=False)
        if warn:
            print(f"  !! LEAKAGE ACCEPTED: {warn}")

    params = spec.resolved_params(_parse_set(a.set))
    run_id = a.run_id or f"{spec.name}-{split.name}-s{a.seed}"
    out_dir = Path(a.out) if a.out else PATHS.run_dir(run_id)

    print(f"train {spec.name}  split={split.name}  graph={split.graph}  "
          f"seed={a.seed}  device={a.device}")
    print(f"  params {params}")
    ctx = TrainContext(graph=split.graph, seed=a.seed, out_dir=out_dir,
                       device=a.device, params=params, corpus=split.corpus,
                       split=split.name)
    t0 = time.time()
    result = spec.train(ctx)
    duration = time.time() - t0
    d = save_run(run_id, spec, result, graph=split.graph, seed=a.seed,
                 params={**params, "split": split.name},
                 duration_s=duration, out_dir=out_dir)
    print(f"  trained in {duration:.0f}s -> {d}")

    if not a.no_eval:
        metrics = evaluate(result.embedding, build_context(split.name),
                           completion_corpus=held_out_recipes(split.name, a.n_completion * 4),
                           n_completion=a.n_completion,
                           scorer=result.scorer)
        save_metrics(d, metrics)
        print()
        print(render_one(run_id, metrics))
    return 0


def cmd_eval(a) -> int:
    p = Path(a.target)
    if p.suffix == ".npy":
        W, name, run_dir = np.load(p), p.stem, None
    else:
        run_dir = _resolve_run(a.target)
        man = Manifest.load(run_dir)
        W, name = load_embedding(run_dir), man.run_id
        if a.split is None:
            a.split = man.params.get("split", DEFAULT_SPLIT)
    split_name = a.split or DEFAULT_SPLIT
    metrics = evaluate(W, build_context(split_name), whiten=a.whiten,
                       completion_corpus=held_out_recipes(split_name, a.n_completion * 4),
                       n_completion=a.n_completion)
    print(render_one(name + (" [whitened]" if a.whiten else ""), metrics))
    if run_dir is not None and not a.whiten:
        save_metrics(run_dir, metrics)
    return 0


def cmd_report(a) -> int:
    rows = collect()
    gate = control_gate(build_context(a.split)) if a.gate else None
    text = report(rows, gate)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
        print(f"\nwrote {a.out}")
    return 0


def cmd_runs(a) -> int:
    for d in iter_runs():
        man = Manifest.load(d)
        m = load_metrics(d)
        auc = f"{m['M4_link_auc']:.4f}" if m and m.get("M4_link_auc") else "unscored"
        print(f"  {man.run_id:<40}{man.model:<12}{man.graph:<26}"
              f"{man.duration_s:>7.0f}s  M4 {auc}")
    return 0


def cmd_explain(a) -> int:
    from .reasoning import Reasoner
    r = Reasoner.load(_resolve_run(a.run) if a.run else None)
    print(r.explain_pair(a.a, a.b, top_bridges=a.bridges))
    return 0


def cmd_neighbors(a) -> int:
    from .reasoning import Reasoner
    r = Reasoner.load(_resolve_run(a.run))
    for name, score in r.neighbors(a.ingredient, k=a.k, dedup=not a.raw):
        print(f"  {score:6.3f}  {name}")
    return 0


def cmd_sweep(a) -> int:
    from .experiments import run_experiment
    return run_experiment(Path(a.spec), dry_run=a.dry_run)


def cmd_recipes(a) -> int:
    from .data.browse import breakdown, coverage, sample, source_records, view
    from .data import text as textmod

    def show(v):
        print(v)
        if not a.text:
            return
        try:
            t = textmod.text_of(v.index)
        except FileNotFoundError:
            print("      (no text index — build it with `make text`)")
            return
        if t.title:
            print(f"      title  {t.title}")
        if t.url:
            print(f"      url    {t.url}")
        if t.raw_ingredients:
            for line in t.raw_ingredients.split("\x1f")[:12]:
                print(f"      ·  {line}")
        if t.steps:
            steps = [s for s in t.steps.split("\x1f") if s.strip()]
            for i, s in enumerate(steps[:4], 1):
                print(f"      {i}. {s[:150]}")
            if len(steps) > 4:
                print(f"      … {len(steps) - 4} more steps")

    if a.verify:
        c = coverage()
        width = max(len(k) for k in c)
        for k, v in c.items():
            shown = (("yes" if v else "NO") if isinstance(v, bool) else
                     f"{v:,}" if isinstance(v, int) else
                     f"{v:.2f}" if isinstance(v, float) else str(v))
            print(f"  {k:<{width}}  {shown}")
        print()
        print(f"  {'source':<24}{'lang':<10}{'recipes':>12}{'ing/rec':>9}")
        for src, lang, n, mean in breakdown():
            print(f"  {src:<24}{lang[:9]:<10}{n:>12,}{mean:>9.1f}")
        return 0 if c["all_viewable"] else 1

    if a.raw_source:
        recs = source_records(a.raw_source, limit=a.n)
        if not recs:
            print(f"no raw records for {a.raw_source!r} "
                  f"(original corpus tree not available)")
            return 1
        for key, items in recs:
            print(f"  {key}: {items}")
        return 0

    if a.index is not None:
        show(view(a.index))
        return 0

    rows = sample(a.n, source=a.source, lang=a.lang, contains=a.contains,
                  min_size=a.min_size, seed=a.seed)
    if not rows:
        print("no recipes matched")
        return 1
    for r in rows:
        show(r)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="im", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="models, datasets and splits").set_defaults(fn=cmd_list)

    g = sub.add_parser("gate", help="Phase 0 negative control")
    g.add_argument("--split", default=DEFAULT_SPLIT)
    g.set_defaults(fn=cmd_gate)

    t = sub.add_parser("train", help="train, score and record one model")
    t.add_argument("model")
    t.add_argument("--split", default=DEFAULT_SPLIT, choices=sorted(SPLITS))
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cpu")
    t.add_argument("--set", action="append", metavar="KEY=VALUE")
    t.add_argument("--run-id")
    t.add_argument("--out", help="override the run directory (Azure ML outputs)")
    t.add_argument("--no-eval", action="store_true")
    t.add_argument("--n-completion", type=int, default=20_000)
    t.add_argument("--allow-leakage", action="store_true",
                   help="proceed despite an unsound model/split combination; "
                        "the resulting M4 is not comparable")
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("eval", help="score an existing run or .npy")
    e.add_argument("target")
    e.add_argument("--split", default=None, choices=sorted(SPLITS))
    e.add_argument("--whiten", action="store_true",
                   help="score after removing the top 3 principal directions")
    e.add_argument("--n-completion", type=int, default=20_000)
    e.set_defaults(fn=cmd_eval)

    r = sub.add_parser("report", help="leaderboard over scored runs")
    r.add_argument("--out")
    r.add_argument("--gate", action="store_true", help="include the control row")
    r.add_argument("--split", default=DEFAULT_SPLIT)
    r.set_defaults(fn=cmd_report)

    sub.add_parser("runs", help="list recorded runs").set_defaults(fn=cmd_runs)

    x = sub.add_parser("explain", help="why two ingredients do or do not pair")
    x.add_argument("a")
    x.add_argument("b")
    x.add_argument("--run")
    x.add_argument("--bridges", type=int, default=5)
    x.set_defaults(fn=cmd_explain)

    n = sub.add_parser("neighbors", help="nearest ingredients in a trained space")
    n.add_argument("run")
    n.add_argument("ingredient")
    n.add_argument("-k", type=int, default=10)
    n.add_argument("--raw", action="store_true", help="skip the near-duplicate filter")
    n.set_defaults(fn=cmd_neighbors)

    s = sub.add_parser("sweep", help="run a declared experiment")
    s.add_argument("spec")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_sweep)

    b = sub.add_parser("recipes", help="browse the corpus in readable form")
    b.add_argument("index", nargs="?", type=int, default=None,
                   help="show one recipe by corpus index")
    b.add_argument("-n", type=int, default=10, help="sample size")
    b.add_argument("--source", help="substring match on source, e.g. xiachufang")
    b.add_argument("--lang", help="exact language code, e.g. zh")
    b.add_argument("--contains", help="only recipes using this ingredient")
    b.add_argument("--min-size", type=int, default=0)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--verify", action="store_true",
                   help="prove every recipe is reachable and decodes")
    b.add_argument("--raw-source", metavar="KEY",
                   help="print raw ingredient lines from the original files")
    b.add_argument("--text", action="store_true",
                   help="show title, url, quantities and steps")
    b.set_defaults(fn=cmd_recipes)
    return ap


def main(argv: list[str] | None = None) -> int:
    from .data.splits import LeakageError
    a = build_parser().parse_args(argv)
    PATHS.ensure()
    try:
        return a.fn(a)
    except LeakageError as e:
        # A stack trace would bury the one sentence that matters.
        print(f"\nrefusing to run — this would produce an uncomparable number:\n"
              f"  {e}\n\n"
              f"  Pass --allow-leakage only if you intend to measure the "
              f"leakage itself.", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"\n{e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
