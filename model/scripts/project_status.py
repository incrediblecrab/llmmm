"""Render current project facts from their exact, versioned source artifacts."""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from ingredient_model.experiments import Trial, _load, expand
from ingredient_model.workspace import load_workspace

ROOT = Path(__file__).resolve().parents[2]
START = "<!-- CURRENT-RESULTS:START -->"
END = "<!-- CURRENT-RESULTS:END -->"


@dataclass(frozen=True)
class Result:
    trial: Trial
    path: Path
    manifest: dict
    metrics: dict

    def score(self, key: str) -> float:
        value = self.metrics[key]
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"{self.path}: invalid {key}: {value!r}")
        return float(value)

    @property
    def native(self) -> float | None:
        key = "M6_native_recall_at_10"
        if key not in self.metrics:
            if self.trial.model in ("ease", "masked-set"):
                raise ValueError(f"{self.path}: missing native predictor score")
            return None
        return self.score(key)

    @property
    def served(self) -> float:
        native = self.native
        return native if native is not None else self.score("M6_recall_at_10")


def read_results(model_root: Path, sweep: str, trials: list[Trial],
                 generation: str, *, allow_pending: bool = False) -> list[Result]:
    records = []
    for trial in trials:
        directory = model_root / "results" / "runs" / sweep / trial.run_id
        metric_path = directory / "metrics.json"
        if allow_pending and not metric_path.exists():
            continue
        manifest = json.loads((directory / "manifest.json").read_text())
        metrics = json.loads(metric_path.read_text())
        if (manifest["model"] != trial.model or manifest["seed"] != trial.seed
                or manifest["params"]["split"] != "recipe-holdout"
                or trial.split != "recipe-holdout"
                or manifest["environment"]["corpus_generation"] != generation):
            raise ValueError(f"{directory}: result identity or evaluation protocol disagrees")
        n = metrics["M6_n"]
        if type(n) is not int or n <= 0:
            raise ValueError(f"{metric_path}: no completed recipe evaluation")
        record = Result(trial, directory, manifest, metrics)
        record.served
        records.append(record)
    return records


def render_status(root: Path) -> str:
    model_root = root / "model"
    config = load_workspace(model_root)
    generation = json.loads((model_root / "data" / "GENERATION.json").read_text())
    corpus = json.loads((model_root / "results" / "corpus_stats.json").read_text())
    for key in ("generation", "recipes", "slots", "vocab", "sha256"):
        if corpus["generation"][key] != generation[key]:
            raise ValueError(f"corpus_stats.json and GENERATION.json disagree on {key}")
    if (sum(row["kept"] for row in corpus["per_source"]) != generation["recipes"]
            or corpus["n_sources"] != len(corpus["per_source"])):
        raise ValueError("corpus source counts disagree with the canonical total")

    benchmark_path = model_root / "experiments" / f"{config.benchmark_sweep}.yaml"
    benchmark_plan = _load(benchmark_path)
    training_plan = _load(config.training_experiment)
    if (benchmark_plan["name"] != config.benchmark_sweep
            or training_plan["name"] != config.training_sweep):
        raise ValueError("workspace.json and experiment names disagree")
    benchmark = read_results(
        model_root, config.benchmark_sweep, expand(benchmark_plan),
        generation["generation"])
    if not benchmark:
        raise ValueError("the declared benchmark contains no results")
    expected_training = expand(training_plan)
    training = read_results(
        model_root, config.training_sweep, expected_training,
        generation["generation"], allow_pending=True)
    population = {
        (row.metrics["M6_n"], row.score("M6_popularity_recall_at_10"))
        for row in benchmark + training
    }
    if len(population) != 1:
        raise ValueError("result files disagree on completion draw size or popularity baseline")
    n, popularity = next(iter(population))
    benchmark = sorted(benchmark, key=lambda row: (-row.served, row.trial.model))

    lines = [
        f"**Corpus {generation['generation']}: {generation['recipes']:,} recipes, "
        f"{generation['slots']:,} ingredient slots, {generation['vocab']:,} "
        f"ingredients, {corpus['n_sources']} source groups.**",
        "",
        f"Published benchmark: `{config.benchmark_sweep}`, "
        f"**{len(benchmark)} recorded runs**, `recipe-holdout`, "
        f"**{n:,} completion instances**. Popularity baseline: **{popularity:.4f}**.",
        "",
        "All score columns are recall@10. Native is the full predictor where "
        "available; a dash means no separate native scorer. Rows retain the "
        "published ordering: native where available, otherwise raw vectors.",
        "",
        "| Model | Raw vectors | Centered vectors | Native predictor |",
        "|---|---:|---:|---:|",
    ]
    for row in benchmark:
        native = "-" if row.native is None else f"{row.native:.4f}"
        lines.append(
            f"| {row.trial.model} | {row.score('M6_recall_at_10'):.4f} | "
            f"{row.score('M6_centred_recall_at_10'):.4f} | {native} |")

    lines += [
        "",
        f"**Current training: `{config.training_sweep}` "
        f"({len(training)}/{len(expected_training)} runs scored).**",
        "",
    ]
    if training:
        lines += [
            "| Model | Seed | Native recall@10 | Lift over popularity | Training time |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in training:
            native = row.native
            if native is None:
                raise ValueError(f"{row.path}: current training requires a native predictor")
            duration = row.manifest["duration_s"]
            if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
                raise ValueError(f"{row.path}: invalid training duration")
            lines.append(
                f"| {row.trial.model} | {row.trial.seed} | {native:.4f} | "
                f"{native - popularity:+.4f} | {duration:.1f}s |")
        for row in training:
            if row.trial.model == "masked-set":
                params = row.manifest["params"]
                lines += [
                    "",
                    f"`masked-set` settings: **{params['epochs']} epochs**, "
                    f"sampling cap **{params['max_recipes']:,} recipes**, "
                    f"maximum training recipe length **{params['max_len']}**.",
                ]
    else:
        lines.append("No scored runs yet for this declared experiment; run `make -C model train`.")
    lines += [
        "",
        "Sources: [corpus marker](model/data/GENERATION.json), "
        "[corpus accounting](model/results/corpus_stats.json), "
        f"[benchmark runs](model/results/runs/{config.benchmark_sweep}/), "
        f"[current experiment](model/{config.training_experiment.relative_to(model_root).as_posix()}).",
    ]
    return "\n".join(lines) + "\n"


def replace_block(text: str, block: str) -> str:
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError("README must contain exactly one current-results marker pair")
    before, tail = text.split(START)
    _, after = tail.split(END)
    return before + START + "\n" + block + END + after


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        block = render_status(args.root)
        if not args.write and not args.check:
            print(block, end="")
            return 0
        path = args.root / "README.md"
        original = path.read_text()
        expected = replace_block(original, block)
        if args.write:
            path.write_text(expected)
            print(f"Updated {path}")
        elif original != expected:
            print("README current results are stale or edited; "
                  "run model/scripts/project_status.py --write", file=sys.stderr)
            return 1
        else:
            print("README current results match their declared source artifacts")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"project status: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
