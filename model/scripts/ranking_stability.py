"""Is the leaderboard a fact about the models, or about seed 0?

Every published result in this repository was trained at seed 0, and the
headline is a statement about an *ordering*: nine of sixteen models score
below a popularity baseline. An ordering derived from a single draw of the
random number generator is not a result, it is an anecdote, and the honest
way to find out which one this is was always to train it again.

`bootstrap_m6.py` answers a different and narrower question. It resamples the
evaluation instances with the models held fixed, so it reports how much of the
headline is an accident of *which recipes were tested*. This script resamples
the training instead. If a model's score moves more between seeds than it does
between bootstrap replicates, then the interval already on the site is the
smaller of the two sources of error and is quietly the wrong one to quote.

Reads `results/runs/seeds-v2/`. Runs while the sweep is still going: a partial
sweep is reported as a partial sweep rather than silently averaged over an
unbalanced set of models, because a mean taken over whichever trials happened
to finish first is a number about scheduling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "results" / "runs" / "seeds-v2"
OUT = ROOT / "results" / "ranking_stability.json"

METRIC = "M6_recall_at_10"
BASELINE = "M6_popularity_recall_at_10"


def load(sweep: Path) -> tuple[dict[str, dict[int, float]], dict[int, float]]:
    """Scores by model and seed, and the popularity baseline by seed.

    Identity comes from each run's manifest rather than from splitting its
    directory name. `chem-svd-recipe-holdout-s0` has no unambiguous parse —
    the model and the split are both hyphenated and there is no delimiter
    between them — and the first version of this script silently read that
    model as `chem-svd-recipe`. The manifest states all three fields, so
    there is nothing to infer.

    The baseline is collected per seed rather than assumed constant. It is the
    line every other claim is measured against, so if it moves, that is worth
    knowing before anything else is concluded.
    """
    scores: dict[str, dict[int, float]] = {}
    base: dict[int, list[float]] = {}
    for p in sorted(sweep.glob("*/metrics.json")):
        man = p.parent / "manifest.json"
        if not man.exists():
            continue
        d = json.loads(p.read_text())
        if METRIC not in d:
            continue
        m = json.loads(man.read_text())
        seed = int(m["seed"])
        scores.setdefault(str(m["model"]), {})[seed] = float(d[METRIC])
        if d.get(BASELINE) is not None:
            base.setdefault(seed, []).append(float(d[BASELINE]))

    # Every model in a seed evaluates against the same baseline, so anything
    # other than agreement means the split moved underneath the sweep.
    baseline: dict[int, float] = {}
    for seed, vals in base.items():
        u = sorted(set(round(v, 6) for v in vals))
        if len(u) > 1:
            raise SystemExit(
                f"seed {seed}: models disagree on the popularity baseline "
                f"({u}) — the evaluation split is not stable")
        baseline[seed] = u[0]
    return scores, baseline


def seed0_scores(runs: Path) -> dict[str, float]:
    """Metric by model for a directory of runs, keyed by manifest identity.

    Same identity rule as `load` — the directory name is not parseable.
    """
    out: dict[str, float] = {}
    for p in sorted(runs.glob("*/metrics.json")):
        man = p.parent / "manifest.json"
        if not man.exists():
            continue
        d = json.loads(p.read_text())
        m = json.loads(man.read_text())
        if METRIC not in d or int(m["seed"]) != 0:
            continue
        out[str(m["model"])] = float(d[METRIC])
    return out


def native_scored(canonical: Path, baseline: float) -> list[dict]:
    """Models whose published score comes from their own scorer, not a vector.

    Two models on this site are published at a native score: `ease` and
    `masked-set` both rank items through their own rule and clear the
    popularity baseline that way, while their exported embeddings fall well
    below it. The headline count of models below the baseline therefore uses
    each model's best available scorer, and is smaller than the count taken
    over embeddings alone.

    The sweep only re-runs the embedding path, so it cannot speak to the
    seed-stability of those two scores. Recording them here keeps that limit
    attached to the artefact rather than to a sentence someone has to
    remember to write — the count on this page and the count on the
    leaderboard differ for a reason, and the reason should be machine-readable.
    """
    out: list[dict] = []
    for p in sorted(canonical.glob("*/metrics.json")):
        man = p.parent / "manifest.json"
        if not man.exists():
            continue
        d = json.loads(p.read_text())
        nat = d.get("M6_native_recall_at_10")
        if nat is None or METRIC not in d:
            continue
        out.append({
            "model": str(json.loads(man.read_text())["model"]),
            "embedding": float(d[METRIC]),
            "native": float(nat),
            "embedding_below_baseline": float(d[METRIC]) < baseline,
            "native_below_baseline": float(nat) < baseline,
        })
    return out


def reproduction(sweep: Path, canonical: Path) -> dict:
    """Does the sweep reproduce the already-published seed-0 numbers?

    The sweep is a second execution of the same models at the same seed, so
    its seed-0 trials should return the canonical values exactly. This matters
    because the whole point of the sweep is to attribute variation to the
    seed: if the pipeline itself had drifted between the published runs and
    this sweep, seed-to-seed differences would be measuring that drift instead
    and there would be no way to tell from the spread alone.

    Reported rather than asserted. A mismatch is a real finding about the
    pipeline, and this script's job is to measure, not to decide what a
    non-zero answer means.
    """
    if not canonical.exists():
        return {"available": False, "reason": f"no {canonical.name} runs"}
    canon, sweep0 = seed0_scores(canonical), seed0_scores(sweep)
    shared = sorted(set(canon) & set(sweep0))
    diffs = {m: abs(canon[m] - sweep0[m]) for m in shared}
    worst = max(diffs.values(), default=0.0)
    return {
        "available": True,
        "canonical_runs": canonical.name,
        "n_compared": len(shared),
        "n_mismatched": sum(1 for v in diffs.values() if v > 1e-12),
        "max_abs_diff": worst,
        "exact": worst == 0.0 and len(shared) > 0,
        "models_compared": shared,
        "not_yet_in_sweep": sorted(set(canon) - set(sweep0)),
    }


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(-a)).astype(float)
    rb = np.argsort(np.argsort(-b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", type=Path, default=SWEEP)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--canonical", type=Path, default=ROOT / "results" / "runs" / "all-v2",
        help="published single-seed runs the sweep's seed 0 should reproduce")
    a = ap.parse_args()

    if not a.sweep.exists():
        raise SystemExit(f"no sweep at {a.sweep}")
    scores, baseline = load(a.sweep)
    if not scores:
        raise SystemExit(f"no completed trials in {a.sweep}")

    seeds = sorted({s for v in scores.values() for s in v})
    complete = sorted(m for m, v in scores.items() if len(v) == len(seeds))
    partial = sorted(set(scores) - set(complete))

    print(f"{a.sweep.relative_to(ROOT)}: {sum(len(v) for v in scores.values())} "
          f"trials, {len(scores)} models, seeds {seeds}")
    if partial:
        print(f"  {len(partial)} model(s) not yet at every seed, held out of "
              f"the ranking: {', '.join(partial)}")
    if len(complete) < 2 or len(seeds) < 2:
        print("\nnot enough complete models or seeds to rank yet — rerun when "
              "the sweep has progressed")
        return 0

    M = np.array([[scores[m][s] for s in seeds] for m in complete])

    # Rank within each seed, best first, over the models that ran everywhere.
    ranks = np.argsort(np.argsort(-M, axis=0), axis=0) + 1

    print(f"\n{len(complete)} models at all {len(seeds)} seeds\n")
    print(f"{'model':<18}{'mean':>9}{'sd':>9}{'range':>9}"
          f"{'rank':>7}{'rank range':>13}")
    rows = []
    for i, m in enumerate(complete):
        v, r = M[i], ranks[i]
        row = {
            "model": m,
            "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)),
            "min": float(v.min()), "max": float(v.max()),
            "rank_best": int(r.min()), "rank_worst": int(r.max()),
            "rank_mean": float(r.mean()),
            "by_seed": {str(s): float(x) for s, x in zip(seeds, v)},
            "below_popularity": [bool(x < baseline[s])
                                 for s, x in zip(seeds, v)],
        }
        rows.append(row)
        rr = (f"{r.min()}" if r.min() == r.max() else f"{r.min()}–{r.max()}")
        print(f"{m[:17]:<18}{v.mean():>9.4f}{v.std(ddof=1):>9.4f}"
              f"{v.max()-v.min():>9.4f}{r.mean():>7.1f}{rr:>13}")

    # The headline, recounted at every seed independently.
    below = {s: int(sum(scores[m][s] < baseline[s] for m in complete))
             for s in seeds}
    moved = sum(1 for r in rows if r["rank_best"] != r["rank_worst"])

    # Worth separating rather than averaging over. Several families here have
    # no stochastic component once the corpus is fixed — a truncated SVD of a
    # PPMI matrix returns the same factorisation every time — so their seed
    # variance is not small, it is absent, and the ranking can only be moved
    # by the ones that actually vary. Reporting a single mean sd would blur
    # the two cases into one uninformative number.
    determ = [r["model"] for r in rows if r["sd"] == 0.0]
    varies = [r for r in rows if r["sd"] > 0.0]

    print(f"\npopularity baseline by seed: "
          + ", ".join(f"s{s} {baseline[s]:.4f}" for s in seeds))
    print(f"below the baseline by seed:  "
          + ", ".join(f"s{s} {below[s]}" for s in seeds))
    print(f"{moved} of {len(complete)} models change rank across seeds")
    print(f"{len(determ)} of {len(complete)} are seed-invariant: "
          f"{', '.join(determ) if determ else 'none'}")

    # A changed rank and a changed conclusion are not the same event. Two
    # models trading places matters only if the swap carries one of them
    # across the popularity line, so that is counted separately: this is the
    # number that decides whether the site's claim survives reseeding.
    crossing = sum(1 for r in rows if r["min"] < baseline[seeds[0]] < r["max"])
    print(f"{crossing} of {len(complete)} cross the popularity baseline at "
          f"any seed")

    # The count above is over embeddings. The published headline is not.
    natives = native_scored(a.canonical, float(np.mean(list(baseline.values()))))
    lifted = [d for d in natives
              if d["embedding_below_baseline"] and not d["native_below_baseline"]]
    if lifted:
        headline = min(below.values()) - len(lifted)
        print(f"\nheadline counts {headline} of {len(complete)} below the "
              f"baseline, not {min(below.values())}, because "
              + ", ".join(f"{d['model']} clears it natively ({d['native']:.4f} "
                          f"vs {d['embedding']:.4f} embedded)" for d in lifted))
        print("those native scorers were not re-run across seeds")
    if varies:
        worst = max(varies, key=lambda r: r["max"] - r["min"])
        gaps = np.diff(np.sort(M.mean(axis=1)))
        gaps = gaps[gaps > 0]
        print(f"largest seed spread {worst['max'] - worst['min']:.4f} "
              f"({worst['model']}); smallest gap between adjacent models "
              f"{gaps.min():.4f}" if len(gaps) else "")

    # Spearman between every pair of seeds, and a bootstrap over models for
    # the mean of those correlations. Quoting +1.000 without an interval was
    # the thing this script exists to stop.
    pairs = [(i, j) for i in range(len(seeds)) for j in range(i + 1, len(seeds))]
    rho = [spearman(M[:, i], M[:, j]) for i, j in pairs]
    rng = np.random.default_rng(a.seed)
    draws = np.empty(a.boot)
    for b in range(a.boot):
        idx = rng.integers(0, len(complete), len(complete))
        draws[b] = np.mean([spearman(M[idx, i], M[idx, j]) for i, j in pairs])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    print(f"\nseed-pair Spearman: mean {np.mean(rho):.4f}, "
          f"min {min(rho):.4f}, 95% CI over models [{lo:.4f}, {hi:.4f}]")

    repro = reproduction(a.sweep, a.canonical)
    if repro["available"]:
        print(f"reproduction of {repro['canonical_runs']} at seed 0: "
              f"{repro['n_compared']} models, {repro['n_mismatched']} "
              f"mismatched, max |diff| {repro['max_abs_diff']:.2e}"
              f"{'  (exact)' if repro['exact'] else ''}")
    else:
        print(f"reproduction check skipped: {repro['reason']}")

    payload = {
        "note": "Each model retrained at several seeds on the same split. "
                "Answers whether the published ordering is a property of the "
                "models or of seed 0. Complementary to m6_intervals.json, "
                "which resamples the evaluation with training held fixed.",
        "metric": METRIC,
        "seeds": seeds,
        "n_models_complete": len(complete),
        "models_partial": partial,
        "baseline_by_seed": {str(s): baseline[s] for s in seeds},
        "n_below_popularity_by_seed": {str(s): below[s] for s in seeds},
        "below_popularity_scope": {
            "basis": METRIC,
            "note": "Counted over exported embeddings. The leaderboard "
                    "headline scores each model at its best available "
                    "scorer, so models with a native rule are counted there "
                    "at that rule and the two totals differ by exactly those "
                    "models. The sweep does not re-run native scorers, so it "
                    "carries no seed evidence about them.",
            "native_not_reseeded": natives,
        },
        "n_models_changing_rank": moved,
        "n_models_crossing_baseline": crossing,
        "seed_invariant": determ,
        "max_seed_spread": (max(r["max"] - r["min"] for r in varies)
                            if varies else 0.0),
        "min_adjacent_gap": float(
            min(g for g in np.diff(np.sort(M.mean(axis=1))) if g > 0))
            if len(complete) > 1 else None,
        "spearman_between_seeds": {
            "pairs": [[seeds[i], seeds[j]] for i, j in pairs],
            "values": rho,
            "mean": float(np.mean(rho)),
            "min": float(min(rho)),
            "ci95": [float(lo), float(hi)],
            "n_boot": a.boot,
        },
        "models": rows,
        "reproduces_published_seed0": repro,
        "complete": not partial and len(seeds) >= 5,
    }
    a.out.write_text(json.dumps(payload, indent=1))
    print(f"\n-> {a.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
