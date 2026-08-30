"""Compare the same models trained on two corpus generations.

The question this answers is narrow: when the corpus got better, did the
leaderboard change its mind about which model is best?

That question is worth separating from "did the numbers go up", because the two
have different evidential value here. The two generations were evaluated on
different held-out sets — v2 has more recipes, so its recipe-holdout split is
not v1's split with extra rows, it is a different draw. An absolute delta
therefore mixes the effect of the corpus with the effect of the test set, and
cannot be attributed to either. The ranking is the more robust reading: each
generation ranks its models against its own test set, and if two independently
drawn evaluations agree on the order, that agreement is about the models.

So the delta column is printed, because hiding a number is worse than
qualifying it, and it is labelled confounded every time it appears.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ingredient_model.config import PATHS

# Ordered by how much they should move a decision, not alphabetically. The
# lift metric leads because it is the only one already net of a baseline: a
# model that beats popularity by nothing has learned nothing worth shipping,
# however good its raw recall looks.
HEADLINE = [
    ("M6_lift_over_popularity", "M6 lift over popularity"),
    ("M6_recall_at_10", "M6 recall@10"),
    ("M6_mrr", "M6 MRR"),
    ("M4_link_auc", "M4 link AUC"),
    ("M2_triplet_accuracy_strict", "M2 triplet (strict)"),
    ("M3_recall_at_10_strict", "M3 recall@10 (strict)"),
]


def load_runs(root: Path) -> dict[str, dict]:
    """Map model name -> metrics for every completed run under root."""
    out: dict[str, dict] = {}
    for path in sorted(root.glob("*/metrics.json")):
        manifest = path.parent / "manifest.json"
        if not manifest.exists():
            continue
        model = json.loads(manifest.read_text()).get("model")
        if model:
            out[model] = json.loads(path.read_text())
    return out


def generation_of(root: Path) -> str:
    """Read the corpus generation stamped into this sweep's manifests.

    Runs made before the marker existed report 'unknown' rather than being
    guessed at. An unstamped run is genuinely unidentified, and labelling it v1
    because it is old would be an inference presented as a fact.
    """
    seen = set()
    for manifest in sorted(root.glob("*/manifest.json")):
        env = json.loads(manifest.read_text()).get("environment", {})
        seen.add(env.get("corpus_generation", "unknown"))
    if not seen:
        return "unknown"
    return "/".join(sorted(seen))


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation, computed directly to avoid a scipy dependency."""

    def rank(xs: list[float]) -> list[float]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        ranks = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            shared = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = shared
            i = j + 1
        return ranks

    ra, rb = rank(a), rank(b)
    n = len(ra)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before", default="baselines")
    ap.add_argument("--after", default="all-v2")
    ap.add_argument("--metric", default="M6_recall_at_10",
                    help="metric used for the ranking comparison")
    args = ap.parse_args()

    root = PATHS.results / "runs"
    before_dir, after_dir = root / args.before, root / args.after
    for d in (before_dir, after_dir):
        if not d.exists():
            raise SystemExit(f"no such sweep: {d}")

    before, after = load_runs(before_dir), load_runs(after_dir)
    shared = sorted(set(before) & set(after))

    print(f"before  {args.before:12s} generation={generation_of(before_dir)}  "
          f"{len(before)} runs")
    print(f"after   {args.after:12s} generation={generation_of(after_dir)}  "
          f"{len(after)} runs")
    print(f"shared  {len(shared)} models: {', '.join(shared) or '(none)'}")
    if only_after := sorted(set(after) - set(before)):
        print(f"new in after (no comparison possible): {', '.join(only_after)}")
    if not shared:
        raise SystemExit("\nnothing to compare.")

    print("\nDeltas are CONFOUNDED: the two generations use different held-out")
    print("sets, so a change mixes corpus effect with test-set effect.\n")

    for key, label in HEADLINE:
        rows = [(m, before[m][key], after[m][key]) for m in shared
                if key in before[m] and key in after[m]]
        if not rows:
            continue
        print(f"{label}")
        print(f"  {'model':<26} {'before':>9} {'after':>9} {'delta':>9}")
        for model, b, a in sorted(rows, key=lambda r: -r[2]):
            print(f"  {model:<26} {b:9.4f} {a:9.4f} {a - b:+9.4f}")
        rho = spearman([r[1] for r in rows], [r[2] for r in rows])
        print(f"  rank correlation (Spearman, n={len(rows)}): {rho:+.3f}\n")

    key = args.metric
    rows = [(m, before[m][key], after[m][key]) for m in shared
            if key in before[m] and key in after[m]]
    if rows:
        b_order = [m for m, _, _ in sorted(rows, key=lambda r: -r[1])]
        a_order = [m for m, _, _ in sorted(rows, key=lambda r: -r[2])]
        print(f"ranking by {key}")
        print(f"  before: {' > '.join(b_order)}")
        print(f"  after:  {' > '.join(a_order)}")
        moved = [m for i, m in enumerate(b_order) if a_order[i] != m]
        print(f"  {'order unchanged' if not moved else 'order changed'}"
              f"{'' if not moved else ': ' + ', '.join(moved) + ' moved'}")


if __name__ == "__main__":
    main()
