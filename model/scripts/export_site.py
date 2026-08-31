"""Export everything the static site needs, and nothing it does not.

The site makes a claim about its own trustworthiness — that any number it
publishes can be traced to the run that produced it. That claim is only worth
something if the site's data is generated from the artefacts rather than
transcribed from them, so this script is the single path from `results/runs/`
to `site/public/data/` and the site has no other source.

Two bundles, separated because they have different sizes and different
lifetimes. The measurement bundle is roughly a hundred kilobytes of JSON
derived from `metrics.json` files, and it changes whenever a model is
retrained. The artefact bundle is tens of megabytes of binary derived from the
embeddings themselves, and it changes only when the corpus does.

On the demo's honesty: it scores with the `recipe-holdout` EASE matrix and
draws its recipes from `held_out_recipes()`, so every suggestion it makes is
about a recipe the model never trained on. Using the full-split model would
give visibly better suggestions and would quietly contradict the argument the
site is built to make.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from ingredient_model.config import PATHS, corpus_generation
from ingredient_model.data import load_recipes
from ingredient_model.data.recipes import RECIPE_IDS
from ingredient_model.data.splits import held_out_recipes

SITE_DATA = Path(__file__).resolve().parents[2] / "site" / "public" / "data"

# Reported on every chart that ranks models. Kept here rather than in the site
# so that adding a metric to the leaderboard is a change to the exporter, which
# is version-controlled beside the runs, and not a change to a template.
HEADLINE = [
    "M6_recall_at_10", "M6_mrr", "M6_lift_over_popularity",
    "M6_native_recall_at_10", "M6_native_lift_over_popularity",
    "M6_centred_recall_at_10", "M6_popularity_recall_at_10",
    "M4_link_auc", "M4_link_auc_ci95",
    "M2_triplet_accuracy_strict", "M2_triplet_accuracy_strict_ci95",
    "M2_triplet_accuracy_broad", "M3_recall_at_10_strict",
    "M1_participation_ratio", "M5_max_pc_freq_corr", "M6_n",
]


def _commit() -> str:
    """The commit the data was exported from, or a marker when unavailable.

    Recorded rather than assumed. A site that claims traceability and cannot
    name its own provenance is making a promise it has not kept.
    """
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parents[2])
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True,
                               cwd=Path(__file__).resolve().parents[2])
        sha = out.stdout.strip() or "unknown"
        return f"{sha}-dirty" if dirty.stdout.strip() else sha
    except Exception:
        return "unknown"


def _runs(sweep: str) -> dict[str, dict]:
    """Load every completed run in a sweep, keyed by model name."""
    root = PATHS.results / "runs" / sweep
    out: dict[str, dict] = {}
    for metrics_path in sorted(root.glob("*/metrics.json")):
        manifest_path = metrics_path.parent / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        model = manifest.get("model")
        if not model:
            continue
        out[model] = {
            "run_id": manifest.get("run_id", metrics_path.parent.name),
            "family": manifest.get("family"),
            "seed": manifest.get("seed"),
            "duration_s": manifest.get("duration_s"),
            "shape": manifest.get("shape"),
            "generation": manifest.get("environment", {})
                                  .get("corpus_generation", "unknown"),
            "dir": str(metrics_path.parent.relative_to(PATHS.results)),
            "metrics": json.loads(metrics_path.read_text()),
        }
    return out


def export_measurements(out_dir: Path) -> None:
    runs = _runs("all-v2")
    if not runs:
        raise SystemExit("no runs under results/runs/all-v2 — train first")

    gen = corpus_generation()
    rows = []
    for model, r in sorted(runs.items()):
        m = r["metrics"]
        row = {k: m[k] for k in HEADLINE if k in m}
        # The scorer a model would actually serve. Two models factorise a
        # richer object than the vector table they export, and the gap between
        # the two columns is the site's second finding, so which number is
        # which has to survive into the data rather than being reconstructed.
        native = "M6_native_recall_at_10" in m
        row.update({
            "model": model,
            "family": r["family"],
            "run_id": r["run_id"],
            "run_dir": r["dir"],
            "seed": r["seed"],
            "generation": r["generation"],
            "duration_s": r["duration_s"],
            "dims": (r["shape"] or [None, None])[1],
            "has_native": native,
            "best_recall_at_10": m["M6_native_recall_at_10"] if native
                                 else m.get("M6_recall_at_10"),
            "best_lift": m["M6_native_lift_over_popularity"] if native
                         else m.get("M6_lift_over_popularity"),
        })
        rows.append(row)

    rows.sort(key=lambda r: -(r["best_lift"] or -9))
    popularity = next((r["metrics"]["M6_popularity_recall_at_10"]
                       for r in runs.values()
                       if "M6_popularity_recall_at_10" in r["metrics"]), None)

    # Do the offline metrics agree with the one that reflects the task? This
    # is computed rather than asserted because it is the site's third finding
    # and because the answer was not the expected one: link AUC, the metric a
    # graph paper would report, carries no rank information about completion
    # at all. Reported with the p-value, since at sixteen models a correlation
    # this size is not distinguishable from none.
    from scipy.stats import spearmanr
    served = [r["best_recall_at_10"] for r in rows]
    embedded = [r.get("M6_recall_at_10") for r in rows]
    agreement = {}
    for name in ("M4_link_auc", "M2_triplet_accuracy_strict",
                 "M1_participation_ratio", "M5_max_pc_freq_corr"):
        xs = [r.get(name) for r in rows]
        if any(x is None for x in xs):
            continue
        for label, ys in (("served", served), ("embedding", embedded)):
            if any(y is None for y in ys):
                continue
            res = spearmanr(xs, ys)
            agreement[f"{name}__vs__M6_{label}"] = {
                "spearman": float(res.statistic), "p": float(res.pvalue),
                "n": len(xs),
            }

    below = sum(1 for r in rows if (r["best_lift"] or 0) < 0)

    # Confidence intervals from scripts/bootstrap_m6.py, attached to the rows
    # they belong to rather than published as a separate table. Optional, so
    # that a fresh checkout can export a leaderboard before the bootstrap has
    # been run; the site renders intervals when they are present and plain
    # numbers when they are not.
    intervals = PATHS.results / "m6_intervals.json"
    ci = None
    if intervals.exists():
        boot = json.loads(intervals.read_text())
        # Confirm this is the artefact bootstrap_m6.py writes, not another
        # file that happens to share the name. A second script once wrote its
        # own schema to this path, and the resulting failure was a bare
        # KeyError three call-frames away from the cause.
        required = {"models", "n_boot", "n_below_popularity",
                    "popularity_recall_at_10"}
        absent = required - set(boot)
        if absent:
            raise SystemExit(
                f"{intervals.name} is missing {sorted(absent)}, so it was not "
                f"written by scripts/bootstrap_m6.py. Re-run that script; if "
                f"another tool wrote this path, point it elsewhere "
                f"(scripts/m6_intervals.py writes m6_replication.json).")
        by_model = {r["model"]: r for r in boot["models"]}
        matched = 0
        for row in rows:
            b = by_model.get(row["model"])
            if not b:
                continue
            # Guard against attaching an interval to a number it was not
            # computed for. A stale bootstrap silently decorating a retrained
            # leaderboard is exactly the failure this file exists to prevent.
            if abs(b["served_recall_at_10"] - (row["best_recall_at_10"] or 0)) > 5e-4:
                raise SystemExit(
                    f"m6_intervals.json is stale for {row['model']}: "
                    f"{b['served_recall_at_10']:.4f} vs leaderboard "
                    f"{row['best_recall_at_10']:.4f}. Re-run "
                    f"scripts/bootstrap_m6.py --refresh.")
            row["ci95"] = b["served_ci95"]
            row["lift_ci95"] = b["served_minus_popularity_ci95"]
            row["p_two_sided"] = b["p_two_sided"]
            row["p_is_bound"] = b["p_is_bound"]
            matched += 1
        ci = {
            "n_boot": boot["n_boot"],
            "n_instances": boot["n_instances"],
            "popularity_ci95": boot["popularity_ci95"],
            "n_below_popularity_ci95": boot["n_below_popularity_ci95"],
            "n_below_popularity_distribution":
                boot["n_below_popularity_distribution"],
            "n_models_with_ci": matched,
        }
        if boot["n_below_popularity"] != below:
            raise SystemExit(
                f"bootstrap counts {boot['n_below_popularity']} below "
                f"popularity, leaderboard counts {below}")

    (out_dir / "leaderboard.json").write_text(json.dumps({
        "generation": gen.get("generation"),
        "split": "recipe-holdout",
        "popularity_recall_at_10": popularity,
        "n_models": len(rows),
        "n_below_popularity": below,
        "bootstrap": ci,
        "agreement": agreement,
        "models": rows,
    }, indent=1))
    lo, hi = (ci or {}).get("n_below_popularity_ci95", (None, None))
    span = f" (95% CI {lo} to {hi})" if lo is not None else " (no intervals)"
    print(f"  leaderboard.json   {len(rows)} models, "
          f"{below} below popularity {popularity:.4f}{span}")

    before, after = _runs("baselines"), runs
    shared = sorted(set(before) & set(after))
    metrics = ["M6_lift_over_popularity", "M6_recall_at_10", "M6_mrr",
               "M4_link_auc", "M2_triplet_accuracy_strict"]
    (out_dir / "comparison.json").write_text(json.dumps({
        # Stated in the data, not only in the prose, because a figure gets
        # screenshotted away from its caption and the confound has to travel
        # with the numbers.
        "caveat": "The two generations are scored against different held-out "
                  "sets, so rankings are comparable but individual deltas "
                  "confound corpus effect with test-set effect.",
        "before": {"sweep": "baselines", "generation": "unstamped"},
        "after": {"sweep": "all-v2", "generation": gen.get("generation")},
        "metrics": metrics,
        "models": [{
            "model": m,
            **{k: {"before": before[m]["metrics"].get(k),
                   "after": after[m]["metrics"].get(k)} for k in metrics},
        } for m in shared],
    }, indent=1))
    print(f"  comparison.json    {len(shared)} shared models")

    corpus = load_recipes(RECIPE_IDS)
    counts = np.bincount(corpus.flat, minlength=corpus.n_vocab)
    order = np.argsort(-counts)
    (out_dir / "vocab.json").write_text(json.dumps({
        "n": int(corpus.n_vocab),
        "total_slots": int(counts.sum()),
        # Ordered by frequency so the popularity baseline is the identity
        # ranking. The demo's "what a frequency table would say" is then the
        # first few entries of this array and needs no separate model.
        "by_frequency": [{"id": int(i), "name": str(corpus.itos[i]),
                          "count": int(counts[i])} for i in order],
    }, indent=1))
    print(f"  vocab.json         {corpus.n_vocab} ingredients")

    stats_path = PATHS.results / "corpus_stats.json"
    payload = {"generation": gen}
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        # The waterfall is only worth publishing if it lands on the corpus it
        # claims to describe. A scan run against a different generation would
        # produce a plausible figure that quietly contradicts every other
        # number on the site, so it fails here instead.
        w = stats["waterfall"]
        if w["kept"] != corpus.n_recipes:
            raise SystemExit(
                f"corpus_stats.json kept {w['kept']:,} but the corpus holds "
                f"{corpus.n_recipes:,} — the scan is from another generation")
        per = stats["per_source"]
        src_rows = per if isinstance(per, list) else [
            {"source": k, **v} for k, v in per.items()]
        for field in ("scanned", "kept"):
            total = sum(r.get(field, 0) for r in src_rows)
            if total != w[field]:
                raise SystemExit(
                    f"corpus_stats.json per-source {field} sums to {total:,}, "
                    f"waterfall says {w[field]:,}")
        payload["stats"] = stats
        print(f"  corpus.json        {w['scanned']:,} scanned -> "
              f"{w['kept']:,} kept across {len(src_rows)} sources")
    else:
        print("  (results/corpus_stats.json absent — corpus.json is partial)")
    (out_dir / "corpus.json").write_text(json.dumps(payload, indent=1))

    # The leakage demonstration, measured rather than remembered. The trust
    # page's whole argument is that an optimistic protocol is the default, and
    # it was making that argument with a number typed into the template that
    # no run in this repository produced — the one place on the site where
    # that was least affordable. Both sides are now runs.
    leaky = _runs("leakage-demo").get("ease")
    honest = runs.get("ease")
    if leaky and honest:
        a = leaky["metrics"]["M4_link_auc"]
        b = honest["metrics"]["M4_link_auc"]
        ci95 = honest["metrics"]["M4_link_auc_ci95"]
        (out_dir / "splits.json").write_text(json.dumps({
            "note": "The same model and the same metric under two split "
                    "protocols. Hiding a tenth of the graph's edges hides "
                    "nothing from a model that reads the recipes those edges "
                    "came from.",
            "metric": "M4_link_auc",
            "model": "ease",
            "leaky": {"split": "edge-holdout", "value": a,
                      "ci95": leaky["metrics"].get("M4_link_auc_ci95"),
                      "run_dir": leaky["dir"]},
            "honest": {"split": "recipe-holdout", "value": b, "ci95": ci95,
                       "run_dir": honest["dir"]},
            "gap": a - b,
            "gap_in_ci_units": (a - b) / ci95 if ci95 else None,
        }, indent=1))
        print(f"  splits.json        leaky {a:.4f} vs honest {b:.4f}, "
              f"{(a - b) / ci95:.1f}x the CI")
    else:
        print("  (no leakage-demo run — splits.json not written)")

    # Multi-seed stability, published only once every model has run at every
    # seed. A partial sweep produces a real number over whichever trials
    # finished first, which is a fact about the scheduler rather than about
    # the models, and it would be indistinguishable on the page from the
    # finished result.
    stab_path = PATHS.results / "ranking_stability.json"
    if stab_path.exists():
        stab = json.loads(stab_path.read_text())
        if stab.get("complete"):
            names = {r["model"] for r in stab["models"]}
            missing = {r["model"] for r in rows} - names
            if missing:
                raise SystemExit(
                    "ranking_stability.json is marked complete but omits "
                    f"{sorted(missing)} — re-run scripts/ranking_stability.py")
            (out_dir / "stability.json").write_text(json.dumps(stab, indent=1))
            print(f"  stability.json     {stab['n_models_complete']} models x "
                  f"{len(stab['seeds'])} seeds, "
                  f"{stab['n_models_changing_rank']} rank changes")
        else:
            n_done = stab["n_models_complete"]
            print(f"  (seed sweep still running — {n_done} of "
                  f"{len(rows)} models at all seeds, stability.json withheld)")
    else:
        print("  (no ranking_stability.json — stability.json not written)")

    (out_dir / "provenance.json").write_text(json.dumps({
        "commit": _commit(),
        "generation": gen,
        "note": "Regenerate with scripts/export_site.py. The site has no "
                "other data source, so a number that is not here is a number "
                "the site cannot show.",
    }, indent=1))
    print(f"  provenance.json    commit {_commit()}")


def export_artifacts(out_dir: Path, k: int = 50) -> None:
    runs = _runs("all-v2")
    corpus = load_recipes(RECIPE_IDS)
    n = corpus.n_vocab

    # float16 throughout. The values are cosine similarities and model scores
    # in a narrow range, three decimals is more than the site displays, and
    # halving the transfer matters more than precision nobody can see.
    index = {}
    ids_all, scores_all = [], []
    for model, r in sorted(runs.items()):
        emb_path = PATHS.results / r["dir"] / "embedding.npy"
        if not emb_path.exists():
            continue
        W = np.load(emb_path).astype(np.float32)
        U = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)
        S = U @ U.T
        np.fill_diagonal(S, -np.inf)
        top = np.argpartition(-S, k, axis=1)[:, :k]
        row = np.take_along_axis(S, top, 1)
        srt = np.argsort(-row, axis=1)
        index[model] = len(ids_all)
        ids_all.append(np.take_along_axis(top, srt, 1).astype(np.uint16))
        scores_all.append(np.take_along_axis(row, srt, 1).astype(np.float16))

    np.stack(ids_all).tofile(out_dir / "neighbors_ids.bin")
    np.stack(scores_all).tofile(out_dir / "neighbors_scores.bin")
    (out_dir / "neighbors.json").write_text(json.dumps({
        "models": list(index), "k": k, "n": n,
        "ids_dtype": "uint16", "scores_dtype": "float16",
        "layout": "models x n x k, model order as listed",
    }, indent=1))
    mb = (np.stack(ids_all).nbytes + np.stack(scores_all).nbytes) / 1048576
    print(f"  neighbors          {len(index)} models x {n} x {k}, {mb:.1f} MB")

    test = held_out_recipes("recipe-holdout", limit=None)
    lens = test.sizes
    lo, hi = 4, 9
    pick = np.flatnonzero((lens >= lo) & (lens <= hi))
    rng = np.random.default_rng(0)
    pick = rng.choice(pick, min(400, len(pick)), replace=False)
    recipes = []
    for r in pick:
        a, b = test.offsets[r], test.offsets[r + 1]
        recipes.append({"ids": [int(x) for x in test.flat[a:b]],
                        "source": str(test.source[r])})

    # Only the rows the demo can reach.
    #
    # The browser scores a bowl by summing the item-item rows of the visible
    # ingredients, so the only rows it ever indexes are those of ingredients
    # that appear in the exported recipes. Shipping the full square sends five
    # megabytes that nothing can read, and on a slow connection that was the
    # difference between a two-minute wait and a twenty-second one.
    #
    # This is lossless rather than an approximation: every score the demo
    # computes is over all candidates and is bit-identical to the score from
    # the full matrix. What changes is which rows are present, not what any
    # row contains. The row map ships alongside so the client resolves an
    # ingredient id to a row rather than assuming the identity mapping, and
    # can fail loudly if it is ever asked for a row that was not sent.
    ease = PATHS.results / runs["ease"]["dir"] / "item_scores.npy"
    B = np.load(ease).astype(np.float16)
    reachable = sorted({i for r in recipes for i in r["ids"]})
    rows = np.array(reachable, dtype=np.int32)
    B[rows].tofile(out_dir / "ease_scores.bin")
    full_mb = B.nbytes / 1048576
    sent_mb = B[rows].nbytes / 1048576
    print(f"  ease_scores.bin    {len(rows)} of {B.shape[0]} rows x "
          f"{B.shape[1]} float16, {sent_mb:.1f} MB "
          f"({full_mb / sent_mb:.1f}x smaller than the full matrix)")

    # What the demo should score, on the population the demo actually draws
    # from. The harness numbers describe recipes with three or more
    # ingredients; the demo shows four to nine, because a two-ingredient bowl
    # is not a game and a forty-ingredient one is not readable. Those are
    # different populations, so comparing the browser against the harness
    # conflates a real difference with drift. Measuring the subpopulation
    # directly removes the confound and lets the smoke test be tight.
    B = np.load(PATHS.results / runs["ease"]["dir"] / "item_scores.npy")
    sub = test.select(np.sort(pick))
    # The same frequency table the harness ranks with, and the same one
    # vocab.json ships to the browser — counted over the whole corpus, not
    # over the held-out split. Using the split's own counts would give a
    # popularity baseline that is nearly but not exactly the published one,
    # which is the kind of near-miss that is worse than an obvious error.
    unigram = np.bincount(load_recipes(RECIPE_IDS).flat,
                          minlength=test.n_vocab).astype(float)
    from ingredient_model.eval.completion import recipe_completion
    exp = recipe_completion(np.load(PATHS.results / runs["ease"]["dir"]
                                    / "embedding.npy"),
                            sub, n_test=len(pick), unigram=unigram,
                            scorer=lambda c: B[c].sum(1))

    (out_dir / "demo_recipes.json").write_text(json.dumps({
        "note": "Drawn from held_out_recipes('recipe-holdout'), so no model "
                "in the demo trained on any of these.",
        "min_ingredients": lo, "max_ingredients": hi,
        # Ingredient id -> row index in ease_scores.bin. Ordered, so the client
        # can binary-search or build a lookup; either way it must not assume
        # row i belongs to ingredient i.
        "score_rows": [int(i) for i in rows],
        "expected": {
            "ease": exp["M6_native_recall_at_10"],
            "popularity": exp["M6_popularity_recall_at_10"],
            "n": exp["M6_n"],
        },
        "n": len(recipes), "recipes": recipes,
    }, indent=1))
    print(f"  demo_recipes.json  {len(recipes)} held-out recipes, "
          f"expected ease {exp['M6_native_recall_at_10']:.4f} / "
          f"popularity {exp['M6_popularity_recall_at_10']:.4f}")


def verify(out_dir: Path) -> None:
    """Re-score EASE from the exported fp16 bytes and compare to the run.

    The site shows a live model, so the honest question is whether the thing
    in the browser is the thing that was measured. Halving the precision to
    halve the download is only defensible if the cost is known, so it is
    measured here rather than asserted. Measured cost: recall@10 0.5872 at
    float32 against 0.5871 at float16, one instance in twenty thousand.

    Slow — a few minutes — because it runs the real evaluation rather than a
    proxy for it. That is the point, so it is a flag and not part of `make
    check`.
    """
    from ingredient_model.eval.completion import recipe_completion

    runs = _runs("all-v2")
    run_dir = PATHS.results / runs["ease"]["dir"]
    stored = json.loads((run_dir / "metrics.json").read_text())
    W = np.load(run_dir / "embedding.npy")
    B32 = np.load(run_dir / "item_scores.npy").astype(np.float32)
    B16 = np.fromfile(out_dir / "ease_scores.bin",
                      np.float16).reshape(B32.shape).astype(np.float32)

    corpus = held_out_recipes("recipe-holdout")
    for label, B in (("float32", B32), ("float16 (exported)", B16)):
        r = recipe_completion(W, corpus,
                              scorer=lambda c, B=B: B[c].sum(1))
        print(f"  {label:20s} native recall@10 "
              f"{r['M6_native_recall_at_10']:.4f}")
    print(f"  {'run metrics.json':20s} native recall@10 "
          f"{stored['M6_native_recall_at_10']:.4f}")
    print(f"  max quantisation error {np.abs(B32 - B16).max():.2e} "
          f"over range [{B32.min():.3f}, {B32.max():.3f}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--what", choices=["measurements", "artifacts", "all"],
                    default="all")
    ap.add_argument("--out", type=Path, default=SITE_DATA)
    ap.add_argument("--verify", action="store_true",
                    help="re-score EASE from the exported bytes (slow)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"-> {args.out}")
    if args.what in ("measurements", "all"):
        export_measurements(args.out)
    if args.what in ("artifacts", "all"):
        export_artifacts(args.out)
    if args.verify:
        verify(args.out)


if __name__ == "__main__":
    main()
