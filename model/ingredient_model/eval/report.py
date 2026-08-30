"""Rendering: single-model summaries and cross-run leaderboards."""
from __future__ import annotations

from pathlib import Path

from ..artifacts import Manifest, iter_runs, load_metrics
from ..config import PATHS
from ..data.labels import TIERS

CHANCE = {"M2": 0.50, "M4": 0.50}


def render_one(name: str, r: dict) -> str:
    split = r.get("split")
    header = f"  {name}" + (f"   [split: {split}]" if split else "")
    lines = [header,
             f"    M1 participation ratio   "
             f"{r['M1_participation_ratio']:8.1f}  / {r['d']}"]
    for tier in TIERS:
        lines.append(
            f"    M2 triplet acc [{tier:<6}]  {r[f'M2_triplet_accuracy_{tier}']:8.4f}"
            f"  +/-{r[f'M2_triplet_accuracy_{tier}_ci95']:.4f}  (chance 0.50)")
    for tier in TIERS:
        lines.append(f"    M3 recall@10   [{tier:<6}]  "
                     f"{r[f'M3_recall_at_10_{tier}']:8.4f}")
    lines += [
        f"    M4 held-out link AUC     {r['M4_link_auc']:8.4f}"
        f"  +/-{r['M4_link_auc_ci95']:.4f}  (chance 0.50)",
        f"    M5 max PC-freq corr      {r['M5_max_pc_freq_corr']:8.4f}",
        f"    M5 top20 freq jaccard    {r['M5_top20_freq_jaccard']:8.4f}",
    ]
    if "M6_recall_at_10" in r:
        lines.append(f"    M6 completion recall@10  {r['M6_recall_at_10']:8.4f}"
                     f"   mrr {r['M6_mrr']:.4f}   n={r['M6_n']:,}")
        if "M6_popularity_recall_at_10" in r:
            lines.append(
                f"       vs popularity baseline"
                f"{r['M6_popularity_recall_at_10']:9.4f}"
                f"   lift {r['M6_lift_over_popularity']:+.4f}")
        if "M6_native_recall_at_10" in r:
            lines.append(
                f"       native scorer         "
                f"{r['M6_native_recall_at_10']:9.4f}"
                f"   mrr {r['M6_native_mrr']:.4f}"
                f"   lift {r.get('M6_native_lift_over_popularity', float('nan')):+.4f}")
    return "\n".join(lines)


COLUMNS = [
    ("model", None), ("run", "_label"), ("family", None), ("split", None),
    ("M1", "M1_participation_ratio"),
    ("M2 broad", "M2_triplet_accuracy_broad"),
    ("M2 strict", "M2_triplet_accuracy_strict"),
    ("M3@10", "M3_recall_at_10_broad"),
    ("M4 AUC", "M4_link_auc"),
    ("M5 freq", "M5_top20_freq_jaccard"),
    # M6 last but most important: it is the only column valid for every family
    # at once. `M6 best` takes the native scorer where a model has one, because
    # for item-item models the embedding is a lossy export rather than the model.
    ("M6@10", "M6_recall_at_10"),
    ("M6 best", "_M6_best"),
    ("vs pop", "_M6_best_lift"),
]


def collect(root: Path | None = None) -> list[dict]:
    rows = []
    for d in iter_runs(root):
        metrics = load_metrics(d)
        if metrics is None:
            continue
        man = Manifest.load(d)
        row = {"run_id": man.run_id, "model": man.model,
               "family": man.family, "graph": man.graph.replace(".npz", ""),
               "split": man.params.get("split", "?"),
               "seed": man.seed, "duration_s": man.duration_s, **metrics}
        emb, nat = row.get("M6_recall_at_10"), row.get("M6_native_recall_at_10")
        best = max(x for x in (emb, nat) if x is not None) if (emb or nat) else None
        row["_M6_best"] = best
        pop = row.get("M6_popularity_recall_at_10")
        row["_M6_best_lift"] = None if best is None or pop is None else best - pop
        # A sweep can put ten near-identical rows in the table; without the
        # varied parameter visible they are indistinguishable and the leaderboard
        # reads as if the same model were scored ten times.
        sweep = d.parent.name if d.parent != PATHS.runs else ""
        varied = {k: v for k, v in man.params.items()
                  if k in ("reg", "d_model", "epochs", "walk_len", "window")}
        label = sweep or man.run_id
        if sweep and "reg" in varied:
            label = f"{sweep}/reg={varied['reg']:g}"
        row["_label"] = label
        rows.append(row)
    return rows


def leaderboard(rows: list[dict] | None = None, sort_by: str = "_M6_best",
                root: Path | None = None) -> str:
    """Markdown table over completed runs.

    Sorted by M6 rather than by M4: M6 is the only metric that is leak-free for
    every family, so it is the only one under which the whole table is a
    ranking rather than a list. M1 would be worse still — participation ratio is
    a health check, not a quality score, and ranking on it rewards an isotropy
    no product decision depends on.
    """
    if rows is None:
        rows = collect(root)
    if not rows:
        return "_no scored runs_"
    rows = sorted(rows, key=lambda r: -(r.get(sort_by) or float("-inf")))
    head = "| " + " | ".join(c for c, _ in COLUMNS) + " |"
    rule = "|" + "|".join(["---"] * len(COLUMNS)) + "|"
    out = [head, rule]
    for r in rows:
        cells = []
        for label, key in COLUMNS:
            if key is None:
                cells.append(str(r.get(label, "")))
            else:
                v = r.get(key)
                cells.append("—" if v is None else
                             (str(v) if isinstance(v, str) else
                              f"{v:.1f}" if label == "M1" else
                              f"{v:+.4f}" if label == "vs pop" else f"{v:.4f}"))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def report(rows: list[dict], gate: dict | None = None) -> str:
    parts = ["# Model leaderboard", ""]
    if gate is not None:
        g = gate["results"]["random_gaussian"]
        parts += [
            f"**Negative control:** {'PASS' if gate['passed'] else 'FAIL'} — "
            f"random vectors score M2 {g['M2_triplet_accuracy_broad']:.4f}, "
            f"M4 {g['M4_link_auc']:.4f} against a chance level of 0.50. "
            f"Anything near those numbers is noise.", ""]
    parts += [leaderboard(rows), "",
              f"_{len(rows)} scored runs._"]
    return "\n".join(parts)
