#!/usr/bin/env python3
"""Measure what the canonical corpus kept, and what it left behind.

The promoted v2 corpus records the recipes that survived normalisation, but not the two ways a raw recipe can disappear before it reaches the artefact: it can be an exact duplicate by raw ingredient text, or it can match no known ingredient after normalisation. That split matters because the two failures mean different things. Duplicates describe the source collection; empty recipes describe the normaliser's coverage.

This script replays the original readers in read-only mode, using the normaliser named in ``data/GENERATION.json``, and compares that raw denominator with the promoted ``recipe_ids.npz``. It is intentionally an audit pass rather than a rebuild: the corpus files are evidence here, not outputs to be replaced.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ingredient_model.config import PATHS, corpus_generation  # noqa: E402
from ingredient_model.data.normalizer import get_normalizer  # noqa: E402

EXPECTED_SCANNED = 5_325_676
EXPECTED_KEPT = 4_653_430
SEP = "\u241f"


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def load_normalizer():
    """Instantiate the normaliser that built the current corpus."""
    marker = corpus_generation()
    which = marker.get("normalizer", "base")
    if which == "corrected":
        return which, get_normalizer(fix_zh_qty=True, extra=True)

    if str(PATHS.prior_tools) not in sys.path:
        sys.path.insert(0, str(PATHS.prior_tools))
    import normalize as nm  # type: ignore

    return which, nm.Normalizer()


def load_kept_counts(path: Path) -> tuple[Counter[str], list[dict[str, Any]], int]:
    """Read the promoted corpus once for source and ingredient summaries."""
    with np.load(path, allow_pickle=False) as z:
        sources = z["source"]
        flat = z["flat"]
        itos = z["itos"]

        kept_by_source = Counter(
            {str(k): int(v) for k, v in zip(*np.unique(sources,
                                                       return_counts=True))}
        )
        counts = np.bincount(flat, minlength=len(itos))
        order = np.argsort(counts)[::-1][:20]
        total_slots = int(counts.sum())
        top = [
            {
                "rank": i + 1,
                "ingredient": str(itos[idx]),
                "count": int(counts[idx]),
                "slot_share_pct": pct(int(counts[idx]), total_slots),
            }
            for i, idx in enumerate(order)
        ]
    return kept_by_source, top, total_slots


def scan_raw(limit: int | None = None) -> tuple[dict[str, Counter[str]], dict[str, int]]:
    """Replay the builder's drop rules without writing derived artefacts."""
    if str(PATHS.prior_tools) not in sys.path:
        sys.path.insert(0, str(PATHS.prior_tools))
    import corpus  # type: ignore

    which, nz = load_normalizer()
    print(f"normaliser: {which}")

    seen_hashes: set[int] = set()
    per_source: dict[str, Counter[str]] = {}
    totals = {"scanned": 0, "dup": 0, "empty": 0, "kept": 0}
    t0 = time.time()

    for key, lang, items in corpus.iter_all(limit):
        source = str(key)
        totals["scanned"] += 1
        row = per_source.setdefault(source, Counter())
        row["scanned"] += 1

        h = hash(SEP.join(sorted(str(i).strip().lower() for i in items)))
        if h in seen_hashes:
            totals["dup"] += 1
            row["dup"] += 1
            continue
        seen_hashes.add(h)

        ids = nz.normalize(lang, items)
        if not ids:
            totals["empty"] += 1
            row["empty"] += 1
            continue

        totals["kept"] += 1
        row["kept"] += 1

        if totals["scanned"] % 500_000 == 0:
            print(
                f"  {totals['scanned']:,} scanned, "
                f"{totals['kept']:,} kept, {totals['dup']:,} dup, "
                f"{totals['empty']:,} empty",
                flush=True,
            )

    elapsed = (time.time() - t0) / 60
    print(f"scan complete in {elapsed:.1f} min")
    return per_source, totals


def build_payload(
    per_source_raw: dict[str, Counter[str]],
    totals: dict[str, int],
    kept_by_source: Counter[str],
    top_ingredients: list[dict[str, Any]],
    total_slots: int,
) -> dict[str, Any]:
    kept_total = sum(kept_by_source.values())
    sources = []
    for source in sorted(set(per_source_raw) | set(kept_by_source)):
        raw = per_source_raw.get(source, Counter())
        scanned = int(raw.get("scanned", 0))
        kept = int(kept_by_source.get(source, 0))
        sources.append({
            "source": source,
            "scanned": scanned,
            "dup": int(raw.get("dup", 0)),
            "empty": int(raw.get("empty", 0)),
            "kept": kept,
            "survival_rate_pct": pct(kept, scanned),
            "final_corpus_share_pct": pct(kept, kept_total),
        })

    return {
        "note": (
            "Read-only replay of the original corpus readers. Duplicate and "
            "empty counts use the same rules as scripts/rebuild_corpus.py; "
            "kept counts by source and ingredient counts come from the "
            "promoted recipe_ids.npz artefact."
        ),
        "generation": corpus_generation(),
        "waterfall": {
            "scanned": totals["scanned"],
            "scanned_pct": 100.0,
            "dropped_duplicate": totals["dup"],
            "dropped_duplicate_pct": pct(totals["dup"], totals["scanned"]),
            "dropped_matched_nothing": totals["empty"],
            "dropped_matched_nothing_pct": pct(totals["empty"],
                                               totals["scanned"]),
            "kept": totals["kept"],
            "kept_pct": pct(totals["kept"], totals["scanned"]),
            "dropped_total": totals["dup"] + totals["empty"],
            "dropped_total_pct": pct(totals["dup"] + totals["empty"],
                                     totals["scanned"]),
        },
        "per_source": sorted(sources, key=lambda r: (-r["kept"],
                                                    r["source"])),
        "n_sources": len(sources),
        "ingredients": {
            "slots": total_slots,
            "top_20": top_ingredients,
        },
    }


def validate(payload: dict[str, Any]) -> None:
    wf = payload["waterfall"]
    if wf["scanned"] != EXPECTED_SCANNED:
        raise SystemExit(
            f"scanned {wf['scanned']:,} != expected {EXPECTED_SCANNED:,}"
        )
    if wf["kept"] != EXPECTED_KEPT:
        raise SystemExit(f"kept {wf['kept']:,} != expected {EXPECTED_KEPT:,}")
    if (wf["dropped_duplicate"] + wf["dropped_matched_nothing"] +
            wf["kept"] != wf["scanned"]):
        raise SystemExit("waterfall does not reconcile")

    kept_by_source = sum(int(r["kept"]) for r in payload["per_source"])
    if kept_by_source != EXPECTED_KEPT:
        raise SystemExit(
            f"per-source kept {kept_by_source:,} != expected "
            f"{EXPECTED_KEPT:,}"
        )


def print_summary(payload: dict[str, Any]) -> None:
    wf = payload["waterfall"]
    print("\nCorpus waterfall")
    print(f"  scanned              {wf['scanned']:>12,}  {wf['scanned_pct']:6.2f}%")
    print(
        f"  dropped duplicate    {wf['dropped_duplicate']:>12,}  "
        f"{wf['dropped_duplicate_pct']:6.2f}%"
    )
    print(
        f"  matched nothing      {wf['dropped_matched_nothing']:>12,}  "
        f"{wf['dropped_matched_nothing_pct']:6.2f}%"
    )
    print(f"  kept                 {wf['kept']:>12,}  {wf['kept_pct']:6.2f}%")

    print("\nPer-source survival")
    print(
        f"  {'source':<20} {'scanned':>10} {'kept':>10} "
        f"{'survival':>9} {'share':>8}"
    )
    for row in sorted(payload["per_source"], key=lambda r: r["source"]):
        print(
            f"  {row['source']:<20} {row['scanned']:>10,} "
            f"{row['kept']:>10,} {row['survival_rate_pct']:>8.2f}% "
            f"{row['final_corpus_share_pct']:>7.2f}%"
        )

    print("\nTop ingredients by slot count")
    print(f"  {'ingredient':<24} {'count':>10} {'share':>8}")
    for row in payload["ingredients"]["top_20"]:
        print(
            f"  {row['ingredient']:<24} {row['count']:>10,} "
            f"{row['slot_share_pct']:>7.2f}%"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=PATHS.results / "corpus_stats_audit.json",
                    type=Path,
                    help="deliberately NOT results/corpus_stats.json. That "
                         "file is the canonical waterfall, written by the "
                         "corpus-building pass and reconciled against the "
                         "graph unigram; this script is an independent audit "
                         "of the same quantities and must not overwrite the "
                         "artefact it exists to check.")
    ap.add_argument("--limit", default=None, type=int,
                    help="debug only; validation expects the full corpus")
    args = ap.parse_args()

    kept_by_source, top_ingredients, total_slots = load_kept_counts(
        PATHS.recipes / "recipe_ids.npz"
    )
    per_source_raw, totals = scan_raw(args.limit)
    payload = build_payload(per_source_raw, totals, kept_by_source,
                            top_ingredients, total_slots)
    validate(payload)

    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print_summary(payload)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
