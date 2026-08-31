#!/usr/bin/env python
"""Fail when a number in the documentation matches no artefact.

Roughly two hundred distinct metric-shaped numbers are written into
`README.md`, `model/README.md` and `model/ARCHITECTURE.md`. Every one of them
is a copy of a value that lives in a `metrics.json`, a manifest or a data
file, and nothing has ever checked that the copies still agree with their
originals. They do not: the graph was described as having 203,504 edges long
after it had more, and that string was sitting in the registry and printing
itself in `im list`.

The mechanism here is deliberately narrow. It does not check that a number is
*correctly attributed* — that would require understanding the prose — only
that the value exists somewhere in the artefacts at all. That is a weak
guarantee per number and a strong one in aggregate: a value nothing produces
is either a typo, a stale figure from a superseded generation, or an
invention, and all three are worth failing on.

Numbers legitimately not derived from artefacts are listed in ALLOW with a
reason each. Keeping the exemptions in one visible list is the point; a check
with silent exceptions is not a check.

The current documents also contain older analysis outputs that were printed but
never persisted. They are accepted through `docs_baseline.json`, not because
they are proven right, but because making the new gate immediately noisy would
get it disabled. That file is debt: new unresolved numbers fail, and any entry
that starts resolving is reported so the baseline can shrink.

    python scripts/check_docs.py           # report and exit non-zero on drift
    python scripts/check_docs.py --list    # show what the artefacts provide
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "model"
DOCS = [ROOT / "README.md", MODEL / "README.md", MODEL / "ARCHITECTURE.md"]
BASELINE = Path(__file__).resolve().parent / "docs_baseline.json"

# Metric-shaped: a decimal with three or more places, or an integer written
# with thousands separators. Deliberately not matching bare small integers,
# which are overwhelmingly prose ("three seeds", "16 models") rather than
# measurements, and matching them would bury the signal.
#
# The sign is part of the number. Documents here use the typographic minus
# U+2212 in tables, and dropping it would let a document claim +0.0433 for a
# value the artefacts record as −0.0433. A sign error is exactly the kind of
# drift worth catching.
NUMBER = re.compile(r"[-−+]?\b\d{1,3}(?:,\d{3})+\b|[-−+]?\b\d*\.\d{3,4}\b")

ALLOW: dict[str, str] = {
    "0.001": "learning rate, written in prose, not read from a manifest",
    "0.005": "learning rate",
    "0.010": "significance threshold",
    "0.050": "significance threshold",
}


def _walk(obj, out: set[float]) -> None:
    """Every number anywhere in a JSON document, at any depth."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.add(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _walk(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, out)


def known_values() -> tuple[set[float], dict[str, int]]:
    """Every number any artefact in the repository produces.

    Runs from *all* sweeps, not just the current one, because the docs
    deliberately quote superseded generations when making a before-and-after
    argument. A checker that only knew the latest numbers would flag the
    comparison it is supposed to protect.
    """
    vals: set[float] = set()
    where: dict[str, int] = {}

    def add_from(path: Path, label: str) -> None:
        try:
            before = len(vals)
            _walk(json.loads(path.read_text()), vals)
            where[label] = where.get(label, 0) + len(vals) - before
        except Exception:
            pass

    for p, label in artefact_json_files():
        add_from(p, label)

    # Sizes that exist only as the shape of a binary artefact.
    before = len(vals)
    npz_counts: dict[str, dict[str, int]] = {}
    for p in sorted((MODEL / "data").rglob("*.npz")):
        try:
            with np.load(p, allow_pickle=False) as z:
                counts: dict[str, int] = {}
                for k in z.files:
                    a = z[k]
                    if a.ndim == 0:
                        item = a.item()
                        if isinstance(item, (int, float)):
                            vals.add(float(item))
                    else:
                        n = len(a)
                        vals.add(float(n))
                        counts[k] = n
                npz_counts[str(p.relative_to(MODEL / "data"))] = counts
        except Exception:
            continue
    where["data/*.npz shapes"] = len(vals) - before

    # Derived quantities a reader would reasonably quote.
    extra = set()
    for v in list(vals):
        if v.is_integer() and v > 1000:
            extra.add(v / 1000.0)

    # The v1 graph total is represented as a train and held-out pair because
    # that was the prior study's protocol. The docs sometimes name the total,
    # which is still traceable, but only if the two artefacts are combined.
    for prefix in ("ii_graph", "ii_graph_rh"):
        train = npz_counts.get(f"graphs/{prefix}_train.npz", {}).get("src")
        heldout = npz_counts.get(f"graphs/{prefix}_heldout.npz", {}).get("src")
        if train is not None and heldout is not None:
            total = train + heldout
            extra.add(float(total))
            full = npz_counts.get("graphs/ii_graph.npz", {}).get("src")
            if full is not None:
                extra.add(float(full - total))

    # Generation-over-generation deltas. The corpus rebuild is argued in the
    # documentation as a difference — "recovers 924,315 ingredient
    # occurrences" — and a difference is not stored anywhere, so it would be
    # unresolvable without being computed. Only the current-versus-previous
    # pairs are derived, not all pairwise differences: a checker that accepted
    # any difference of any two known values would accept almost anything.
    gen_path = MODEL / "data" / "GENERATION.json"
    if gen_path.exists():
        try:
            g = json.loads(gen_path.read_text())
            prev = g.get("previous", {})
            for key in ("recipes", "slots", "vocab"):
                if key in g and key in prev:
                    extra.add(float(g[key] - prev[key]))
                    if prev[key]:
                        extra.add(round(100.0 * (g[key] / prev[key] - 1), 2))
        except Exception:
            pass

    vals |= extra
    return vals, where


def artefact_json_files() -> list[tuple[Path, str]]:
    """The JSON artefacts a documented number may legitimately come from.

    Shared by the value collector and by `trace`, so the set of things the
    gate accepts and the set of things it can attribute are the same set. If
    they drifted, the gate could resolve a number it could not explain.
    """
    out: list[tuple[Path, str]] = []
    for p in sorted((MODEL / "results").rglob("metrics.json")):
        out.append((p, "metrics.json"))
    for p in sorted((MODEL / "results").rglob("manifest.json")):
        out.append((p, "manifest.json"))
    for name in ("corpus_stats.json", "m6_intervals.json",
                 "ranking_stability.json"):
        p = MODEL / "results" / name
        if p.exists():
            out.append((p, name))
    for p in sorted(MODEL.rglob("GENERATION.json")):
        out.append((p, "GENERATION.json"))
    return out


def trace(token: str, limit: int = 3) -> list[str]:
    """Where a documented number could have come from, as `file::json.path`.

    Only used to explain a baseline entry that has started resolving. The
    gate accepts a number when *any* artefact anywhere produces it, and the
    pool is now several thousand values, so a three- or four-decimal match
    against a completely unrelated metric is a coincidence waiting to happen.
    Naming the source is what lets a reader tell the two apart before
    dropping the entry.
    """
    text = token.replace(",", "").replace("−", "-").lstrip("+")
    try:
        want = float(text)
    except ValueError:
        return []
    dp = len(text.split(".")[1]) if "." in text else None

    hits: list[str] = []

    def walk(o, path: str, rel: str) -> None:
        if len(hits) >= limit:
            return
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}" if path else str(k), rel)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]", rel)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            ok = round(float(o), dp) == want if dp is not None else float(o) == want
            if ok:
                hits.append(f"{rel}::{path}")

    for p, _ in artefact_json_files():
        if len(hits) >= limit:
            break
        try:
            walk(json.loads(p.read_text()), "", str(p.relative_to(ROOT)))
        except Exception:
            continue
    return hits


def matches(token: str, vals: set[float]) -> bool:
    """Does this written number correspond to some artefact value?

    Compared at the precision it was written to. A doc saying 0.980 should be
    satisfied by a stored 0.97972, because that is what rounding to three
    places means; it should not be satisfied by 0.9749.
    """
    text = token.replace(",", "").replace("−", "-").lstrip("+")
    try:
        want = float(text)
    except ValueError:
        return False
    if "." in text:
        dp = len(text.split(".")[1])
        return any(round(v, dp) == want for v in vals)
    return want in vals


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true",
                    help="report where the known values came from")
    ap.add_argument("--update-baseline", action="store_true",
                    help="record the current unresolved set as accepted")
    ap.add_argument("--strict", action="store_true",
                    help="fail on every unresolved number, baseline included")
    a = ap.parse_args()

    vals, where = known_values()
    if a.list:
        print(f"{len(vals):,} distinct values from:")
        for k, v in sorted(where.items(), key=lambda kv: -kv[1]):
            print(f"  {v:>8,}  {k}")
        return 0

    unresolved: list[tuple[Path, int, str, str]] = []
    checked = 0
    for doc in DOCS:
        if not doc.exists():
            continue
        in_code = False
        for i, line in enumerate(doc.read_text().splitlines(), 1):
            if line.lstrip().startswith("```"):
                in_code = not in_code
                continue
            # Fenced blocks are transcripts of commands and their output.
            # They are evidence, not claims, and rewriting them to match a
            # later run would falsify the record.
            if in_code:
                continue
            for m in NUMBER.finditer(line):
                token = m.group(0)
                checked += 1
                if token in ALLOW or matches(token, vals):
                    continue
                unresolved.append((doc, i, token, line.strip()))

    # Keyed on document and value rather than line number, so that editing
    # prose around a number does not invalidate the record of it.
    keys = {f"{d.relative_to(ROOT)}::{t}" for d, _, t, _ in unresolved}

    if a.update_baseline:
        BASELINE.write_text(json.dumps({
            "note": "Numbers in the documentation that no artefact in this "
                    "repository produces. Almost all are outputs of one-off "
                    "analyses that were printed and never persisted; they are "
                    "not known to be wrong, they are unverifiable, which is a "
                    "different and smaller problem. Recorded here so that new "
                    "drift fails the check while the existing gap stays "
                    "visible and countable. The list should only ever shrink: "
                    "persist an analysis, and its numbers resolve on their "
                    "own.",
            "accepted": sorted(keys),
        }, indent=1))
        print(f"baseline updated: {len(keys)} accepted -> "
              f"{BASELINE.relative_to(ROOT)}")
        return 0

    accepted: set[str] = set()
    if BASELINE.exists() and not a.strict:
        accepted = set(json.loads(BASELINE.read_text())["accepted"])

    new = [u for u in unresolved
           if f"{u[0].relative_to(ROOT)}::{u[2]}" not in accepted]
    fixed = sorted(accepted - keys)

    print(f"checked {checked:,} numbers in {len(DOCS)} documents "
          f"against {len(vals):,} artefact values")
    if accepted:
        print(f"{len(accepted)} known-unverifiable, {len(new)} new")

    if fixed:
        word = "entry" if len(fixed) == 1 else "entries"
        print(f"\n{len(fixed)} baseline {word} now resolve:")
        for k in fixed:
            token = k.split("::", 1)[1]
            src = trace(token)
            print(f"  {k}")
            for s in src:
                print(f"      <- {s}")
            if not src:
                print("      <- (no JSON artefact; resolved by an .npz shape "
                      "or a derived value)")
        print("\n  Check each source is actually the quantity the document "
              "means before\n  dropping it. A match only says some artefact "
              "holds that value, not that\n  it holds it for that reason — at "
              f"{len(vals):,} values, collisions happen.\n"
              "  Then run --update-baseline.")

    if not new:
        if not accepted:
            print("all resolve to an artefact")
        return 0

    print(f"\n{len(new)} value(s) that no artefact produces:\n")
    for doc, i, token, line in new:
        rel = doc.relative_to(ROOT)
        print(f"  {rel}:{i}  {token}")
        print(f"      {line[:110]}")
    print("\nEither the number is stale, or its artefact is missing from this "
          "checkout, or it belongs in ALLOW with a reason. If it is an output "
          "of an analysis that is not persisted, run --update-baseline.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
