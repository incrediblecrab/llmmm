#!/usr/bin/env python3
"""Strip stale row-count suffixes from corpus directory names.

Sources 01-09 are named for what they are (`03-povarenok`); the later ones
carried a row count (`12-povarenok-detail`, `18-hebrew-9.7k`). The counts go
stale the moment a source is re-pulled and they duplicate information that
belongs in a manifest, so this reduces every directory to `NN-name` and writes
the counts to recipe/MANIFEST.md instead.

Qualifiers that distinguish two cuts of the same source are kept:
`foodcom-522k-canonical` -> `foodcom-canonical` (vs `foodcom-raw`), and
`povarenok-detail` stays `povarenok-detail` (vs the `03-povarenok` baseline).

Updates the EXPANSION registry paths in tools/corpus.py and extends
recipe/_layout_migration.json so both steps remain reversible.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipe"
LEDGER = RECIPE / "_layout_migration.json"
MANIFEST = RECIPE / "MANIFEST.md"

# trailing row counts: -522k, -9.7k, -881, -1.8k
COUNT = re.compile(r"-\d+(?:\.\d+)?k$|-\d{3,}$")


def clean(name: str) -> str:
    num, rest = name.split("-", 1)
    prev = None
    while prev != rest:                     # foodcom-522k-canonical -> foodcom-canonical
        prev = rest
        rest = COUNT.sub("", rest)
        parts = rest.split("-")
        for i, p in enumerate(parts):
            if COUNT.fullmatch("-" + p):
                rest = "-".join(parts[:i] + parts[i + 1:])
                break
    return f"{num}-{rest}"


def plan() -> list[tuple[Path, Path]]:
    moves = []
    for p in sorted(RECIPE.iterdir()):
        if not p.is_dir() or not p.name[:2].isdigit():
            continue
        new = clean(p.name)
        if new != p.name:
            moves.append((p, RECIPE / new))
    return moves


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    moves = plan()
    for src, dst in moves:
        print(f"  {src.name:<32} ->  {dst.name}")
    print(f"\n{len(moves)} renames")
    if not a.apply:
        print("dry run; pass --apply to execute")
        return

    names = [d.name for _, d in moves]
    if len(names) != len(set(names)):
        sys.exit("rename would collide")
    for _, dst in moves:
        if dst.exists():
            sys.exit(f"refusing to overwrite {dst}")

    # capture the counts we are about to erase from the names
    rows = []
    for src, dst in moves:
        m = re.search(r"-(\d+(?:\.\d+)?k|\d{3,})", src.name)
        rows.append((dst.name, m.group(1) if m else "", src.name))

    for src, dst in moves:
        shutil.move(str(src), str(dst))

    led = json.loads(LEDGER.read_text())
    led.setdefault("renames", []).extend(
        {"from": s.name, "to": d.name} for s, d in moves)
    LEDGER.write_text(json.dumps(led, indent=2))

    # corpus.py registry paths
    cp = ROOT / "tools" / "corpus.py"
    s = cp.read_text()
    n = 0
    for src, dst in sorted(moves, key=lambda kv: -len(kv[0].name)):
        if f'"{src.name}/' in s:
            s = s.replace(f'"{src.name}/', f'"{dst.name}/'); n += 1
    cp.write_text(s)

    all_dirs = sorted(p.name for p in RECIPE.iterdir()
                      if p.is_dir() and p.name[:2].isdigit())
    cnt = {d: c for d, c, _ in rows}
    lines = ["# Corpus sources", "",
             "One directory per source, `NN-name`. Row counts live here rather than",
             "in directory names so they can be corrected without a rename.", "",
             "| # | source | approx rows | former name |", "|---|---|---|---|"]
    former = {d: f for d, _, f in rows}
    for d in all_dirs:
        lines.append(f"| {d[:2]} | {d[3:]} | {cnt.get(d, '—')} | "
                     f"{former.get(d, '—')} |")
    MANIFEST.write_text("\n".join(lines) + "\n")

    print(f"renamed {len(moves)}; corpus.py paths updated: {n}")
    print(f"manifest -> {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
