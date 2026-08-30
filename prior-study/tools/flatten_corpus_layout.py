#!/usr/bin/env python3
"""Flatten recipe/expansion/* into recipe/ under the existing NN- naming scheme.

The corpus started as nine numbered sources (01-09) and later grew an
`expansion/` subtree with a different naming convention, so the layout encoded
*when* a source was added rather than what it is. This flattens the two into one
namespace: every source becomes `NN-name` at the root of recipe/.

Sources 01-09 are left alone -- their names are hardcoded in build_recipe_cooc.py,
verify_corpus_table_a1.py, audit_expansion.py and corpus.py, and they are already
in the target format. Numbering continues at 10 in EXPANSION-registry order.

Writes recipe/_layout_migration.json so the move is reversible (this tree is not
under version control).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipe"
EXP = RECIPE / "expansion"
LEDGER = RECIPE / "_layout_migration.json"

# Order mirrors the EXPANSION registry in corpus.py (largest/most load-bearing
# first), so the numbers stay meaningful rather than alphabetical.
ORDER = [
    "foodcom-522k-canonical", "foodcom-raw-231k", "povarenok-detail",
    "turkish-102k", "thefoodprocessor-74k", "allrecipes-33k", "bhuvii-17k",
    "kaggle-food-13k", "hebrew-9.7k", "indian-7k", "persian-6k", "greek-5k",
    "moroccan-4.6k", "japanese-3k", "filipino-2k", "halal-2k", "thai-1k",
    "taiwan-1.8k", "romanian-881", "ingredient-substitutions-74k",
]


def plan() -> list[tuple[Path, Path]]:
    if not EXP.is_dir():
        sys.exit(f"nothing to do: {EXP} does not exist (already flattened?)")

    present = {p.name for p in EXP.iterdir() if p.is_dir() and p.name != "_duplicates"}
    unknown = present - set(ORDER)
    if unknown:
        sys.exit(f"unlisted expansion dirs, refusing to guess numbering: {sorted(unknown)}")

    moves: list[tuple[Path, Path]] = []
    for i, name in enumerate(n for n in ORDER if n in present):
        moves.append((EXP / name, RECIPE / f"{i + 10:02d}-{name}"))

    # loose files at the expansion root (README) and the verified-redundant tree
    for p in EXP.iterdir():
        if p.is_file():
            moves.append((p, RECIPE / f"_expansion_{p.name}"))
        elif p.name == "_duplicates":
            moves.append((p, RECIPE / "_duplicates"))
    return moves


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="perform the moves")
    a = ap.parse_args()

    moves = plan()
    for src, dst in moves:
        print(f"  {src.relative_to(RECIPE)}  ->  {dst.relative_to(RECIPE)}")
    print(f"\n{len(moves)} moves")

    if not a.apply:
        print("dry run; pass --apply to execute")
        return

    for _, dst in moves:
        if dst.exists():
            sys.exit(f"refusing to overwrite existing {dst}")

    done = []
    for src, dst in moves:
        shutil.move(str(src), str(dst))
        done.append({"from": str(src.relative_to(RECIPE)),
                     "to": str(dst.relative_to(RECIPE))})
    LEDGER.write_text(json.dumps(
        {"note": "recipe/expansion flattened into recipe/; reverse to undo",
         "moves": done}, indent=2))

    try:
        EXP.rmdir()
        print(f"removed empty {EXP.relative_to(RECIPE)}/")
    except OSError as e:
        print(f"note: {EXP} not empty ({e})")

    print(f"moved {len(done)}; ledger -> {LEDGER.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
