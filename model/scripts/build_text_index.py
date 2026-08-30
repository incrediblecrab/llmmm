"""Build the recipe text index.

Replays the original readers, verifies every row against the stored corpus, and
writes `data/recipes/recipe_text.parquet`.

Run `scripts/check_text_alignment.py` first — it catches a drifting metadata
stream in seconds, where this takes tens of minutes to reach the same conclusion.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ingredient_model.data.text import build_index


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N recipes (smoke test)")
    ap.add_argument("--no-steps", action="store_true",
                    help="omit instructions; roughly halves the file")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    t0 = time.time()
    path = build_index(a.out, limit=a.limit, with_steps=not a.no_steps)
    size = path.stat().st_size / 1e9
    print(f"wrote {path} ({size:.2f} GB) in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
