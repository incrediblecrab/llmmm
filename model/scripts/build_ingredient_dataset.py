"""Stage a complete ingredient-only Hugging Face dataset locally; never upload."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import pyarrow as pa

from ingredient_model.ingredient_dataset import build_ingredient_dataset

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="immutable ingredient-only index directory")
    parser.add_argument("--out", type=Path, required=True, help="new Git-ignored local output directory")
    parser.add_argument("--source-revision", required=True, help="exact 40-character lowercase git SHA")
    args = parser.parse_args()
    if os.path.lexists(args.out):
        parser.error("ingredient datasets never overwrite existing outputs, including symlinks")
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", str(args.out.resolve()) + "/"],
        cwd=ROOT, capture_output=True, text=True, check=False)
    if ignored.returncode == 1:
        parser.error("the entire output directory must be Git-ignored; publication is separate")
    if ignored.returncode:
        parser.error(f"could not verify Git-ignored output: {ignored.stderr.strip()}")
    pa.set_cpu_count(min(4, pa.cpu_count()))
    pa.set_io_thread_count(min(4, pa.io_thread_count()))
    manifest = build_ingredient_dataset(args.index, args.out, source_revision=args.source_revision)
    print(json.dumps(manifest, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
