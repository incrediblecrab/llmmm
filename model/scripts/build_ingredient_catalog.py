"""Build a local ingredient-only browser index from the authorized complete corpus."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from ingredient_model.ingredient_catalog import build_ingredient_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("model/data/recipes/recipe_search.sqlite"))
    parser.add_argument("--corpus", type=Path, default=Path("model/data/recipes/recipe_ids.npz"))
    parser.add_argument("--out", type=Path, required=True, help="new Git-ignored local output directory")
    parser.add_argument("--rows-per-shard", type=int, default=16_384)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.out.resolve()
    ignored = subprocess.run(["git", "check-ignore", "--quiet", "--", str(output / "ingredient-index.json")],
                             cwd=root, check=False)
    if ignored.returncode:
        parser.error("the full catalog must be built in a Git-ignored location; publication is separate")
    result = build_ingredient_catalog(args.catalog.resolve(), args.corpus.resolve(), output,
                                      rows_per_shard=args.rows_per_shard,
                                      progress=lambda event: print(json.dumps(event), flush=True))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
