"""Verify completed all-record training, not just data availability or settings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ingredient_model.config import PATHS
from ingredient_model.production import verify_full_training


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=PATHS.runs
                        / "production-v2-all-20260909/llmmm-recipes-full-s42")
    parser.add_argument("--expected-recipes", type=int, default=4_653_430)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        result = verify_full_training(
            args.run, PATHS.data, expected_recipes=args.expected_recipes)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f"Training completion NOT verified: {error}\n")
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
