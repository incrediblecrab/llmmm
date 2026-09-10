"""Build and numerically verify a browser-only demo using only public inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ingredient_model.recipe_demo import build_demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new static Space build directory")
    parser.add_argument("--source-revision", help="optional exact committed source; required for publication")
    parser.add_argument("--dataset-revision", help="optional exact published dataset commit")
    parser.add_argument("--ipv4", action="store_true", help="use IPv4 for Hub downloads on affected networks")
    args = parser.parse_args()
    result = build_demo(Path(__file__).resolve().parents[2], args.out.resolve(),
                        source_revision=args.source_revision, dataset_revision=args.dataset_revision,
                        ipv4=args.ipv4)
    print(json.dumps({"out": str(args.out), "catalog": result["catalog_summary"],
                      "verification": result["verification"]}, indent=2))


if __name__ == "__main__":
    main()
