"""Build a local-only full ingredient search preview; no upload or paid service."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from ingredient_model.ingredient_demo import build_ingredient_demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new Git-ignored preview directory")
    parser.add_argument("--ipv4", action="store_true")
    parser.add_argument("--source-revision")
    parser.add_argument("--dataset-revision")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    ignored = subprocess.run(["git", "check-ignore", "--quiet", "--",
                              str(args.out.resolve() / "catalog.json")], cwd=root, check=False)
    if ignored.returncode:
        parser.error("the full preview must remain in a Git-ignored directory")
    result = build_ingredient_demo(
        root, args.index.resolve(), args.out.resolve(), ipv4=args.ipv4,
        source_revision=args.source_revision, dataset_revision=args.dataset_revision)
    print(json.dumps({"out": str(args.out), "catalog": result["catalog_summary"],
                      "download_bytes": result["index_bytes"]["initial_compressed_download"],
                      "publication_status": result["provenance"]["publication_status"]}, indent=2))


if __name__ == "__main__":
    main()
