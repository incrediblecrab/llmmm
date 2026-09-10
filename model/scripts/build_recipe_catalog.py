"""Build the private recipe catalog, metadata prefilter, and a safe report.

Run from the repository root:
    PYTHONPATH=model model/.venv/bin/python model/scripts/build_recipe_catalog.py

For an existing catalog, without modifying SQLite:
    PYTHONPATH=model model/.venv/bin/python model/scripts/build_recipe_catalog.py \
        --metadata-only model/data/recipes/recipe_search.sqlite

All outputs are published without replacing existing files. The JSON report
contains aggregates and provenance only; the SQLite file contains private text
and must stay local. No network services or accelerators are used.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

from ingredient_model.config import PATHS
from ingredient_model.data.recipe_catalog import CATALOG_FILE, build_catalog
from ingredient_model.data.recipe_search_metadata import (
    build_search_metadata,
    default_search_metadata_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="new SQLite path for a full build")
    parser.add_argument("--report", type=Path, help="optional new aggregate report; never overwritten")
    parser.add_argument("--metadata-only", type=Path, metavar="CATALOG",
                        help="build only an adjacent metadata index for this existing SQLite file")
    parser.add_argument("--metadata-out", type=Path, metavar="DIRECTORY",
                        help="explicit metadata-index directory instead of the adjacent default")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--text-sha256", help="optional explicit immutable text-index pin")
    args = parser.parse_args()
    if args.metadata_only and (args.out is not None or args.text_sha256 is not None):
        parser.error("--metadata-only cannot be combined with --out or --text-sha256")
    catalog = args.metadata_only or args.out or PATHS.recipes / CATALOG_FILE
    metadata_out = args.metadata_out or default_search_metadata_path(catalog)
    report_path = args.report
    if report_path is None and args.metadata_only is None:
        report_path = PATHS.results / "recipe_catalog_build.json"
    if os.path.lexists(metadata_out):
        parser.error(f"metadata index already exists and will not be overwritten: {metadata_out}")
    if report_path is not None:
        if os.path.lexists(report_path):
            parser.error(f"report already exists and will not be overwritten: {report_path}")
        if catalog.resolve() == report_path.resolve() or (
                metadata_out.resolve() == report_path.resolve()
                or metadata_out.resolve() in report_path.resolve().parents):
            parser.error("report must be outside the catalog and metadata index")

    def progress(event):
        print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)

    if args.metadata_only is not None:
        report = build_search_metadata(catalog, metadata_out)
    else:
        report = build_catalog(
            catalog, batch_size=args.batch_size, expected_text_sha256=args.text_sha256,
            progress=progress)
        report["search_metadata"] = build_search_metadata(catalog, metadata_out)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        staged = report_path.with_name(f".{report_path.name}.{uuid.uuid4().hex}.building")
        try:
            descriptor = os.open(staged, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.link(staged, report_path)
            directory = os.open(report_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            staged.unlink(missing_ok=True)
    cache_report = report if args.metadata_only is not None else report["search_metadata"]
    print(json.dumps({
        "catalog": str(catalog), "metadata_index": str(metadata_out),
        "report": str(report_path) if report_path is not None else None,
        "rows": cache_report["catalog"]["n_recipes"],
        "slots": cache_report["catalog"]["n_slots"],
        "readable": cache_report["coverage"]["readable"],
        "source_totals": cache_report["coverage"]["source_totals"],
        "metadata_bytes": cache_report["output_bytes"],
        "metadata_elapsed_seconds": cache_report["elapsed_seconds"],
    }), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
