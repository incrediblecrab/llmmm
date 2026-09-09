"""Measure generation-text coverage without treating nonempty fields as quality."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ingredient_model.config import PATHS
from ingredient_model.data.text import TEXT_FILE
from ingredient_model.workspace import verify_training_corpus

FIELDS = ("title", "raw_ingredients", "steps")
COUNT_FIELDS = (*FIELDS, "ingredient_letters", "complete_text",
                "complete_with_ingredient_letters", "unresolved_quantity_units",
                "quantity_alignment_errors", "unparsed_step_fields")


def summarize_batch(batch: pa.RecordBatch) -> dict[str, dict[str, int]]:
    text = {
        field: pc.utf8_trim_whitespace(pc.fill_null(batch.column(field), ""))
        for field in FIELDS
    }
    present = {field: pc.greater(pc.utf8_length(value), 0)
               for field, value in text.items()}
    raw = pc.replace_substring_regex(
        text["raw_ingredients"], r"(?s)^c\s*\((.*)\)$", r"\1")
    raw = pc.replace_substring_regex(
        raw, r"(?i)\b(?:none|nan|null|na)\b", "")
    present["ingredient_letters"] = pc.match_substring_regex(raw, r"\p{L}")
    present["complete_text"] = pc.and_(
        pc.and_(present["title"], present["raw_ingredients"]), present["steps"])
    present["complete_with_ingredient_letters"] = pc.and_(
        present["complete_text"], present["ingredient_letters"])
    if "quantity_status" in batch.schema.names:
        status = pc.fill_null(batch.column("quantity_status"), "")
        allowed = pa.array(["", "values_without_units", "not_supplied", "count_mismatch"])
        if pc.any(pc.invert(pc.is_in(status, value_set=allowed))).as_py():
            raise ValueError("unknown quantity-status metadata")
        present["unresolved_quantity_units"] = pc.is_in(
            status, value_set=pa.array(["values_without_units", "count_mismatch"]))
        present["quantity_alignment_errors"] = pc.equal(status, "count_mismatch")
    else:
        present["unresolved_quantity_units"] = pa.array(np.zeros(len(batch), dtype=bool))
        present["quantity_alignment_errors"] = pa.array(np.zeros(len(batch), dtype=bool))
    present["unparsed_step_fields"] = (
        pc.equal(pc.fill_null(batch.column("text_status"), ""), "unparsed_steps")
        if "text_status" in batch.schema.names else pa.array(np.zeros(len(batch), dtype=bool)))
    sources = batch.column("source")
    if sources.null_count:
        raise ValueError("text index contains null source identities")
    out = {}
    for source in pc.unique(sources).to_pylist():
        if not source:
            raise ValueError("text index contains an empty source identity")
        belongs = pc.equal(sources, source)
        out[source] = {"rows": int(pc.sum(belongs).as_py())}
        for field, mask in present.items():
            out[source][field] = int(pc.sum(pc.and_(belongs, mask)).as_py())
    return out


def audit(path: Path, generation: dict, *, batch_size: int = 50_000) -> dict:
    if type(generation.get("recipes")) is not int or generation["recipes"] <= 0:
        raise ValueError("the declared corpus must contain a positive recipe count")
    before = path.stat()
    table = pq.ParquetFile(path)
    required = {"idx", "source", *FIELDS}
    missing = required - set(table.schema_arrow.names)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if table.metadata.num_rows != generation["recipes"]:
        raise ValueError("text-index row count differs from the declared corpus")
    version = int((table.schema_arrow.metadata or {}).get(b"text_schema_version", b"1"))
    if version not in (1, 2):
        raise ValueError(f"unsupported text schema version {version}")
    if version == 2 and not {"ingredient_quantities", "quantity_status"} <= set(table.schema_arrow.names):
        raise ValueError("text schema v2 is missing quantity metadata")
    per_source: dict[str, dict[str, int]] = defaultdict(
        lambda: dict.fromkeys(("rows", *COUNT_FIELDS), 0))
    offset = 0
    columns = ["idx", "source", *FIELDS]
    if "quantity_status" in table.schema_arrow.names:
        columns.append("quantity_status")
    if "text_status" in table.schema_arrow.names:
        columns.append("text_status")
    for batch in table.iter_batches(columns=columns, batch_size=batch_size):
        index = batch.column("idx")
        if index.null_count or not np.array_equal(
                index.to_numpy(), np.arange(offset, offset + len(batch))):
            raise ValueError(f"text-index row identity drift at offset {offset}")
        offset += len(batch)
        for source, counts in summarize_batch(batch).items():
            for field, count in counts.items():
                per_source[source][field] += count
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("text index changed during its audit")
    totals = {field: sum(counts[field] for counts in per_source.values())
              for field in ("rows", *COUNT_FIELDS)}
    return {
        "scope": "Text coverage screening, not a declaration of training readiness or data rights.",
        "corpus_generation": generation["generation"],
        "corpus_sha256": generation["sha256"],
        "text_index": {"filename": path.name, "bytes": after.st_size,
                       "sha256": digest.hexdigest(), "schema_version": version},
        "definitions": {
            "ingredient_letters": "Raw ingredient field contains a Unicode letter after removing an R c(...) wrapper and null/nan/none/NA tokens.",
            "complete_text": "Title, raw ingredients and steps are all nonempty.",
            "complete_with_ingredient_letters": "Complete text plus the ingredient-letter screen.",
            "unresolved_quantity_units": "Separate quantity values exist, but the source does not supply their units.",
            "quantity_alignment_errors": "Ingredient-name and separate quantity arrays differ in length and must not be paired.",
            "unparsed_step_fields": "Malformed instruction serialization is retained verbatim and explicitly excluded from generation-ready claims.",
        },
        "limitations": [
            "Nonempty or letter-containing text may still be malformed, quantity-only, or semantically incomplete.",
            "Sequential row IDs do not independently prove that source text was joined to the correct recipe.",
            "No recipe text is reproduced in this report.",
            "Licensing, near-duplicate contamination and culinary correctness are not established by coverage.",
        ],
        "totals": totals,
        "source_field_issues": [
            {
                "source": source,
                "raw_column": "RecipeIngredientQuantities",
                "rows_in_source": counts["rows"],
                "rows_failing_ingredient_letter_screen":
                    counts["raw_ingredients"] - counts["ingredient_letters"],
                "status": "Do not use as generation supervision until ingredient names and quantities are correctly rejoined.",
            }
            for source, counts in sorted(per_source.items())
            if version == 1 and source == "foodcom-522k"
        ] + [
            {
                "source": source,
                "rows_with_quantity_values_without_units": counts["unresolved_quantity_units"],
                "rows_with_quantity_count_mismatches": counts["quantity_alignment_errors"],
                "rows_with_unparsed_steps": counts["unparsed_step_fields"],
                "status": "Values are retained as supplied. Missing units must not be guessed, and mismatched arrays must not be paired.",
            }
            for source, counts in sorted(per_source.items())
            if counts["unresolved_quantity_units"]
        ],
        "by_source": [
            {"source": source, **counts}
            for source, counts in sorted(per_source.items(), key=lambda item: -item[1]["rows"])
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PATHS.recipes / TEXT_FILE)
    parser.add_argument("--out", type=Path, default=PATHS.results / "generation_data_audit.json")
    args = parser.parse_args()
    if args.out.suffix != ".json":
        parser.error("--out must end in .json")
    if args.out.resolve() == args.input.resolve():
        parser.error("audit output must not overwrite the input text index")
    try:
        generation = json.loads(PATHS.generation_file.read_text())
        verify_training_corpus(PATHS.data, generation["generation"])
        result = audit(args.input, generation)
        result["audit_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    except (OSError, ValueError, KeyError, pa.ArrowException) as error:
        print(f"generation data audit: {error}", file=sys.stderr)
        return 1
    n = result["totals"]["rows"]
    print(f"{n:,} indexed recipes; coverage is not a quality certificate.")
    for field, count in result["totals"].items():
        if field != "rows":
            print(f"  {field}: {count:,} ({count / n:.2%})")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
