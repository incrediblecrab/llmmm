"""Package a complete ingredient-only index locally, without training or uploading."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import closing
from itertools import islice
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ._hashing import file_sha256
from .data.recipe_search_metadata import _publish_directory
from .ingredient_catalog import (
    ARRAYS, FORBIDDEN_FIELDS, _json_bytes, load_ingredient_catalog, public_source_url,
)

ROWS_PER_SHARD = 262_144
ROWS_PER_BATCH = 16_384
SLOTS_PER_BATCH = 262_144
PARQUET_SCHEMA = pa.schema([
    pa.field("id", pa.uint32(), nullable=False),
    pa.field("ingredient_ids", pa.list_(pa.field("element", pa.uint16(), nullable=False)),
             nullable=False),
    pa.field("ingredients", pa.list_(pa.field("element", pa.string(), nullable=False)),
             nullable=False),
    pa.field("source", pa.string(), nullable=False),
    pa.field("language", pa.string(), nullable=False),
    pa.field("total_minutes", pa.float64()),
    pa.field("servings", pa.float64()),
    pa.field("source_url", pa.string()),
])
_INDEX_FIELDS = {
    "schema_version", "format", "endianness", "n_recipes", "n_slots", "vocabulary",
    "ingredient_frequency", "statistics_scope", "source_names", "language_names",
    "arrays", "url_shards", "rows_per_url_shard", "identity", "coverage", "bytes",
    "fields_included", "fields_excluded", "semantics", "publication",
}
_BLOB_FIELDS = {"file", "bytes", "sha256", "raw_bytes", "raw_sha256", "compression"}
_INCLUDED_FIELDS = [
    "canonical_ingredient_ids", "source_total_minutes", "source_servings",
    "source_code", "language_code", "source_url",
]
_URL_STATUSES = {"source_url", "missing", "invalid_or_oversized", "sensitive_query_not_exported"}
_SOURCE_REPOSITORY = "https://github.com/incrediblecrab/llmmm"
_MODEL_URL = "https://huggingface.co/incrediblecrab/llmmm-recipes"
_DEMO_URL = "https://huggingface.co/spaces/incrediblecrab/llmmm-recipes-demo"


def _unique_fields(pairs: list[tuple[str, object]]) -> dict:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate fields in ingredient-only JSON")
    return result


def _require_fields(value: object, fields: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label}: unexpected or missing fields")


def _validate_blob_record(record: dict, fields: set[str], filename: str) -> None:
    _require_fields(record, _BLOB_FIELDS | fields, filename)
    if record["file"] != filename or record["compression"] != "gzip":
        raise ValueError(f"{filename}: unexpected compressed file identity")
    for name in ("bytes", "raw_bytes"):
        if type(record[name]) is not int or record[name] <= 0:
            raise ValueError(f"{filename}: invalid {name}")
    for name in ("sha256", "raw_sha256"):
        if not isinstance(record[name], str) or not re.fullmatch(r"[a-f0-9]{64}", record[name]):
            raise ValueError(f"{filename}: invalid {name}")


def _validate_projection(metadata: dict) -> None:
    _require_fields(metadata, _INDEX_FIELDS, "ingredient index")
    _require_fields(metadata["arrays"], set(ARRAYS), "array inventory")
    for name, (filename, _) in ARRAYS.items():
        _validate_blob_record(metadata["arrays"][name], {"dtype", "count"}, filename)
    if not isinstance(metadata["url_shards"], list):
        raise ValueError("source URL shard inventory must be a list")
    for position, record in enumerate(metadata["url_shards"]):
        _validate_blob_record(record, {"first_id", "rows"}, f"urls/{position:04d}.json.gz")
        if type(record["first_id"]) is not int or type(record["rows"]) is not int:
            raise ValueError("source URL shard boundaries must be integers")
    _require_fields(metadata["identity"],
                    {"corpus_sha256", "catalog_sha256", "vocabulary_sha256"}, "corpus identity")
    if any(not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value)
           for value in metadata["identity"].values()):
        raise ValueError("invalid corpus identity hashes")
    _require_fields(metadata["coverage"], {
        "records", "ingredient_slots", "min_ingredients", "max_ingredients", "singletons",
        "source_total_times", "source_servings", "url_statuses", "rows_by_source",
    }, "ingredient coverage")
    _require_fields(metadata["bytes"], {
        "initial_compressed_download", "initial_uncompressed_arrays", "derived_uint32_offsets",
        "url_shards_compressed", "largest_url_shard_compressed",
    }, "index byte statistics")
    _require_fields(metadata["semantics"],
                    {"row_identity", "times", "servings", "matching", "url_shards"}, "index semantics")
    if any(not isinstance(value, str) or not value.strip() or len(value) > 1024
           for value in metadata["semantics"].values()):
        raise ValueError("index semantics must be bounded strings")
    if (metadata["fields_included"] != _INCLUDED_FIELDS
            or metadata["fields_excluded"] != list(FORBIDDEN_FIELDS)):
        raise ValueError("unexpected ingredient-only field declarations")
    _require_fields(metadata["publication"], {"status", "scope_clearance"}, "input publication")
    if (metadata["publication"]["status"] != "local_export_only"
            or not isinstance(metadata["publication"]["scope_clearance"], str)):
        raise ValueError("expected an immutable local-export ingredient index")


def _validate_coverage(metadata: dict, arrays: dict[str, np.ndarray]) -> None:
    rows = metadata["n_recipes"]
    coverage = metadata["coverage"]
    observed = {
        "records": rows,
        "ingredient_slots": metadata["n_slots"],
        "min_ingredients": int(arrays["lengths"].min()),
        "max_ingredients": int(arrays["lengths"].max()),
        "singletons": int(np.count_nonzero(arrays["lengths"] == 1)),
        "source_total_times": int(np.count_nonzero(np.isfinite(arrays["total_minutes"]))),
        "source_servings": int(np.count_nonzero(np.isfinite(arrays["servings"]))),
    }
    if any(type(coverage[name]) is not int or coverage[name] != count
           for name, count in observed.items()):
        raise ValueError("ingredient coverage differs from the complete numeric arrays")
    statuses = coverage["url_statuses"]
    if (not isinstance(statuses, dict) or set(statuses) - _URL_STATUSES
            or any(type(count) is not int or count < 0 for count in statuses.values())
            or sum(statuses.values()) != rows
            or statuses.get("source_url", 0) != int(arrays["has_source_url"].sum())):
        raise ValueError("source URL coverage differs from the complete link flags")
    counts = np.bincount(arrays["source_codes"], minlength=len(metadata["source_names"]))
    by_source = {name: int(count) for name, count in zip(metadata["source_names"], counts)}
    if (not isinstance(coverage["rows_by_source"], dict)
            or any(type(count) is not int for count in coverage["rows_by_source"].values())
            or coverage["rows_by_source"] != by_source):
        raise ValueError("source coverage differs from the complete source codes")
    array_records, shards = metadata["arrays"].values(), metadata["url_shards"]
    expected_bytes = {
        "initial_compressed_download": sum(record["bytes"] for record in array_records),
        "initial_uncompressed_arrays": sum(record["raw_bytes"] for record in array_records),
        "derived_uint32_offsets": (rows + 1) * 4,
        "url_shards_compressed": sum(record["bytes"] for record in shards),
        "largest_url_shard_compressed": max(record["bytes"] for record in shards),
    }
    if (any(type(value) is not int for value in metadata["bytes"].values())
            or metadata["bytes"] != expected_bytes):
        raise ValueError("index byte statistics differ from the declared files")


def _regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{path.name}: ingredient inputs and outputs must be regular files, not symlinks")


def _check_compressed_file(path: Path, record: dict) -> None:
    _regular_file(path)
    if path.stat().st_size != record["bytes"] or file_sha256(path) != record["sha256"]:
        raise ValueError(f"{record['file']}: compressed bytes failed their integrity check")


def _read_url_shard(directory: Path, record: dict) -> list[str | None]:
    path = directory / record["file"]
    _check_compressed_file(path, record)
    try:
        with gzip.open(path, "rb") as stream:
            raw = stream.read(record["raw_bytes"] + 1)
    except (OSError, EOFError) as error:
        raise ValueError(f"{record['file']}: invalid gzip URL shard") from error
    if len(raw) != record["raw_bytes"] or hashlib.sha256(raw).hexdigest() != record["raw_sha256"]:
        raise ValueError(f"{record['file']}: decompressed bytes failed their integrity check")
    shard = json.loads(raw, object_pairs_hook=_unique_fields)
    _require_fields(shard, {"first_id", "urls"}, "source URL shard")
    if (type(shard["first_id"]) is not int or shard["first_id"] != record["first_id"]
            or not isinstance(shard["urls"], list) or len(shard["urls"]) != record["rows"]):
        raise ValueError(f"{record['file']}: source URL shard alignment differs from the index")
    return shard["urls"]


def _source_urls(directory: Path, metadata: dict, flags: np.ndarray) -> Iterator[str | None]:
    for record in metadata["url_shards"]:
        urls = _read_url_shard(directory, record)
        for offset, value in enumerate(urls):
            normalized, status = public_source_url(value)
            if value is not None and (status != "source_url" or normalized != value):
                raise ValueError(f"{record['file']}: source URL is unsafe or not already normalized")
            if value is not None and urlsplit(value).username is not None:
                raise ValueError(f"{record['file']}: source URL contains credentials")
            if bool(flags[record["first_id"] + offset]) != (value is not None):
                raise ValueError(f"{record['file']}: source URL differs from its has_source_url flag")
        yield from urls
        del urls


def _write_parquet(directory: Path, metadata: dict, arrays: dict[str, np.ndarray]) -> tuple[dict, list[str]]:
    rows = metadata["n_recipes"]
    shard_count = (rows + ROWS_PER_SHARD - 1) // ROWS_PER_SHARD
    vocabulary = pa.array(metadata["vocabulary"], type=pa.string())
    sources = pa.array(metadata["source_names"], type=pa.string())
    languages = pa.array(metadata["language_names"], type=pa.string())
    counts = {"rows": 0, "ingredient_slots": 0,
              "null_counts": {field.name: 0 for field in PARQUET_SCHEMA}}
    files = []
    slot = 0
    with closing(_source_urls(directory / "index", metadata, arrays["has_source_url"])) as urls:
        for position in range(shard_count):
            first = position * ROWS_PER_SHARD
            last = min(first + ROWS_PER_SHARD, rows)
            filename = f"data/train-{position:05d}-of-{shard_count:05d}.parquet"
            path = directory / filename
            with pq.ParquetWriter(path, PARQUET_SCHEMA, compression="zstd",
                                  write_batch_size=ROWS_PER_BATCH) as writer:
                start = first
                while start < last:
                    stop = min(start + ROWS_PER_BATCH, last)
                    offsets = np.empty(stop - start + 1, dtype=np.int32)
                    offsets[0] = 0
                    np.cumsum(arrays["lengths"][start:stop], dtype=np.int32, out=offsets[1:])
                    batch_rows = int(np.searchsorted(offsets[1:], SLOTS_PER_BATCH, side="right"))
                    if batch_rows == 0:
                        raise ValueError("a canonical row exceeds the ingredient-slot batch bound")
                    offsets = offsets[:batch_rows + 1]
                    stop = start + batch_rows
                    end_slot = slot + int(offsets[-1])
                    ingredient_ids = pa.array(arrays["ingredients"][slot:end_slot], type=pa.uint16())
                    batch_urls = list(islice(urls, batch_rows))
                    if len(batch_urls) != batch_rows:
                        raise ValueError("source URL shards ended before the canonical rows")
                    batch = pa.RecordBatch.from_arrays([
                        pa.array(np.arange(start, stop, dtype=np.uint32)),
                        pa.ListArray.from_arrays(offsets, ingredient_ids,
                                                 type=PARQUET_SCHEMA.field("ingredient_ids").type),
                        pa.ListArray.from_arrays(offsets, pc.take(vocabulary, ingredient_ids),
                                                 type=PARQUET_SCHEMA.field("ingredients").type),
                        pc.take(sources, pa.array(arrays["source_codes"][start:stop])),
                        pc.take(languages, pa.array(arrays["language_codes"][start:stop])),
                        *(pa.array(arrays[name][start:stop], type=pa.float64(),
                                   mask=np.isnan(arrays[name][start:stop]))
                          for name in ("total_minutes", "servings")),
                        pa.array(batch_urls, type=pa.string()),
                    ], schema=PARQUET_SCHEMA)
                    writer.write_batch(batch, row_group_size=batch.num_rows)
                    counts["rows"] += batch.num_rows
                    counts["ingredient_slots"] += len(ingredient_ids)
                    for field, column in zip(PARQUET_SCHEMA, batch.columns):
                        counts["null_counts"][field.name] += column.null_count
                    start, slot = stop, end_slot
            with pq.ParquetFile(path) as parquet:
                if (parquet.metadata.num_rows != last - first
                        or not parquet.schema_arrow.equals(PARQUET_SCHEMA)):
                    raise ValueError(f"{filename}: written Parquet rows or schema differ")
            files.append(filename)
        sentinel = object()
        if next(urls, sentinel) is not sentinel:
            raise ValueError("source URL shards contain extra records")
    expected_nulls = {field.name: 0 for field in PARQUET_SCHEMA}
    expected_nulls.update(
        total_minutes=rows - metadata["coverage"]["source_total_times"],
        servings=rows - metadata["coverage"]["source_servings"],
        source_url=rows - metadata["coverage"]["url_statuses"].get("source_url", 0),
    )
    if (counts["rows"] != rows or counts["ingredient_slots"] != metadata["n_slots"]
            or counts["null_counts"] != expected_nulls):
        raise ValueError("Parquet did not retain every canonical row, ingredient slot and source fact")
    return {
        **counts, "split": "train", "compression": "zstd", "shards": shard_count,
        "max_rows_per_shard": ROWS_PER_SHARD, "max_rows_per_batch": ROWS_PER_BATCH,
        "max_ingredient_slots_per_batch": SLOTS_PER_BATCH,
    }, files


def _dataset_card(metadata: dict, source_revision: str) -> str:
    rows = metadata["n_recipes"]
    coverage = metadata["coverage"]
    links = coverage["url_statuses"].get("source_url", 0)
    return f"""---
license: other
pretty_name: llmmm Recipe Ingredients
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/*.parquet
---

# llmmm recipe ingredients

This factual extract contains **{rows:,} canonical ingredient records**,
**{metadata['n_slots']:,} ingredient slots** and **{len(metadata['vocabulary']):,} canonical
ingredient names** from **{len(metadata['source_names']):,} source groups**. These are
normalized ingredient facts, not complete recipes. Counts describe the complete
canonical corpus, not a sample or a count of unique content: duplicate ingredient
sets remain, and a record is not necessarily a unique or complete recipe.

The `train` split preserves every original zero-based `id`, including
{coverage['singletons']:,} singleton records and records without links.
`ingredient_ids` and `ingredients` are aligned, sorted canonical ID sets and their
exact vocabulary names. `source` and `language` retain the index's source
identifiers and language labels.

Source-reported positive `total_minutes` are available for
**{coverage['source_total_times']:,} of {rows:,} records**; source-reported `servings`
for **{coverage['source_servings']:,} of {rows:,} records**. Unknown values are null,
not NaN, zero, guessed totals or sums of component times. Serving counts do not
scale ingredient quantities. These source facts are not independently measured.

`source_url` retains recorded original HTTP(S) links for **{links:,} records**;
**{rows - links:,} records have null URLs**. Missing, invalid or sensitive links
are not replaced with generated URLs. Original titles, descriptions, ingredient
prose, quantities, instructions, authors and images are not included. Consult a
recorded source page for the complete recipe, where a link is available.

Canonical matching does not recover quantities or every compound constituent,
and exclusions are not an allergy-safety guarantee.

[Model and its separate terms]({_MODEL_URL}) |
[Browser demo]({_DEMO_URL}) |
[Exact source revision `{source_revision}`]({_SOURCE_REPOSITORY}/tree/{source_revision})

`index/` preserves the ingredient index's compressed arrays and URL shards
byte-for-byte; only its publication provenance changes. `dataset-manifest.json`
records the source revision, input and copied index hashes, Parquet counts and
every other file's bytes and SHA256. Packaging stages a release locally; it does
not upload it.

## Use

This is a factual ingredient extract. The maintainer confirmed permission to
publish this ingredient-only data. `license: other` is not a grant of rights to
source-page prose or photos, or to the separate model weights. Source-page
material and model weights retain their separate terms.

## Acknowledgements

Thanks to the original recipe contributors and the dataset contributors and
curators identified in the
[source inventory]({_SOURCE_REPOSITORY}/blob/{source_revision}/raw-data/README.md).
Original source identifiers and recorded source URLs are retained for attribution
and provenance wherever available.
"""


def _inventory(directory: Path, filenames: list[str]) -> dict[str, dict]:
    actual = set()
    for path in directory.rglob("*"):
        relative = path.relative_to(directory).as_posix()
        if path.is_symlink():
            raise ValueError("dataset output may not contain symlinks")
        if path.is_dir():
            if relative not in {"index", "index/urls", "data"}:
                raise ValueError("dataset output contains an unexpected directory")
        else:
            _regular_file(path)
            actual.add(relative)
    if actual != set(filenames):
        raise ValueError("dataset output differs from the fixed file allowlist")
    return {name: {"bytes": (directory / name).stat().st_size,
                   "sha256": file_sha256(directory / name)} for name in sorted(actual)}


def build_ingredient_dataset(index_directory: Path, output: Path, *, source_revision: str) -> dict:
    """Atomically stage all ingredient facts in a new, never-overwritten directory.

    The validated compact NumPy arrays remain resident. Expanded ingredient names
    use bounded Arrow batches; URLs use one decoded shard plus one batch, never a
    corpus-sized Python list. This function neither trains nor accesses the network.
    """
    if not isinstance(source_revision, str) or not re.fullmatch(r"[a-f0-9]{40}", source_revision):
        raise ValueError("source_revision must be an exact 40-character lowercase git SHA")
    index_directory, output = Path(index_directory).resolve(), Path(output)
    if os.path.lexists(output):
        raise FileExistsError(f"{output}: ingredient datasets never overwrite existing outputs")
    if output.resolve().is_relative_to(index_directory):
        raise ValueError("dataset output must be outside the immutable input index")
    index_path = index_directory / "ingredient-index.json"
    _regular_file(index_path)
    if index_path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("ingredient index manifest exceeds the 16 MiB bound")
    input_index_sha256 = file_sha256(index_path)
    metadata = json.loads(index_path.read_bytes(), object_pairs_hook=_unique_fields)
    _validate_projection(metadata)
    for filename, _ in ARRAYS.values():
        _regular_file(index_directory / filename)
    if (index_directory / "urls").is_symlink() or not (index_directory / "urls").is_dir():
        raise ValueError("the source URL directory must be a directory, not a symlink")
    loaded_metadata, arrays = load_ingredient_catalog(index_directory)
    if loaded_metadata != metadata or file_sha256(index_path) != input_index_sha256:
        raise ValueError("input ingredient index changed during validation")
    _validate_coverage(metadata, arrays)
    publication = {
        "status": "public_ingredient_only_release_staging",
        "scope": "public-ingredient-only-release",
        "source_revision": source_revision,
        "input_index_sha256": input_index_sha256,
        "maintainer_confirmed_permission": True,
        "private_prose_exported": False,
        "uploaded": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ingredient-dataset-", dir=output.parent) as temporary:
        staged = Path(temporary) / "dataset"
        (staged / "index/urls").mkdir(parents=True, mode=0o700)
        (staged / "data").mkdir()
        filenames = ["README.md", "index/ingredient-index.json"]
        for record in [*metadata["arrays"].values(), *metadata["url_shards"]]:
            source, destination = index_directory / record["file"], staged / "index" / record["file"]
            _regular_file(source)
            shutil.copyfile(source, destination)
            _check_compressed_file(destination, record)
            filenames.append("index/" + record["file"])
        copied_index = staged / "index/ingredient-index.json"
        copied_index.write_bytes(_json_bytes({**metadata, "publication": publication}))
        parquet, parquet_files = _write_parquet(staged, metadata, arrays)
        (staged / "README.md").write_text(_dataset_card(metadata, source_revision), encoding="utf-8")
        schema = []
        for field in PARQUET_SCHEMA:
            item = {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            if pa.types.is_list(field.type):
                item.update(type=f"list<{field.type.value_type}>",
                            item_nullable=field.type.value_field.nullable)
            schema.append(item)
        manifest = {
            "schema_version": 1,
            "format": "llmmm-ingredient-dataset",
            **publication,
            "copied_index_sha256": file_sha256(copied_index),
            "identity": metadata["identity"],
            "statistics_scope": metadata["statistics_scope"],
            "n_recipes": metadata["n_recipes"], "n_slots": metadata["n_slots"],
            "n_vocab": len(metadata["vocabulary"]),
            "source_names": metadata["source_names"], "language_names": metadata["language_names"],
            "coverage": metadata["coverage"], "parquet": parquet, "output_schema": schema,
            "files": _inventory(staged, filenames + parquet_files),
        }
        (staged / "dataset-manifest.json").write_bytes(_json_bytes(manifest))
        if file_sha256(index_path) != input_index_sha256:
            raise ValueError("input ingredient index changed during packaging")
        _publish_directory(staged, output)
    return manifest
