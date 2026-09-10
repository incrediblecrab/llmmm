"""A compact ingredient-only catalog; building it does not authorize publication."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import numpy as np

from ._hashing import file_sha256
from .data.recipe_search_metadata import (
    _catalog_binding, _catalog_stamp, _numeric, _publish_directory, _read_connection,
)
from .recipe_ingredients import CanonicalIngredientIndex
from .recipe_search import _source_url

FORMAT = "llmmm-ingredient-catalog"
ARRAYS = {
    "ingredients": ("ingredients.u16.gz", "<u2"),
    "lengths": ("lengths.u16.gz", "<u2"),
    "total_minutes": ("total-minutes.f64.gz", "<f8"),
    "servings": ("servings.f64.gz", "<f8"),
    "source_codes": ("sources.u8.gz", "|u1"),
    "language_codes": ("languages.u8.gz", "|u1"),
    "has_source_url": ("source-links.u8.gz", "|u1"),
}
FORBIDDEN_FIELDS = (
    "title", "description", "instructions", "steps", "raw_ingredients",
    "ingredient_quantities", "author", "image", "photo",
)
_SENSITIVE_QUERY = re.compile(
    r"(?:access[_-]?token|auth|authorization|password|passwd|secret|api[_-]?key|"
    r"signature|credential|email|session|token)", re.I)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _gzip_file(path: Path, raw: bytes | memoryview) -> dict:
    with path.open("xb") as stream:
        with gzip.GzipFile(filename="", fileobj=stream, mode="wb", compresslevel=6, mtime=0) as archive:
            archive.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "file": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path),
        "raw_bytes": len(raw), "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "compression": "gzip",
    }


def public_source_url(value: object) -> tuple[str | None, str]:
    if value is None or value == "":
        return None, "missing"
    if not isinstance(value, str):
        raise ValueError("catalog source URLs must be strings or null")
    normalized = _source_url(value)
    if normalized is None or len(normalized.encode("utf-8")) > 4096:
        return None, "invalid_or_oversized"
    parts = urlsplit(normalized)
    if any(_SENSITIVE_QUERY.search(name) for name, _ in parse_qsl(parts.query, keep_blank_values=True)):
        return None, "sensitive_query_not_exported"
    return normalized, "source_url"


def build_ingredient_catalog(catalog_path: Path, corpus_path: Path, output: Path, *,
                             rows_per_shard: int = 16_384, progress=None) -> dict:
    """Export every canonical row, including records without readable instructions.

    All constraints use numeric source facts, not copied recipe prose. The result
    remains a local artifact until separate source permissions are established.
    """
    started = time.perf_counter()
    if type(rows_per_shard) is not int or not 1 <= rows_per_shard <= 65_536:
        raise ValueError("rows_per_shard must be an integer between 1 and 65536")
    if os.path.lexists(output):
        raise FileExistsError(f"{output}: ingredient-only builds never overwrite existing outputs")
    identity, catalog_stamp, metadata = _catalog_binding(catalog_path)
    index = CanonicalIngredientIndex.load(
        corpus_path, corpus_sha256=metadata["corpus_sha256"],
        n_recipes=metadata["n_recipes"], n_slots=metadata["n_slots"],
        n_vocab=len(metadata["vocabulary"]))
    n_rows, n_slots = metadata["n_recipes"], metadata["n_slots"]
    if n_rows > 2**32 - 1 or n_slots > 2**32 - 1:
        raise ValueError("this browser format requires uint32 record/slot offsets")
    lengths = np.diff(index.offsets).astype("<u2")
    arrays = {
        "ingredients": index.flat.astype("<u2", copy=False), "lengths": lengths,
        "total_minutes": np.full(n_rows, np.nan, dtype="<f8"),
        "servings": np.full(n_rows, np.nan, dtype="<f8"),
        "source_codes": np.empty(n_rows, dtype="u1"),
        "language_codes": np.empty(n_rows, dtype="u1"),
        "has_source_url": np.empty(n_rows, dtype="u1"),
    }
    frequencies = np.bincount(index.flat, minlength=len(metadata["vocabulary"]))
    if not np.array_equal(frequencies, np.asarray(metadata["ingredient_frequency"])):
        raise ValueError("canonical ingredient counts differ from catalog document frequencies")
    source_names, language_names = {}, {}
    url_statuses = Counter()
    rows_by_source = Counter()
    shard_records, shards = [], []
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ingredient-catalog-", dir=output.parent) as temporary:
        staged = Path(temporary) / "catalog"
        staged.mkdir(mode=0o700)
        urls_directory = staged / "urls"
        urls_directory.mkdir()
        seen = 0
        connection = _read_connection(catalog_path)
        try:
            cursor = connection.execute(
                "SELECT id, url, total_minutes, time_status, servings, servings_status, "
                "source, language, ingredient_ids FROM recipes ORDER BY id")
            while batch := cursor.fetchmany(8192):
                for recipe_id, url, minutes, time_status, servings, servings_status, source, language, ingredients in batch:
                    if recipe_id != seen or recipe_id >= n_rows:
                        raise ValueError("catalog record IDs are not complete, ordered and contiguous")
                    begin, end = index.offsets[recipe_id:recipe_id + 2]
                    if ingredients != index.flat[begin:end].astype("<u2", copy=False).tobytes():
                        raise ValueError(f"recipe {recipe_id}: SQLite ingredients differ from the canonical row")
                    for label, value, names, target in (
                            ("source", source, source_names, "source_codes"),
                            ("language", language, language_names, "language_codes")):
                        if not isinstance(value, str) or not value.strip() or len(value) > 128:
                            raise ValueError(f"recipe {recipe_id}: invalid {label}")
                        if value not in names:
                            if len(names) >= 256:
                                raise ValueError(f"this browser format supports at most 256 {label} codes")
                            names[value] = len(names)
                        arrays[target][recipe_id] = names[value]
                    arrays["total_minutes"][recipe_id] = _numeric(
                        minutes, time_status, "source_total", f"recipe {recipe_id} total_minutes")
                    arrays["servings"][recipe_id] = _numeric(
                        servings, servings_status, "source_servings", f"recipe {recipe_id} servings")
                    cleaned_url, status = public_source_url(url)
                    arrays["has_source_url"][recipe_id] = cleaned_url is not None
                    url_statuses[status] += 1
                    rows_by_source[source] += 1
                    shard_records.append(cleaned_url)
                    seen += 1
                    if len(shard_records) == rows_per_shard or seen == n_rows:
                        first = seen - len(shard_records)
                        name = f"{len(shards):04d}.json.gz"
                        data = _json_bytes({"first_id": first, "urls": shard_records})
                        record = _gzip_file(urls_directory / name, data)
                        record.update(file=f"urls/{name}", first_id=first, rows=len(shard_records))
                        shards.append(record)
                        shard_records = []
                if progress and (seen % (8192 * 32) == 0 or seen == n_rows):
                    progress({"phase": "source_metadata", "rows": seen, "total": n_rows})
        finally:
            connection.close()
        if seen != n_rows or shard_records:
            raise ValueError("catalog scan did not cover every declared canonical row")
        files = {}
        for name, array in arrays.items():
            filename, dtype = ARRAYS[name]
            if array.dtype != np.dtype(dtype):
                raise ValueError(f"{name}: unexpected serialized dtype")
            record = _gzip_file(staged / filename, memoryview(array).cast("B"))
            record.update(dtype=dtype, count=int(array.size))
            files[name] = record
        metadata_projection = {
            "schema_version": 1, "format": FORMAT, "endianness": "little",
            "n_recipes": n_rows, "n_slots": n_slots,
            "vocabulary": metadata["vocabulary"],
            "ingredient_frequency": frequencies.tolist(),
            "statistics_scope": "full-canonical-corpus",
            "source_names": list(source_names), "language_names": list(language_names),
            "arrays": files, "url_shards": shards, "rows_per_url_shard": rows_per_shard,
            "identity": {
                "corpus_sha256": identity["corpus_sha256"],
                "catalog_sha256": identity["sha256"],
                "vocabulary_sha256": identity["vocabulary_sha256"],
            },
            "coverage": {
                "records": seen, "ingredient_slots": n_slots,
                "min_ingredients": int(lengths.min()), "max_ingredients": int(lengths.max()),
                "singletons": int(np.count_nonzero(lengths == 1)),
                "source_total_times": int(np.count_nonzero(np.isfinite(arrays["total_minutes"]))),
                "source_servings": int(np.count_nonzero(np.isfinite(arrays["servings"]))),
                "url_statuses": dict(url_statuses), "rows_by_source": dict(rows_by_source),
            },
            "bytes": {
                "initial_compressed_download": sum(file["bytes"] for file in files.values()),
                "initial_uncompressed_arrays": sum(file["raw_bytes"] for file in files.values()),
                "derived_uint32_offsets": (n_rows + 1) * 4,
                "url_shards_compressed": sum(shard["bytes"] for shard in shards),
                "largest_url_shard_compressed": max(shard["bytes"] for shard in shards),
            },
            "fields_included": [
                "canonical_ingredient_ids", "source_total_minutes", "source_servings",
                "source_code", "language_code", "source_url",
            ],
            "fields_excluded": list(FORBIDDEN_FIELDS),
            "semantics": {
                "row_identity": "All canonical records in original ID order; duplicate sets are not removed.",
                "times": "Positive source-reported totals only; NaN means unknown; no component sums.",
                "servings": "Positive source-reported counts only; NaN means unknown; no quantity scaling.",
                "matching": "Canonical ingredient names, not quantities, every compound constituent or allergy safety.",
                "url_shards": "Only recipe IDs implied by order and original HTTP(S) URLs; no copied titles or prose.",
            },
            "publication": {
                "status": "local_export_only",
                "scope_clearance": "Building this artifact is not a grant of bulk redistribution rights.",
            },
        }
        (staged / "ingredient-index.json").write_bytes(_json_bytes(metadata_projection))
        index.check_unchanged()
        if _catalog_stamp(catalog_path) != catalog_stamp:
            raise ValueError("catalog changed during ingredient-only export")
        report = {
            "schema_version": 1, "status": "verified_local_ingredient_only_export",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "index_sha256": file_sha256(staged / "ingredient-index.json"),
            "coverage": metadata_projection["coverage"], "bytes": metadata_projection["bytes"],
            "all_rows_compared_with_canonical_arrays": True,
            "source_metadata_provenance_checked_per_row": True,
            "private_prose_exported": False, "uploaded": False,
        }
        (staged / "export-report.json").write_bytes(_json_bytes(report))
        _publish_directory(staged, output)
    return report


def load_ingredient_catalog(directory: Path) -> tuple[dict, dict[str, np.ndarray]]:
    metadata = json.loads((directory / "ingredient-index.json").read_text(encoding="utf-8"))
    if (metadata.get("schema_version") != 1 or metadata.get("format") != FORMAT
            or metadata.get("endianness") != "little"
            or metadata.get("statistics_scope") != "full-canonical-corpus"):
        raise ValueError("unsupported ingredient-only catalog")
    rows, slots = metadata["n_recipes"], metadata["n_slots"]
    if (type(rows) is not int or not 1 <= rows <= 10_000_000
            or type(slots) is not int or not rows <= slots <= 500_000_000):
        raise ValueError("invalid ingredient-only record/slot counts")
    for name, maximum in (("vocabulary", 65_536), ("source_names", 256), ("language_names", 256)):
        values = metadata.get(name)
        if (not isinstance(values, list) or not 1 <= len(values) <= maximum
                or not all(isinstance(value, str) and value.strip() and len(value) <= 128 for value in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"ingredient-only {name} must contain unique bounded strings")
    frequency = metadata["ingredient_frequency"]
    if (not isinstance(frequency, list) or len(frequency) != len(metadata["vocabulary"])
            or not all(type(count) is int and 0 <= count <= rows for count in frequency)):
        raise ValueError("ingredient-only document frequencies are invalid")
    if set(metadata["arrays"]) != set(ARRAYS):
        raise ValueError("ingredient-only catalog has an unexpected array inventory")
    shard_size = metadata["rows_per_url_shard"]
    shards = metadata["url_shards"]
    if (type(shard_size) is not int or not 1 <= shard_size <= 65_536
            or not isinstance(shards, list) or len(shards) != (rows + shard_size - 1) // shard_size):
        raise ValueError("source URL shards do not cover the declared population")
    for position, shard in enumerate(shards):
        if (shard["file"] != f"urls/{position:04d}.json.gz"
                or shard["first_id"] != position * shard_size
                or shard["rows"] != min(shard_size, rows - position * shard_size)
                or type(shard["bytes"]) is not int or not 0 < shard["bytes"] <= 16 * 1024 * 1024
                or type(shard["raw_bytes"]) is not int or not 0 < shard["raw_bytes"] <= 256 * 1024 * 1024
                or not re.fullmatch(r"[a-f0-9]{64}", str(shard["sha256"]))
                or not re.fullmatch(r"[a-f0-9]{64}", str(shard["raw_sha256"]))):
            raise ValueError("source URL shard path, identity or record boundaries are invalid")
    arrays = {}
    for name, (filename, dtype) in ARRAYS.items():
        record = metadata["arrays"][name]
        if (record["file"] != filename or record["dtype"] != dtype
                or type(record["count"]) is not int
                or record["count"] != (slots if name == "ingredients" else rows)):
            raise ValueError("ingredient-only array path or dtype differs from the format")
        path = directory / filename
        if path.stat().st_size != record["bytes"] or file_sha256(path) != record["sha256"]:
            raise ValueError(f"{filename}: compressed bytes failed their integrity check")
        expected = record["count"] * np.dtype(dtype).itemsize
        if expected != record["raw_bytes"] or not 0 < expected <= 2**31:
            raise ValueError("uncompressed ingredient-only array length is invalid")
        with gzip.open(path, "rb") as stream:
            data = stream.read(expected + 1)
        if len(data) != expected or hashlib.sha256(data).hexdigest() != record["raw_sha256"]:
            raise ValueError(f"{filename}: decompressed bytes failed their integrity check")
        arrays[name] = np.frombuffer(data, dtype=dtype)
    if (len(arrays["ingredients"]) != slots
            or any(len(array) != rows for name, array in arrays.items() if name != "ingredients")
            or np.any(arrays["lengths"] == 0)
            or int(arrays["lengths"].sum()) != slots):
        raise ValueError("ingredient-only array shapes do not describe the complete population")
    if (np.any(arrays["ingredients"] >= len(metadata["vocabulary"]))
            or np.any(arrays["source_codes"] >= len(metadata["source_names"]))
            or np.any(arrays["language_codes"] >= len(metadata["language_names"]))
            or np.any(arrays["has_source_url"] > 1)):
        raise ValueError("ingredient-only arrays contain unknown ingredient or metadata codes")
    offsets = np.r_[0, np.cumsum(arrays["lengths"], dtype=np.uint64)]
    invalid_order = arrays["ingredients"][1:] <= arrays["ingredients"][:-1]
    invalid_order[offsets[1:-1] - 1] = False
    if invalid_order.any() or not np.array_equal(
            np.bincount(arrays["ingredients"], minlength=len(frequency)), np.asarray(frequency)):
        raise ValueError("ingredient-only rows are not unique sorted sets with the declared frequencies")
    for name in ("total_minutes", "servings"):
        values = arrays[name]
        if np.any(~np.isnan(values) & (~np.isfinite(values) | (values <= 0))):
            raise ValueError(f"ingredient-only {name} must be positive source values or unknown")
    return metadata, arrays
