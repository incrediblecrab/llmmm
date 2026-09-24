"""A compact recipe-search catalog with titles and ingredient lines; building it does not authorize publication."""
from __future__ import annotations

import ast
import gzip
import hashlib
import html
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
from .recipe_links import LINK_RULE, LINK_STATUSES, SITE_RULES, recipe_link
from .recipe_search import _source_url

FORMAT = "llmmm-ingredient-catalog"
ARRAYS = {
    "ingredients": ("ingredients.u16.gz", "<u2"),
    "lengths": ("lengths.u16.gz", "<u2"),
    "total_minutes": ("total-minutes.f64.gz", "<f8"),
    "servings": ("servings.f64.gz", "<f8"),
    "source_codes": ("sources.u8.gz", "|u1"),
    "language_codes": ("languages.u8.gz", "|u1"),
    "link_status": ("link-status.u8.gz", "|u1"),
}
FORBIDDEN_FIELDS = (
    "description", "instructions", "steps", "ingredient_quantities", "author", "image", "photo",
)
SCHEMA_VERSION = 3
INCLUDED_FIELDS = (
    "canonical_ingredient_ids", "source_total_minutes", "source_servings",
    "source_code", "language_code", "source_url", "recipe_link", "link_status", "recipe_title", "ingredient_lines",
)
LINK_SEMANTICS = (
    f"The page each result card opens, in the same ID order and shards as the recorded URLs. {LINK_RULE} "
    "link_status codes 0-3 are none, source, archive and offline: no recorded URL, a link to the recorded page, "
    "a link to its archived copy, and no link because the site passed neither check."
)
TEXT_COVERAGE = ("recipe_titles", "ingredient_line_records", "ingredient_lines")
TEXT_SEMANTICS = (
    "Recorded titles and ingredient lines, one line per catalog separator or embedded line break. Complete HTML "
    "character references are decoded, whitespace is collapsed, HTML ingredient tables become one 'name: amount' "
    "line per row, lines recorded as name and amount fields (povarenok-detail, taiwan-1.8k) become 'name: amount', "
    "a 03-povarenok amount recorded as null (catalog text 'name: None') is dropped, "
    "and a whole list recorded on one line becomes one line per item (joined with ' ; ', or '|' in filipino-2k); "
    "wording is otherwise unchanged. Null means the source recorded none. No instructions."
)
TEXT_MANIFEST = "text-shards.json.gz"
MAX_TEXT_BYTES = 16_384
MAX_LINES = 2_048
_SENSITIVE_QUERY = re.compile(
    r"(?:access[_-]?token|auth|authorization|password|passwd|secret|api[_-]?key|"
    r"signature|credential|email|session|token)", re.I)
_CHARACTER_REFERENCE = re.compile(r"&(?:#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});")
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_HTML_TABLE = re.compile(r"\s*<table[\s>]", re.I)
_TABLE_ROW = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.I | re.S)
_TABLE_CELL = re.compile(r"<t[hd]\b[^>]*>(.*?)</t[hd]>", re.I | re.S)
_TAG = re.compile(r"<[^>]*>")
# povarenok-detail and taiwan-1.8k record each line as a Python dict literal of these fields.
_FIELD_LINE = re.compile(r"\{'(?:count|name)': .*\}")
_FIELD_KEYS = ({"count", "name"}, {"name", "unit"})
# The catalog joins each recorded 03-povarenok {name: amount} pair as "name: amount", so a null amount reads "None".
_NULL_AMOUNT = re.compile(r"\s*: None$")


def _recorded_fields(value: str) -> tuple[str | None, str | None] | None:
    """(name, amount) of a line recorded as a field dict, or None for any other line."""
    if not _FIELD_LINE.fullmatch(value.strip()):
        return None
    try:
        fields = ast.literal_eval(value.strip())
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None
    if (not isinstance(fields, dict) or set(fields) not in _FIELD_KEYS
            or not all(item is None or isinstance(item, str) for item in fields.values())):
        return None
    return fields["name"], fields.get("count", fields.get("unit"))


def _line_text(value: str) -> str:
    fields = _recorded_fields(value)
    if fields is None:
        return _display_text(value)
    return ": ".join(filter(None, (_display_text(item or "") for item in fields)))


def _display_text(value: str) -> str:
    # Only complete character references are decoded, so text such as "PB&J;" or "salt &pepper" is kept.
    decoded = _CHARACTER_REFERENCE.sub(lambda match: html.unescape(match.group(0)), value)
    return " ".join(_CONTROL.sub(" ", decoded).split())


def recipe_title(value: object) -> str | None:
    """The recorded title with character references decoded and whitespace collapsed."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("catalog titles must be strings or null")
    return _display_text(value) or None


def ingredient_lines(value: object, source: str | None = None) -> list[str] | None:
    """The recorded ingredient lines, one per catalog separator or embedded line break.

    Wording is unchanged apart from decoded character references and collapsed
    whitespace; an HTML ingredient table becomes one "name: amount" line per row,
    a line recorded as a {'name', 'count' or 'unit'} dict becomes "name: amount",
    a 03-povarenok "name: None" line (a null amount) becomes "name",
    and a whole list recorded as one line becomes one line per item: joined with
    " ; ", or with "|" in filipino-2k.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("catalog ingredient lines must be strings or null")
    lines = []
    elements = value.split("\x1f")
    for element in elements:
        if _HTML_TABLE.match(element):
            rows = _TABLE_ROW.findall(element)
            texts = [": ".join(filter(None, (_display_text(_TAG.sub(" ", cell)) for cell in _TABLE_CELL.findall(row))))
                     for row in rows] if rows else [_display_text(_TAG.sub(" ", element))]
        else:
            parts = element.splitlines()
            # Only a whole list on one line is split: elsewhere these marks are ordinary punctuation or part of a name.
            if len(elements) == 1 and len(parts) == 1 and _recorded_fields(parts[0]) is None:
                parts = parts[0].split("|" if source == "filipino-2k" else " ; ")
            texts = [_line_text(part) for part in parts]
            if source == "03-povarenok":
                texts = [_NULL_AMOUNT.sub("", text) for text in texts]
        lines.extend(text for text in texts if text)
    return lines or None


def _bounded_text(value: object) -> bool:
    return (isinstance(value, str) and value == value.strip() and bool(value)
            and len(value.encode("utf-8")) <= MAX_TEXT_BYTES and not _CONTROL.search(value))


def _check_text(title: object, lines: object, label: str) -> None:
    if title is not None and not _bounded_text(title):
        raise ValueError(f"{label}: a title must be null or bounded single-line text")
    if lines is not None and (not isinstance(lines, list) or not 1 <= len(lines) <= MAX_LINES
                              or not all(_bounded_text(line) for line in lines)):
        raise ValueError(f"{label}: ingredient lines must be null or 1-{MAX_LINES} bounded single-line strings")


def _checked_gzip_json(path: Path, record: dict, label: str, *, maximum: int = 256 * 1024 * 1024) -> dict:
    if (type(record.get("bytes")) is not int or not 0 < record["bytes"] <= 64 * 1024 * 1024
            or type(record.get("raw_bytes")) is not int or not 0 < record["raw_bytes"] <= maximum
            or not re.fullmatch(r"[a-f0-9]{64}", str(record.get("sha256")))
            or not re.fullmatch(r"[a-f0-9]{64}", str(record.get("raw_sha256")))
            or record.get("compression") != "gzip"):
        raise ValueError(f"{label}: invalid size, digest or compression declaration")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label}: index files must be regular files, not symlinks")
    if path.stat().st_size != record["bytes"] or file_sha256(path) != record["sha256"]:
        raise ValueError(f"{label}: compressed bytes failed their integrity check")
    try:
        with gzip.open(path, "rb") as stream:
            raw = stream.read(record["raw_bytes"] + 1)
    except (OSError, EOFError) as error:
        raise ValueError(f"{label}: invalid gzip data") from error
    if len(raw) != record["raw_bytes"] or hashlib.sha256(raw).hexdigest() != record["raw_sha256"]:
        raise ValueError(f"{label}: decompressed bytes failed their integrity check")
    return json.loads(raw)


def load_text_shards(directory: Path, metadata: dict) -> list[dict]:
    """Validate the text-shard manifest declared by the index and return its shard records."""
    record, rows = metadata.get("text_manifest"), metadata["n_recipes"]
    size = metadata.get("rows_per_text_shard")
    if (not isinstance(record, dict) or record.get("file") != TEXT_MANIFEST
            or type(size) is not int or not 1 <= size <= 65_536
            or record.get("shards") != (rows + size - 1) // size):
        raise ValueError("the text-shard manifest declaration does not cover the population")
    manifest = _checked_gzip_json(directory / TEXT_MANIFEST, record, TEXT_MANIFEST, maximum=64 * 1024 * 1024)
    shards = manifest.get("shards") if isinstance(manifest, dict) and set(manifest) == {"shards"} else None
    if not isinstance(shards, list) or len(shards) != record["shards"]:
        raise ValueError("the text-shard manifest has an unexpected structure")
    for position, shard in enumerate(shards):
        if (not isinstance(shard, dict) or set(shard) != {
                "file", "first_id", "rows", "bytes", "sha256", "raw_bytes", "raw_sha256", "compression"}
                or shard["file"] != f"text/{position:04d}.json.gz"
                or shard["first_id"] != position * size
                or shard["rows"] != min(size, rows - position * size)
                or type(shard["bytes"]) is not int or not 0 < shard["bytes"] <= 16 * 1024 * 1024
                or type(shard["raw_bytes"]) is not int or not 0 < shard["raw_bytes"] <= 256 * 1024 * 1024
                or not re.fullmatch(r"[a-f0-9]{64}", str(shard["sha256"]))
                or not re.fullmatch(r"[a-f0-9]{64}", str(shard["raw_sha256"]))
                or shard["compression"] != "gzip"):
            raise ValueError("text shard path, identity or record boundaries are invalid")
    return shards


def read_text_shard(directory: Path, record: dict) -> tuple[list, list]:
    """Return one shard's aligned titles and ingredient lines after checking both digests."""
    shard = _checked_gzip_json(directory / record["file"], record, record["file"])
    if (not isinstance(shard, dict) or set(shard) != {"first_id", "titles", "ingredient_lines"}
            or shard["first_id"] != record["first_id"]
            or not isinstance(shard["titles"], list) or len(shard["titles"]) != record["rows"]
            or not isinstance(shard["ingredient_lines"], list) or len(shard["ingredient_lines"]) != record["rows"]):
        raise ValueError(f"{record['file']}: text shard alignment differs from the index")
    for offset, (title, lines) in enumerate(zip(shard["titles"], shard["ingredient_lines"])):
        _check_text(title, lines, f"record {record['first_id'] + offset}")
    return shard["titles"], shard["ingredient_lines"]


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
                             rows_per_shard: int = 16_384, rows_per_text_shard: int = 2_048,
                             link_rules=SITE_RULES, progress=None) -> dict:
    """Export every canonical row, including records without readable instructions.

    Constraints use numeric source facts. Titles and ingredient lines are copied for
    display; instructions, descriptions, authors and images are not. Card links come
    from the measured per-site rules. The result remains a local artifact until
    publication is separately authorized.
    """
    started = time.perf_counter()
    for label, value in (("rows_per_shard", rows_per_shard), ("rows_per_text_shard", rows_per_text_shard)):
        if type(value) is not int or not 1 <= value <= 65_536:
            raise ValueError(f"{label} must be an integer between 1 and 65536")
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
        "link_status": np.empty(n_rows, dtype="u1"),
    }
    frequencies = np.bincount(index.flat, minlength=len(metadata["vocabulary"]))
    if not np.array_equal(frequencies, np.asarray(metadata["ingredient_frequency"])):
        raise ValueError("canonical ingredient counts differ from catalog document frequencies")
    source_names, language_names = {}, {}
    url_statuses, link_statuses = Counter(), Counter()
    rows_by_source = Counter()
    shard_records, shards = [], []
    link_records, link_shards = [], []
    titles, line_lists, text_shards = [], [], []
    text_counts = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ingredient-catalog-", dir=output.parent) as temporary:
        staged = Path(temporary) / "catalog"
        staged.mkdir(mode=0o700)
        urls_directory = staged / "urls"
        urls_directory.mkdir()
        links_directory = staged / "links"
        links_directory.mkdir()
        text_directory = staged / "text"
        text_directory.mkdir()
        seen = 0
        connection = _read_connection(catalog_path)
        try:
            cursor = connection.execute(
                "SELECT id, url, total_minutes, time_status, servings, servings_status, "
                "source, language, ingredient_ids, title, raw_ingredients FROM recipes ORDER BY id")
            while batch := cursor.fetchmany(8192):
                for (recipe_id, url, minutes, time_status, servings, servings_status, source, language,
                     ingredients, title, raw_ingredients) in batch:
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
                    title_text, lines = recipe_title(title), ingredient_lines(raw_ingredients, source)
                    cleaned_url, status = public_source_url(url)
                    link, link_status = recipe_link(cleaned_url, title_text, link_rules)
                    arrays["link_status"][recipe_id] = link_status
                    url_statuses[status] += 1
                    link_statuses[LINK_STATUSES[link_status]] += 1
                    rows_by_source[source] += 1
                    shard_records.append(cleaned_url)
                    link_records.append(link)
                    _check_text(title_text, lines, f"recipe {recipe_id}")
                    titles.append(title_text)
                    line_lists.append(lines)
                    text_counts["recipe_titles"] += title_text is not None
                    text_counts["ingredient_line_records"] += lines is not None
                    text_counts["ingredient_lines"] += len(lines or ())
                    seen += 1
                    if len(shard_records) == rows_per_shard or seen == n_rows:
                        first = seen - len(shard_records)
                        name = f"{len(shards):04d}.json.gz"
                        data = _json_bytes({"first_id": first, "urls": shard_records})
                        record = _gzip_file(urls_directory / name, data)
                        record.update(file=f"urls/{name}", first_id=first, rows=len(shard_records))
                        shards.append(record)
                        record = _gzip_file(links_directory / name, _json_bytes({"first_id": first, "links": link_records}))
                        record.update(file=f"links/{name}", first_id=first, rows=len(link_records))
                        link_shards.append(record)
                        shard_records, link_records = [], []
                    if len(titles) == rows_per_text_shard or seen == n_rows:
                        first = seen - len(titles)
                        name = f"{len(text_shards):04d}.json.gz"
                        data = _json_bytes({"first_id": first, "titles": titles, "ingredient_lines": line_lists})
                        record = _gzip_file(text_directory / name, data)
                        record.update(file=f"text/{name}", first_id=first, rows=len(titles))
                        text_shards.append(record)
                        titles, line_lists = [], []
                if progress and (seen % (8192 * 32) == 0 or seen == n_rows):
                    progress({"phase": "source_metadata", "rows": seen, "total": n_rows})
        finally:
            connection.close()
        if seen != n_rows or shard_records or titles:
            raise ValueError("catalog scan did not cover every declared canonical row")
        text_manifest = _gzip_file(staged / TEXT_MANIFEST, _json_bytes({"shards": text_shards}))
        text_manifest["shards"] = len(text_shards)
        files = {}
        for name, array in arrays.items():
            filename, dtype = ARRAYS[name]
            if array.dtype != np.dtype(dtype):
                raise ValueError(f"{name}: unexpected serialized dtype")
            record = _gzip_file(staged / filename, memoryview(array).cast("B"))
            record.update(dtype=dtype, count=int(array.size))
            files[name] = record
        metadata_projection = {
            "schema_version": SCHEMA_VERSION, "format": FORMAT, "endianness": "little",
            "n_recipes": n_rows, "n_slots": n_slots,
            "vocabulary": metadata["vocabulary"],
            "ingredient_frequency": frequencies.tolist(),
            "statistics_scope": "full-canonical-corpus",
            "source_names": list(source_names), "language_names": list(language_names),
            "arrays": files, "url_shards": shards, "rows_per_url_shard": rows_per_shard,
            "link_shards": link_shards,
            "text_manifest": text_manifest, "rows_per_text_shard": rows_per_text_shard,
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
                "url_statuses": dict(url_statuses), "link_statuses": dict(link_statuses),
                "rows_by_source": dict(rows_by_source),
                **{name: text_counts[name] for name in TEXT_COVERAGE},
            },
            "bytes": {
                "initial_compressed_download": sum(file["bytes"] for file in files.values()),
                "initial_uncompressed_arrays": sum(file["raw_bytes"] for file in files.values()),
                "derived_uint32_offsets": (n_rows + 1) * 4,
                "url_shards_compressed": sum(shard["bytes"] for shard in shards),
                "largest_url_shard_compressed": max(shard["bytes"] for shard in shards),
                "link_shards_compressed": sum(shard["bytes"] for shard in link_shards),
                "largest_link_shard_compressed": max(shard["bytes"] for shard in link_shards),
                "text_manifest_compressed": text_manifest["bytes"],
                "text_shards_compressed": sum(shard["bytes"] for shard in text_shards),
                "largest_text_shard_compressed": max(shard["bytes"] for shard in text_shards),
            },
            "fields_included": list(INCLUDED_FIELDS),
            "fields_excluded": list(FORBIDDEN_FIELDS),
            "semantics": {
                "row_identity": "All canonical records in original ID order; duplicate sets are not removed.",
                "times": "Positive source-reported totals only; NaN means unknown; no component sums.",
                "servings": "Positive source-reported counts only; NaN means unknown; no quantity scaling.",
                "matching": "Canonical ingredient names, not quantities, every compound constituent or allergy safety.",
                "url_shards": "Only recipe IDs implied by order and original HTTP(S) URLs; scheme-less records use http://.",
                "link_shards": LINK_SEMANTICS,
                "text_shards": TEXT_SEMANTICS,
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
            "schema_version": 1, "status": "verified_local_recipe_card_export",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "index_sha256": file_sha256(staged / "ingredient-index.json"),
            "coverage": metadata_projection["coverage"], "bytes": metadata_projection["bytes"],
            "all_rows_compared_with_canonical_arrays": True,
            "source_metadata_provenance_checked_per_row": True,
            "cooking_instructions_exported": False, "uploaded": False,
        }
        (staged / "export-report.json").write_bytes(_json_bytes(report))
        _publish_directory(staged, output)
    return report


def load_ingredient_catalog(directory: Path) -> tuple[dict, dict[str, np.ndarray]]:
    metadata = json.loads((directory / "ingredient-index.json").read_text(encoding="utf-8"))
    if (metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("format") != FORMAT
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
    links = metadata.get("link_shards")
    if not isinstance(links, list) or len(links) != len(shards):
        raise ValueError("card link shards do not cover the declared population")
    for position, (shard, link_shard) in enumerate(zip(shards, links)):
        if (link_shard.get("file") != f"links/{position:04d}.json.gz"
                or link_shard.get("first_id") != shard["first_id"] or link_shard.get("rows") != shard["rows"]
                or type(link_shard.get("bytes")) is not int or not 0 < link_shard["bytes"] <= 16 * 1024 * 1024
                or type(link_shard.get("raw_bytes")) is not int
                or not 0 < link_shard["raw_bytes"] <= 256 * 1024 * 1024
                or not re.fullmatch(r"[a-f0-9]{64}", str(link_shard.get("sha256")))
                or not re.fullmatch(r"[a-f0-9]{64}", str(link_shard.get("raw_sha256")))):
            raise ValueError("card link shard path, identity or record boundaries are invalid")
    load_text_shards(directory, metadata)
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
            or np.any(arrays["link_status"] >= len(LINK_STATUSES))):
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
