"""Build the private, full-corpus SQLite catalog used by recipe search.

``build_catalog(output, paths=PATHS)`` never replaces an existing artifact. It
streams the immutable v2 text index, checks its corpus/reader provenance and
every row's identity, then publishes only after checking the persisted text,
ingredient sets, FTS tokens, row counts, slot counts and SQLite integrity.
``load_catalog_metadata(path)`` opens an existing, complete catalog read-only.

Schema v1
---------
``metadata(key, value)`` contains JSON values; ``recipes.id`` is the canonical
zero-based row, and ``recipe_fts.rowid`` is identical. ``ingredient_ids`` is a
sorted, unique, little-endian uint16 blob. Text, separators and serialized
quantities are copied verbatim, not reparsed or zipped.

``total_minutes`` is *only* a positive, finite source-provided total, identified
by ``time_status='source_total'``. Missing, invalid, inconsistent and ambiguous
totals are NULL. ``prep_minutes`` and ``cook_minutes`` retain valid source
components (including explicit zero); their sum is separately labelled
``derived_total_minutes`` and is never a strict-search total. Servings must be
positive numeric counts, not recipe yields such as "one loaf".

Additional provenance columns are ``time_provenance``, ``servings_provenance``
(source column names), ``metadata_status``, ``metadata_source_row`` (zero-based
ordinal in the source reader's yielded sequence), and ``metadata_match_count``.
All per-source coverage, unsupported-source reasons, input/code hashes and
verification results are recorded as metadata without recipe text.

Content coverage version 2 counts ``with_steps`` only when at least one
unit-separator-delimited fragment survives Python ``str.strip()``; ``with_title``
uses the same Unicode-whitespace semantics. ``*_storage_nonempty`` counters
separately retain raw string-presence counts. This is not an instruction-quality
assessment. Text is never stripped in storage. Existing catalogs without
``coverage_definitions`` retain their historical, raw-presence coverage metadata;
loading them neither rewrites nor silently reinterprets those counts.

Enrichment deliberately does not replay the multi-million-row raw corpus.
Supported metadata streams reuse the existing text readers and splitters.
Every candidate must match all indexed text fields exactly and reproduce the
canonical ingredient IDs. Identical text with conflicting metadata is unknown,
not "first match wins". Sources whose index has no recoverable identity are
explicitly unsupported. For isolated fixtures, ``metadata_records=()`` disables
raw-source access; callers may instead supply ``SourceMetadata`` records.
"""
from __future__ import annotations

import glob
import hashlib
import itertools
import json
import math
import numbers
import os
import re
import sqlite3
import struct
import time
import unicodedata
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np

from ..config import PATHS, Paths
from . import text as recipe_text
from .recipes import RECIPE_IDS, RecipeCorpus

SCHEMA_VERSION = 1
CATALOG_FILE = "recipe_search.sqlite"
TEXT_COLUMNS = (
    "source", "title", "url", "raw_ingredients", "steps",
    "ingredient_quantities", "quantity_status", "text_status",
)
REQUIRED_METADATA = {
    "schema_version", "corpus_sha256", "text_index_sha256", "n_recipes",
    "n_slots", "vocabulary", "ingredient_frequency", "partial", "coverage",
}
CONTENT_COUNTERS = (
    "with_title", "with_steps", "with_title_and_steps",
    "with_title_storage_nonempty", "with_steps_storage_nonempty",
    "with_title_and_steps_storage_nonempty",
    "title_whitespace_only", "steps_whitespace_or_separator_only",
)
CONTENT_COVERAGE_DEFINITIONS = {
    "version": 2,
    "with_title": "bool((title or '').strip())",
    "with_steps": "any(fragment.strip() for fragment in (steps or '').split('\\x1f'))",
    "with_title_and_steps": "with_title and with_steps",
    "storage_nonempty": "value IS NOT NULL AND value != ''; not SQLite length(value)>0",
    "whitespace_only": "storage_nonempty and not the corresponding Python-strip predicate",
    "unicode_database_version": unicodedata.unidata_version,
    "notes": (
        "Python strip includes Unicode whitespace and U+001F separators. NUL and "
        "zero-width space are not whitespace. These counters measure non-whitespace "
        "presence, not instruction quality, parsing success, or recipe completeness."),
}
_NUMBER = re.compile(r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_ISO_NUMBER = r"[0-9]+(?:[.,][0-9]+)?"
_ISO = re.compile(
    rf"P(?:(?P<w>{_ISO_NUMBER})W|(?:(?P<d>{_ISO_NUMBER})D)?"
    rf"(?:T(?:(?P<h>{_ISO_NUMBER})H)?(?:(?P<m>{_ISO_NUMBER})M)?"
    rf"(?:(?P<s>{_ISO_NUMBER})S)?)?)"
)
_UNITS = {
    "en": {
        "days": 1440, "day": 1440, "hours": 60, "hour": 60, "hrs": 60, "hr": 60,
        "minutes": 1, "minute": 1, "mins": 1, "min": 1,
        "seconds": 1 / 60, "second": 1 / 60, "secs": 1 / 60, "sec": 1 / 60,
    },
    "el": {
        "ημέρες": 1440, "ημέρα": 1440, "ώρες": 60, "ώρα": 60,
        "λεπτά": 1, "λεπτό": 1, "δευτερόλεπτα": 1 / 60,
    },
    "ja": {"日": 1440, "時間": 60, "分": 1, "秒": 1 / 60},
    "ru": {
        "дней": 1440, "дня": 1440, "день": 1440,
        "часов": 60, "часа": 60, "час": 60,
        "минут": 1, "минуты": 1, "минута": 1,
        "секунд": 1 / 60, "секунды": 1 / 60, "секунда": 1 / 60,
    },
    "zh": {"天": 1440, "小時": 60, "小时": 60, "分鐘": 1, "分钟": 1, "秒": 1 / 60},
}
_UNIT_PATTERNS = {
    language: re.compile(
        r"([0-9]+(?:\.[0-9]+)?)\s*("
        + "|".join(re.escape(unit) for unit in sorted(units, key=len, reverse=True))
        + r")"
    )
    for language, units in _UNITS.items()
}


@dataclass(frozen=True)
class ParsedNumber:
    value: float | None = None
    status: str = "missing"


def _missing(value) -> bool:
    return value is None or (
        isinstance(value, str) and value.strip().lower() in {"", "na", "n/a", "null", "none"}
    )


def _numeric(value) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, str):
        if not _NUMBER.fullmatch(value.strip()):
            return None
    elif not isinstance(value, numbers.Real):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def parse_duration(value, *, encoding: str, allow_zero: bool = False) -> ParsedNumber:
    """Parse an explicitly declared duration field, never arbitrary prose.

    ``encoding`` is ``iso8601``, ``minutes`` (a documented numeric-minute
    column), or a unit-labelled language in ``en/el/ja/ru/zh``. Calendar months
    and years, ranges, signed values, nonfinite values, and trailing prose are
    invalid. Numeric NaN is invalid; source readers turn actual nullable cells
    into None before parsing. Zero is valid only for explicit components.
    """
    if encoding not in {"iso8601", "minutes", *_UNITS}:
        raise ValueError(f"unsupported duration encoding: {encoding}")
    if _missing(value):
        return ParsedNumber()
    number = None
    if encoding == "minutes":
        number = _numeric(value)
    elif isinstance(value, str):
        value = value.strip()
        if encoding == "iso8601":
            match = _ISO.fullmatch(value)
            if match:
                parts = [(name, component) for name, component in match.groupdict().items()
                         if component is not None]
                has_time = any(name in {"h", "m", "s"} for name, _ in parts)
                fractional = [i for i, (_, component) in enumerate(parts)
                              if "." in component or "," in component]
                if (parts and ("T" not in value or has_time)
                        and (not fractional or fractional == [len(parts) - 1])):
                    factors = {"w": 10080, "d": 1440, "h": 60, "m": 1, "s": 1 / 60}
                    number = sum(float(component.replace(",", ".")) * factors[name]
                                 for name, component in parts)
        else:
            value = value.lower()
            position, previous, total = 0, math.inf, 0.0
            valid = False
            while position < len(value):
                match = _UNIT_PATTERNS[encoding].match(value, position)
                if not match:
                    valid = False
                    break
                factor = _UNITS[encoding][match[2]]
                if factor >= previous:
                    valid = False
                    break
                total += float(match[1]) * factor
                previous = factor
                position = match.end()
                while position < len(value) and value[position].isspace():
                    position += 1
                valid = True
            if valid:
                number = total
    if number is None or not math.isfinite(number) or number < 0 or (
            number == 0 and not allow_zero):
        return ParsedNumber(status="invalid")
    return ParsedNumber(number, "provided")


def parse_servings(value, *, encoding: str = "number") -> ParsedNumber:
    """Accept exact positive counts; ranges, yields and prose stay unknown."""
    if encoding not in {"number", "en", "el"}:
        raise ValueError(f"unsupported servings encoding: {encoding}")
    if _missing(value):
        return ParsedNumber()
    number = _numeric(value)
    if number is None and isinstance(value, str) and encoding != "number":
        suffix = {
            "en": r"(?:servings?|people|persons?)",
            "el": r"(?:μερίδες|μερίδα|άτομα)",
        }[encoding]
        match = re.fullmatch(
            rf"([0-9]+(?:\.[0-9]+)?)\s+{suffix}", value.strip(), flags=re.IGNORECASE)
        if match:
            number = _numeric(match[1])
    if number is None or number <= 0:
        return ParsedNumber(status="invalid")
    return ParsedNumber(number, "provided")


@dataclass(frozen=True)
class MetadataSpec:
    total: tuple[str, str] | None = None
    prep: tuple[str, str] | None = None
    cook: tuple[str, str] | None = None
    servings: tuple[str, str] | None = None
    note: str = ""


SOURCE_SPECS = {
    "foodcom-522k": MetadataSpec(
        ("TotalTime", "iso8601"), ("PrepTime", "iso8601"), ("CookTime", "iso8601"),
        ("RecipeServings", "number"), "RecipeYield is not a serving count."),
    "foodcom-raw-231k": MetadataSpec(
        total=("minutes", "minutes"),
        note="Source numeric total minutes; no servings column."),
    "povarenok-detail": MetadataSpec(
        cook=("cooking_time", "ru"), servings=("portions_count", "number"),
        note="Cooking time is a component, not a documented total."),
    "allrecipes-33k": MetadataSpec(
        ("total_time", "en"), ("prep_time", "en"), ("cook_time", "en"),
        ("servings", "number"), "Only full, unit-labelled duration fields are parsed."),
    "indian-7k": MetadataSpec(
        prep=("prep_time (in mins)", "minutes"), cook=("cook_time (in mins)", "minutes"),
        note="No total field. Prep + cook is retained only as a derived sum."),
    "greek-5k": MetadataSpec(
        total=("Total Time", "el"), prep=("Preparation Time", "el"),
        servings=("Number of Servings", "el")),
    "japanese-3k": MetadataSpec(total=("合計時間", "ja")),
    "filipino-2k": MetadataSpec(
        ("total_time", "en"), ("prep_time", "en"), ("cook_time", "en"), ("servings", "en")),
    "halal-2k": MetadataSpec(
        ("total_time", "en"), ("prep_time", "en"), ("cook_time", "en"), ("servings", "en"),
        "Bare numeric durations have no declared unit and remain invalid."),
    "taiwan-1.8k": MetadataSpec(
        cook=("cooking_time", "zh"), servings=("servings", "number"),
        note="Cooking time is not promoted to a total."),
}
UNSUPPORTED_SOURCES = {
    "01-recipenlg": "CSV schema has no structured timing or serving fields.",
    "02-xiachufang": (
        "Sampled JSONL keys have no timing/servings; sparse optional fields were not "
        "exhaustively scanned. No duration is inferred from descriptions or instructions."),
    "03-povarenok": "CSV schema contains only URL, name and ingredients.",
    "04-spanish": "The immutable text index has no mirrored source identity.",
    "05-vietnamese": "Chat/prose source; no supported structured metadata reader.",
    "06-turkish": "Structured components exist, but the text index has no mirrored identity.",
    "07-indian": "Structured totals/servings exist in part, but the text index has no identity.",
    "08-indonesian": "The immutable text index has no mirrored source identity.",
    "09-chefkoch": "No supported duration fields; calendar dates are not recipe durations.",
    "turkish-102k": "Parquet schema has no structured timing or serving fields.",
    "thefoodprocessor-74k": "Recipe prose only; times are not inferred from text.",
    "bhuvii-17k": "No supported structured timing or serving fields.",
    "kaggle-food-13k": "CSV schema has no structured timing or serving fields.",
    "hebrew-9.7k": "JSON-LD may have metadata, but the text index has no mirrored identity.",
    "persian-6k": (
        "No total field. Bare preparation/cooking numbers have no declared unit; "
        "suitable_for is not treated as a documented serving count."),
    "moroccan-4.6k": "Prompt/prose source without a mirrored text identity.",
    "thai-1k": "CSV schema has no structured timing or serving fields.",
    "thai-1k-seasoning": "CSV schema has no structured timing or serving fields.",
    "romanian-881": "No supported structured metadata in the unnamed CSV fields.",
}


@dataclass(frozen=True)
class SourceMetadata:
    """One reader-yielded source record; ``text`` uses the v2 column names."""

    source: str
    text: Mapping[str, str | None]
    ingredient_ids: tuple[int, ...]
    source_row: int = 0
    total: ParsedNumber = field(default_factory=ParsedNumber)
    prep: ParsedNumber = field(default_factory=ParsedNumber)
    cook: ParsedNumber = field(default_factory=ParsedNumber)
    servings: ParsedNumber = field(default_factory=ParsedNumber)
    total_field: str = ""
    prep_field: str = ""
    cook_field: str = ""
    servings_field: str = ""


def _hash_fields(digest, values) -> None:
    for value in values:
        if value is None:
            digest.update(b"\xff" * 8)
        else:
            encoded = value.encode("utf-8")
            digest.update(struct.pack("<Q", len(encoded)))
            digest.update(encoded)


def _text_key(values) -> bytes:
    digest = hashlib.sha256(b"recipe-catalog-text-identity-v1\0")
    _hash_fields(digest, values)
    return digest.digest()


def _hash_file(path: Path) -> dict:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size, after.st_mtime_ns, after.st_ino):
        raise RuntimeError(f"input changed while hashing: {path}")
    return {"path": str(path), "bytes": after.st_size, "sha256": digest.hexdigest()}


def _report_file(info: dict) -> dict:
    """Keep reproducibility fingerprints without exporting filesystem paths."""
    return {
        "artifact_name": Path(info["path"]).name,
        "bytes": info["bytes"], "sha256": info["sha256"],
    }


def _source_frames(raw, kind, rel, columns):
    import pandas as pd
    import pyarrow.parquet as pq

    path = raw.EXP / rel
    if kind == "csv":
        # usecols changes pandas' treatment of malformed/overlong CSV rows.
        # Keep the reader's exact options so its skip sequence cannot drift.
        yield from pd.read_csv(
            path, chunksize=100_000, engine="c", on_bad_lines="skip", dtype=str)
    elif kind == "parquet":
        for filename in sorted(glob.glob(str(path))):
            parquet = pq.ParquetFile(filename)
            for batch in parquet.iter_batches(
                    columns=columns, batch_size=8192, use_threads=False):
                yield batch.to_pandas()
    else:
        raise ValueError(f"unsupported frame source: {kind}")


def _source_value_rows(raw, key, kind, rel, column, splitter, spec):
    import pandas as pd
    import pyarrow.parquet as pq

    if kind == "jsonl":
        def rows():
            with (raw.EXP / rel).open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        yield json.loads(line)
                    except ValueError:
                        continue
    else:
        requested = {column}
        requested.update(pair[0] for pair in (
            spec.total, spec.prep, spec.cook, spec.servings) if pair)
        if kind == "parquet":
            files = sorted(glob.glob(str(raw.EXP / rel)))
            if not files:
                raise FileNotFoundError(raw.EXP / rel)
            available = set(pq.ParquetFile(files[0]).schema_arrow.names)
            identity = recipe_text._resolve_cols(available, key)
            requested.update(identity[name] for name in ("title", "url") if identity.get(name))

        def rows():
            for frame in _source_frames(raw, kind, rel, sorted(requested)):
                for values in frame.itertuples(index=False, name=None):
                    yield dict(zip(frame.columns, values, strict=True))

    for row in rows():
        if column not in row and kind != "jsonl":
            raise RuntimeError(f"ingredient column disappeared for {key}: {column}")
        value = row.get(column)
        if kind != "jsonl" and (value is None or (
                isinstance(value, float) and math.isnan(value))):
            continue
        items = splitter(value)
        if not items:
            continue
        identity = recipe_text._resolve_cols(set(row), key)
        expected = {
            "raw": (recipe_text._foodcom_fields(value, None, None)["raw"]
                    if key == "foodcom-522k" else recipe_text._join(value)),
        }
        for name in ("title", "url"):
            cell = row.get(identity.get(name))
            expected[name] = (str(cell or "") if kind == "jsonl"
                              else recipe_text._join(cell))
        parsed = {}
        for name, pair in (("total", spec.total), ("prep", spec.prep),
                           ("cook", spec.cook), ("servings", spec.servings)):
            if pair is None:
                parsed[name] = ParsedNumber()
                continue
            if pair[0] not in row and kind != "jsonl":
                raise RuntimeError(f"metadata column disappeared for {key}: {pair[0]}")
            cell = row.get(pair[0])
            if not isinstance(cell, (list, dict, tuple, np.ndarray)) and pd.isna(cell):
                cell = None
            parsed[name] = (parse_servings(cell, encoding=pair[1]) if name == "servings"
                            else parse_duration(
                                cell, encoding=pair[1], allow_zero=name in {"prep", "cook"}))
        yield items, expected, parsed


def _default_metadata(corpus, paths, inputs, code, audit):
    raw, nm = recipe_text._load_llmmm()
    if Path(raw.__file__).resolve().parent != paths.prior_tools.resolve():
        raise RuntimeError("configured paths and the original corpus reader disagree")
    if paths.corpus is not None and Path(raw.BASE).resolve() != paths.corpus.resolve():
        raise RuntimeError("configured paths and the original raw corpus root disagree")
    normalizer = recipe_text._corpus_normalizer(nm)
    if list(normalizer.itos) != corpus.itos:
        raise RuntimeError("normalizer and corpus vocabulary are misaligned")
    from . import normalizer as normalizer_module

    for path in (Path(raw.__file__), Path(nm.__file__), Path(normalizer_module.__file__),
                 paths.prior_tools / "lexicons.py", paths.prior_tools / "multilingual.py",
                 paths.prior_tools / "build_recipe_cooc.py"):
        code[str(path)] = _hash_file(path)
    if hasattr(nm, "brc"):
        vocabulary_path = nm.brc.RAW / "epicure-cooc" / "vocab.json"
        inputs[str(vocabulary_path)] = _hash_file(vocabulary_path)
    for path in sorted(normalizer_module.ALIAS_DIR.glob("*.json")):
        code[str(path)] = _hash_file(path)
    registry = {key: (lang, kind, rel, col, splitter)
                for key, lang, kind, rel, col, splitter in raw.EXPANSION}
    sentinel = object()
    for key in audit:
        if key not in SOURCE_SPECS:
            continue
        spec = SOURCE_SPECS[key]
        if key not in registry:
            raise RuntimeError(f"source reader disappeared: {key}")
        lang, kind, rel, col, splitter = registry[key]
        if not recipe_text._mirrorable(kind, col):
            raise RuntimeError(f"source no longer has a mirrored text identity: {key}")
        filenames = sorted(glob.glob(str(raw.EXP / rel)))
        if not filenames:
            raise FileNotFoundError(raw.EXP / rel)
        audit[key]["inputs"] = []
        for filename in filenames:
            info = _hash_file(Path(filename))
            inputs[filename] = info
            audit[key]["inputs"].append(info)
        meta_stream = recipe_text._meta_stream(raw, key)
        if meta_stream is None:
            raise RuntimeError(f"metadata stream disappeared: {key}")
        values = _source_value_rows(raw, key, kind, rel, col, splitter, spec)
        for ordinal, (meta, value) in enumerate(
                itertools.zip_longest(meta_stream, values, fillvalue=sentinel)):
            if meta is sentinel or value is sentinel:
                raise RuntimeError(f"metadata reader length misalignment for {key}")
            items, expected, parsed = value
            if any(meta.get(name, "") != expected[name] for name in expected):
                raise RuntimeError(f"metadata reader identity misalignment for {key} row {ordinal}")
            ids = tuple(sorted(normalizer.normalize(lang, items)))
            audit[key]["reader_rows"] += 1
            if not ids:
                audit[key]["empty_normalized"] += 1
                continue
            fields = {
                "source": key, "title": meta["title"], "url": meta["url"],
                "raw_ingredients": meta["raw"], "steps": meta["steps"],
                "ingredient_quantities": meta.get("ingredient_quantities", "[]"),
                "quantity_status": meta.get("quantity_status", ""),
                "text_status": meta.get("text_status", ""),
            }
            yield SourceMetadata(
                source=key, text=fields, ingredient_ids=ids, source_row=ordinal,
                **parsed, **{f"{name}_field": pair[0] if pair else ""
                             for name, pair in (("total", spec.total), ("prep", spec.prep),
                                                ("cook", spec.cook), ("servings", spec.servings))})


_SCHEMA = """
CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE recipes(
    id INTEGER PRIMARY KEY, source TEXT NOT NULL, language TEXT NOT NULL,
    title TEXT, url TEXT, ingredient_ids BLOB NOT NULL,
    raw_ingredients TEXT, steps TEXT, ingredient_quantities TEXT,
    quantity_status TEXT, text_status TEXT, total_minutes REAL, servings REAL,
    time_status TEXT NOT NULL, servings_status TEXT NOT NULL,
    prep_minutes REAL, cook_minutes REAL, derived_total_minutes REAL,
    time_provenance TEXT, servings_provenance TEXT, metadata_status TEXT NOT NULL,
    metadata_source_row INTEGER, metadata_match_count INTEGER NOT NULL,
    CHECK(total_minutes IS NULL OR (total_minutes > 0 AND time_status='source_total')),
    CHECK(servings IS NULL OR (servings > 0 AND servings_status='source_servings'))
);
CREATE VIRTUAL TABLE recipe_fts USING fts5(ingredient_tokens, tokenize='ascii');
CREATE TABLE _enrichment(
    fingerprint BLOB PRIMARY KEY, source TEXT NOT NULL, ingredient_ids BLOB NOT NULL,
    source_row INTEGER NOT NULL,
    total REAL, total_status TEXT NOT NULL, prep REAL, prep_status TEXT NOT NULL,
    cook REAL, cook_status TEXT NOT NULL, servings REAL, servings_status TEXT NOT NULL,
    total_field TEXT, prep_field TEXT, cook_field TEXT, servings_field TEXT,
    copies INTEGER NOT NULL DEFAULT 1, total_conflict INTEGER NOT NULL DEFAULT 0,
    component_conflict INTEGER NOT NULL DEFAULT 0, servings_conflict INTEGER NOT NULL DEFAULT 0,
    ingredient_conflict INTEGER NOT NULL DEFAULT 0, matches INTEGER NOT NULL DEFAULT 0
) WITHOUT ROWID;
"""
_INSERT_ENRICHMENT = """
INSERT INTO _enrichment(
    fingerprint,source,ingredient_ids,source_row,total,total_status,prep,prep_status,
    cook,cook_status,servings,servings_status,total_field,prep_field,cook_field,servings_field
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
ON CONFLICT(fingerprint) DO UPDATE SET
    copies=copies+1,
    total_conflict=total_conflict OR total IS NOT excluded.total
        OR total_status!=excluded.total_status OR total_field!=excluded.total_field,
    component_conflict=component_conflict OR prep IS NOT excluded.prep
        OR cook IS NOT excluded.cook OR prep_status!=excluded.prep_status
        OR cook_status!=excluded.cook_status OR prep_field!=excluded.prep_field
        OR cook_field!=excluded.cook_field,
    servings_conflict=servings_conflict OR servings IS NOT excluded.servings
        OR servings_status!=excluded.servings_status OR servings_field!=excluded.servings_field,
    ingredient_conflict=ingredient_conflict OR ingredient_ids!=excluded.ingredient_ids
"""
_ENRICHMENT_SELECT = """
SELECT ingredient_ids,source_row,total,total_status,prep,prep_status,cook,cook_status,
    servings,servings_status,total_field,prep_field,cook_field,servings_field,
    copies,total_conflict,component_conflict,servings_conflict
FROM _enrichment WHERE fingerprint=?
"""


def _pack_ids(ids, n_vocab):
    array = np.asarray(ids)
    if array.ndim != 1 or not len(array) or array.dtype.kind not in "iu":
        raise RuntimeError("metadata ingredient IDs must be a nonempty integer vector")
    if int(array.min()) < 0 or int(array.max()) >= n_vocab or np.any(array[1:] <= array[:-1]):
        raise RuntimeError("metadata ingredient IDs are invalid, unsorted or duplicated")
    return array.astype("<u2", copy=False).tobytes()


def _stage_metadata(connection, records, corpus, audit, batch_size, progress):
    batch = []
    count = 0
    for record in records:
        if record.source not in audit or record.text.get("source") != record.source:
            raise RuntimeError("metadata source identity misalignment")
        values = [record.text[name] for name in TEXT_COLUMNS]
        if any(value is not None and not isinstance(value, str) for value in values):
            raise RuntimeError("metadata text fields must be strings or null")
        if record.source_row < 0:
            raise RuntimeError("metadata source row must be nonnegative")
        for name in ("total", "prep", "cook", "servings"):
            parsed = getattr(record, name)
            if parsed.status not in {"provided", "missing", "invalid"}:
                raise ValueError(f"invalid parsed metadata status: {parsed.status}")
            if parsed.status == "provided":
                if (parsed.value is None or not math.isfinite(parsed.value)
                        or parsed.value < 0 or (parsed.value == 0 and name in {"total", "servings"})
                        or not getattr(record, f"{name}_field")):
                    raise ValueError(f"invalid provided metadata: {name}")
            elif parsed.value is not None:
                raise ValueError("unknown metadata must not carry a numeric value")
        blob = _pack_ids(record.ingredient_ids, corpus.n_vocab)
        batch.append((
            _text_key(values), record.source, blob, record.source_row,
            record.total.value, record.total.status, record.prep.value, record.prep.status,
            record.cook.value, record.cook.status, record.servings.value, record.servings.status,
            record.total_field, record.prep_field, record.cook_field, record.servings_field))
        audit[record.source]["metadata_records"] += 1
        audit[record.source]["supported"] = True
        count += 1
        if len(batch) >= batch_size:
            connection.executemany(_INSERT_ENRICHMENT, batch)
            connection.commit()
            batch.clear()
            if count % (batch_size * 16) == 0:
                progress({"phase": "source_metadata", "records": count, "source": record.source})
    if batch:
        connection.executemany(_INSERT_ENRICHMENT, batch)
        connection.commit()
    if connection.execute("SELECT 1 FROM _enrichment WHERE ingredient_conflict LIMIT 1").fetchone():
        raise RuntimeError("metadata ingredient identity misalignment for identical text")
    progress({"phase": "source_metadata_complete", "records": count})


def _resolved_metadata(record, *, supported):
    if record is None:
        status = "unmatched" if supported else "unsupported"
        return (None, None, status, status, None, None, None, "", "", status, None, 0)
    (_blob, ordinal, total, total_status, prep, _prep_status, cook, _cook_status,
     servings, servings_status, total_field, prep_field, cook_field, servings_field,
     copies, total_conflict, component_conflict, servings_conflict) = record
    if component_conflict:
        prep = cook = None
    derived = prep + cook if prep is not None and cook is not None else None
    if derived is not None and (not math.isfinite(derived) or derived <= 0):
        derived = None
    if total_conflict:
        total, time_status = None, "ambiguous_total"
    elif total_status == "invalid":
        total, time_status = None, "invalid_total"
    elif total is not None:
        if any(component is not None and component > total for component in (prep, cook)):
            total, time_status = None, "inconsistent_total"
        else:
            time_status = "source_total"
    elif derived is not None:
        time_status = "derived_only"
    elif prep is not None or cook is not None:
        time_status = "component_only"
    else:
        time_status = "missing_total" if total_field else "no_total_field"
    if servings_conflict:
        servings, servings_status = None, "ambiguous"
    elif servings_status == "provided":
        servings_status = "source_servings"
    elif not servings_field:
        servings_status = "unsupported"
    metadata_status = "ambiguous" if (
        total_conflict or component_conflict or servings_conflict) else "matched"
    provenance = total_field or " + ".join(value for value in (prep_field, cook_field) if value)
    return (total, servings, time_status, servings_status, prep, cook, derived,
            provenance, servings_field, metadata_status, ordinal, copies)


def has_usable_steps(value: str | None) -> bool:
    """Whether any ``value.split('\\x1f')`` fragment survives Python ``strip``.

    U+001F is itself Python whitespace, so stripping the entire field is exactly
    equivalent and avoids allocating a fragment list for every corpus row.
    """
    return bool(value and value.strip())


def _count_content_coverage(coverage, title, steps):
    raw_title, raw_steps = bool(title), bool(steps)
    usable_title, usable_steps = bool(title and title.strip()), has_usable_steps(steps)
    coverage["with_title"] += usable_title
    coverage["with_steps"] += usable_steps
    coverage["with_title_and_steps"] += usable_title and usable_steps
    coverage["with_title_storage_nonempty"] += raw_title
    coverage["with_steps_storage_nonempty"] += raw_steps
    coverage["with_title_and_steps_storage_nonempty"] += raw_title and raw_steps
    coverage["title_whitespace_only"] += raw_title and not usable_title
    coverage["steps_whitespace_or_separator_only"] += raw_steps and not usable_steps


def _empty_coverage():
    return {
        "n_recipes": 0, "n_slots": 0, **dict.fromkeys(CONTENT_COUNTERS, 0),
        "total_minutes_known": 0, "total_minutes_unknown": 0,
        "total_minutes_missing": 0, "total_minutes_invalid": 0, "total_minutes_ambiguous": 0,
        "servings_known": 0, "servings_unknown": 0, "servings_missing": 0,
        "servings_invalid": 0, "servings_ambiguous": 0, "derived_total_minutes_known": 0,
        "time_status": Counter(), "servings_status": Counter(), "metadata_status": Counter(),
        "quantity_status": Counter(), "text_status": Counter(),
    }


def _count_coverage(coverage, text, n_ids, metadata):
    total, servings, time_status, servings_status, _prep, _cook, derived = metadata[:7]
    coverage["n_recipes"] += 1
    coverage["n_slots"] += n_ids
    _count_content_coverage(coverage, text[1], text[4])
    coverage["total_minutes_known"] += total is not None
    coverage["total_minutes_unknown"] += total is None
    coverage["total_minutes_invalid"] += time_status in {"invalid_total", "inconsistent_total"}
    coverage["total_minutes_ambiguous"] += time_status == "ambiguous_total"
    coverage["total_minutes_missing"] += total is None and time_status not in {
        "invalid_total", "inconsistent_total", "ambiguous_total"}
    coverage["servings_known"] += servings is not None
    coverage["servings_unknown"] += servings is None
    coverage["servings_invalid"] += servings_status == "invalid"
    coverage["servings_ambiguous"] += servings_status == "ambiguous"
    coverage["servings_missing"] += servings is None and servings_status not in {"invalid", "ambiguous"}
    coverage["derived_total_minutes_known"] += derived is not None
    for name, value in (("time_status", time_status), ("servings_status", servings_status),
                        ("metadata_status", metadata[9]), ("quantity_status", text[6]),
                        ("text_status", text[7])):
        coverage[name][value or "not_supplied"] += 1


def _load_corpus(path, generation):
    with np.load(path, allow_pickle=True) as stored:
        corpus = RecipeCorpus(
            flat=stored["flat"], offsets=stored["offsets"],
            lang=stored["lang"], source=stored["source"],
            itos=[str(value) for value in stored["itos"]])
    if (corpus.flat.ndim != 1 or corpus.flat.dtype.kind not in "iu"
            or corpus.offsets.ndim != 1 or corpus.offsets.dtype.kind not in "iu"
            or corpus.n_recipes <= 0 or not 0 < corpus.n_vocab <= 65536
            or len(set(corpus.itos)) != corpus.n_vocab):
        raise RuntimeError("invalid corpus arrays or vocabulary")
    if (corpus.offsets[0] != 0 or corpus.offsets[-1] != len(corpus.flat)
            or np.any(corpus.offsets[1:] <= corpus.offsets[:-1])
            or corpus.lang.ndim != 1 or corpus.source.ndim != 1
            or len(corpus.lang) != corpus.n_recipes or len(corpus.source) != corpus.n_recipes):
        raise RuntimeError("corpus offsets or row metadata are misaligned")
    if not len(corpus.flat) or int(corpus.flat.min()) < 0 or int(corpus.flat.max()) >= corpus.n_vocab:
        raise RuntimeError("corpus ingredient IDs are outside the vocabulary")
    bad_order = corpus.flat[1:] <= corpus.flat[:-1]
    bad_order[corpus.offsets[1:-1].astype(np.int64) - 1] = False
    if np.any(bad_order):
        raise RuntimeError("corpus recipes must contain sorted unique ingredient IDs")
    for key, actual in (("recipes", corpus.n_recipes), ("slots", len(corpus.flat)),
                        ("vocab", corpus.n_vocab)):
        if key in generation and generation[key] != actual:
            raise RuntimeError(f"generation {key} does not match the corpus")
    return corpus


def _validate_text(parquet, corpus, corpus_sha256, reader_sha256):
    import pyarrow as pa

    schema = parquet.schema_arrow
    metadata = schema.metadata or {}
    if metadata.get(b"corpus_sha256") != corpus_sha256.encode():
        raise RuntimeError("text index corpus hash misalignment")
    if metadata.get(b"text_schema_version") != b"2" or metadata.get(b"partial") != b"false":
        raise RuntimeError("a complete immutable v2 text index is required")
    if metadata.get(b"reader_sha256") != reader_sha256.encode():
        raise RuntimeError("text index reader hash misalignment")
    if parquet.metadata.num_rows != corpus.n_recipes:
        raise RuntimeError("text index row count does not match the corpus")
    if not {"idx", *TEXT_COLUMNS}.issubset(schema.names):
        raise RuntimeError("text index lacks required v2 columns")
    if not pa.types.is_integer(schema.field("idx").type) or any(
            not (pa.types.is_string(schema.field(name).type)
                 or pa.types.is_large_string(schema.field(name).type))
            for name in TEXT_COLUMNS):
        raise RuntimeError("unexpected text index column types")


def _set_metadata(connection, values):
    connection.executemany(
        "INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)",
        [(key, json.dumps(value, ensure_ascii=False, allow_nan=False))
         for key, value in values.items()])
    connection.commit()


def _verify_catalog(
        connection, corpus, text_digest, tokens, batch_size, progress, *, expected_coverage):
    connection.execute("INSERT INTO recipe_fts(recipe_fts) VALUES ('integrity-check')")
    connection.commit()
    integrity = connection.execute("PRAGMA integrity_check").fetchall()
    if integrity != [("ok",)]:
        raise RuntimeError(f"SQLite integrity check failed: {integrity[:3]}")
    expected = (corpus.n_recipes, 0, corpus.n_recipes - 1)
    for table, column in (("recipes", "id"), ("recipe_fts", "rowid")):
        actual = connection.execute(
            f"SELECT COUNT(*), MIN({column}), MAX({column}) FROM {table}").fetchone()
        if actual != expected:
            raise RuntimeError(f"{table} row coverage is not exact: {actual}")
    slots = connection.execute("SELECT SUM(length(ingredient_ids) / 2) FROM recipes").fetchone()[0]
    if slots != len(corpus.flat):
        raise RuntimeError("persisted ingredient slot count does not match the corpus")
    cursor = connection.execute("""
        SELECT r.id,r.language,r.ingredient_ids,r.source,r.title,r.url,
            r.raw_ingredients,r.steps,r.ingredient_quantities,r.quantity_status,r.text_status,
            f.ingredient_tokens
        FROM recipes r LEFT JOIN recipe_fts f ON f.rowid=r.id ORDER BY r.id
    """)
    digest = hashlib.sha256()
    content_counts = {
        source: dict.fromkeys(CONTENT_COUNTERS, 0) for source in expected_coverage
    }
    index = 0
    while rows := cursor.fetchmany(batch_size):
        for row in rows:
            row_id, language, blob = row[:3]
            ids = corpus.recipe(index)
            if (row_id != index or language != str(corpus.lang[index])
                    or row[3] != str(corpus.source[index])
                    or blob != ids.astype("<u2", copy=False).tobytes()
                    or row[-1] != " ".join(tokens[int(value)] for value in ids)):
                raise RuntimeError(f"persisted recipe/FTS identity misalignment at row {index}")
            digest.update(struct.pack("<Q", index))
            _hash_fields(digest, row[3:-1])
            _count_content_coverage(content_counts[row[3]], row[4], row[7])
            index += 1
        if index % (batch_size * 32) == 0:
            progress({"phase": "verify", "rows": index})
    if index != corpus.n_recipes or digest.hexdigest() != text_digest:
        raise RuntimeError("persisted recipe text does not exactly match the text index")
    for source, counts in content_counts.items():
        if any(counts[key] != expected_coverage[source][key] for key in CONTENT_COUNTERS):
            raise RuntimeError(f"persisted text content coverage disagrees with reported counts ({source})")
    return {
        "sqlite_integrity": "ok", "fts_integrity": "ok", "recipe_rows": index,
        "fts_rows": index, "ingredient_slots": slots, "fts_tokens": slots,
        "all_row_identities_checked": True, "all_ingredient_blobs_checked": True,
        "all_fts_tokens_checked": True, "all_text_fields_checked": True,
        "content_coverage_checked": True, "content_coverage_version": 2,
        "text_content_sha256": text_digest,
    }


def build_catalog(
        output: Path, *, paths: Paths = PATHS, corpus_path: Path | None = None,
        text_index_path: Path | None = None,
        metadata_records: Iterable[SourceMetadata] | None = None,
        expected_text_sha256: str | None = None,
        require_metadata_match: bool | None = None, batch_size: int = 8192,
        progress: Callable[[dict], None] | None = None) -> dict:
    """Build and atomically publish a complete local catalog; return safe aggregates.

    Explicit fixture paths still require a GENERATION.json pin and an aligned,
    nonpartial v2 parquet. ``metadata_records=None`` enables the supported raw
    readers; an explicit iterable provides an isolated, testable metadata source.
    In automatic mode every supported canonical row must have an exact source
    match; a changed source is an error, not missing timing. Explicit fixture
    records can be sparse unless ``require_metadata_match=True`` is requested.
    No partial/limited catalog is published under any output name.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"{output}: existing catalogs are not overwritten")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if require_metadata_match is None:
        require_metadata_match = metadata_records is None
    progress = progress or (lambda event: None)
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    corpus_path = Path(corpus_path or paths.recipes / RECIPE_IDS)
    text_index_path = Path(text_index_path or paths.recipes / recipe_text.TEXT_FILE)
    generation_info = _hash_file(paths.generation_file)
    generation = json.loads(paths.generation_file.read_text(encoding="utf-8"))
    corpus_info = _hash_file(corpus_path)
    if generation.get("sha256") != corpus_info["sha256"]:
        raise RuntimeError("generation corpus hash misalignment")
    corpus = _load_corpus(corpus_path, generation)
    text_info = _hash_file(text_index_path)
    if expected_text_sha256 is not None and text_info["sha256"] != expected_text_sha256:
        raise RuntimeError("text index hash does not match the explicit pin")
    text_reader_info = _hash_file(Path(recipe_text.__file__))
    parquet = pq.ParquetFile(text_index_path)
    _validate_text(parquet, corpus, corpus_info["sha256"], text_reader_info["sha256"])
    sources = sorted(str(value) for value in np.unique(corpus.source))
    source_audit = {
        source: {
            "supported": metadata_records is None and source in SOURCE_SPECS,
            "reason": (SOURCE_SPECS[source].note or "Explicit structured metadata fields.")
            if metadata_records is None and source in SOURCE_SPECS else (
                UNSUPPORTED_SOURCES.get(source, "No supported metadata reader.")
                if metadata_records is None else "Caller-supplied metadata records."),
            "fields": {
                name: list(pair) if pair else None for name, pair in (
                    ("total", SOURCE_SPECS[source].total), ("prep", SOURCE_SPECS[source].prep),
                    ("cook", SOURCE_SPECS[source].cook), ("servings", SOURCE_SPECS[source].servings))
            } if metadata_records is None and source in SOURCE_SPECS else {},
            "inputs": [], "reader_rows": 0, "empty_normalized": 0, "metadata_records": 0,
        } for source in sources
    }
    inputs = {info["path"]: info for info in (generation_info, corpus_info, text_info)}
    code = {
        str(Path(__file__)): _hash_file(Path(__file__)),
        str(Path(recipe_text.__file__)): text_reader_info,
    }
    from . import recipes as recipes_module
    code[str(Path(recipes_module.__file__))] = _hash_file(Path(recipes_module.__file__))
    script = Path(__file__).resolve().parents[2] / "scripts" / "build_recipe_catalog.py"
    if script.is_file():
        code[str(script)] = _hash_file(script)
    frequencies = np.bincount(corpus.flat, minlength=corpus.n_vocab).tolist()
    metadata = {
        "schema_version": SCHEMA_VERSION, "corpus_sha256": corpus_info["sha256"],
        "text_index_sha256": text_info["sha256"], "n_recipes": corpus.n_recipes,
        "n_slots": len(corpus.flat), "vocabulary": corpus.itos,
        "ingredient_frequency": frequencies, "partial": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_name(f".{output.name}.{uuid.uuid4().hex}.building")
    descriptor = os.open(staged, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    os.close(descriptor)
    arrow_threads = (pa.cpu_count(), pa.io_thread_count())
    pa.set_cpu_count(min(2, arrow_threads[0]))
    pa.set_io_thread_count(min(2, arrow_threads[1]))
    connection = None
    try:
        connection = sqlite3.connect(staged)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA cache_size=-131072")
        connection.execute("PRAGMA threads=2")
        connection.executescript(_SCHEMA)
        _set_metadata(connection, metadata)
        progress({"phase": "inputs_verified", "rows": corpus.n_recipes, "slots": len(corpus.flat)})
        records = metadata_records
        if records is None:
            records = _default_metadata(corpus, paths, inputs, code, source_audit)
        _stage_metadata(connection, records, corpus, source_audit, batch_size, progress)
        tokens = [f"i{index}" for index in range(corpus.n_vocab)]
        per_source = {source: _empty_coverage() for source in sources}
        digest = hashlib.sha256()
        index = 0
        for batch in parquet.iter_batches(
                columns=["idx", *TEXT_COLUMNS], batch_size=batch_size, use_threads=False):
            data = batch.to_pydict()
            if not np.array_equal(data["idx"], np.arange(index, index + batch.num_rows)):
                raise RuntimeError(f"text row identity misalignment at row {index}")
            if not np.array_equal(data["source"], corpus.source[index:index + batch.num_rows]):
                raise RuntimeError(f"text source identity misalignment at row {index}")
            rows, fts_rows, used_keys = [], [], []
            for values in zip(*(data[name] for name in TEXT_COLUMNS), strict=True):
                source = values[0]
                ids = corpus.recipe(index)
                blob = ids.astype("<u2", copy=False).tobytes()
                supported = source_audit[source]["supported"]
                record = None
                if supported:
                    key = _text_key(values)
                    record = connection.execute(_ENRICHMENT_SELECT, (key,)).fetchone()
                    if record is None and require_metadata_match:
                        raise RuntimeError(
                            f"metadata text identity misalignment at corpus row {index} ({source})")
                    if record is not None:
                        if record[0] != blob:
                            raise RuntimeError(f"metadata ingredient alignment lost at corpus row {index}")
                        used_keys.append((key,))
                resolved = _resolved_metadata(record, supported=supported)
                rows.append((
                    index, source, str(corpus.lang[index]), values[1], values[2], blob,
                    *values[3:], *resolved))
                fts_rows.append((index, " ".join(tokens[int(value)] for value in ids)))
                digest.update(struct.pack("<Q", index))
                _hash_fields(digest, values)
                _count_coverage(per_source[source], values, len(ids), resolved)
                index += 1
            connection.executemany(
                "INSERT INTO recipes VALUES (" + ",".join("?" for _ in range(23)) + ")", rows)
            connection.executemany(
                "INSERT INTO recipe_fts(rowid,ingredient_tokens) VALUES (?,?)", fts_rows)
            connection.executemany(
                "UPDATE _enrichment SET matches=matches+1 WHERE fingerprint=?", used_keys)
            connection.commit()
            if index % (batch_size * 16) == 0:
                progress({"phase": "catalog_rows", "rows": index, "source": source})
        if index != corpus.n_recipes:
            raise RuntimeError("text stream ended before all corpus rows were stored")
        for source, keys, copies, matched, ambiguous in connection.execute("""
                SELECT source,COUNT(*),SUM(copies),SUM(matches>0),
                    SUM(total_conflict OR component_conflict OR servings_conflict)
                FROM _enrichment GROUP BY source"""):
            source_audit[source].update(
                unique_text_keys=keys, duplicate_metadata_records=copies - keys,
                matched_text_keys=matched, unmatched_text_keys=keys - matched,
                ambiguous_text_keys=ambiguous)
        connection.execute("DROP TABLE _enrichment")
        progress({"phase": "indexing", "rows": index})
        connection.executescript("""
            CREATE INDEX recipes_total_minutes ON recipes(total_minutes)
                WHERE total_minutes IS NOT NULL;
            CREATE INDEX recipes_servings ON recipes(servings) WHERE servings IS NOT NULL;
            CREATE INDEX recipes_source ON recipes(source);
            ANALYZE;
        """)
        connection.execute("INSERT INTO recipe_fts(recipe_fts) VALUES ('optimize')")
        connection.commit()
        progress({"phase": "integrity_check", "rows": index})
        verification = _verify_catalog(
            connection, corpus, digest.hexdigest(), tokens, batch_size, progress,
            expected_coverage=per_source)
        progress({"phase": "rechecking_input_hashes"})
        for info in itertools.chain(inputs.values(), code.values()):
            if _hash_file(Path(info["path"])) != info:
                raise RuntimeError(f"input changed during catalog build: {info['path']}")
        totals = _empty_coverage()
        for coverage in per_source.values():
            for name, value in coverage.items():
                if isinstance(value, Counter):
                    totals[name].update(value)
                else:
                    totals[name] += value
        coverage = {"total": totals, "by_source": per_source}
        sizes = np.diff(corpus.offsets)
        report = {
            "schema_version": SCHEMA_VERSION, "partial": False,
            "corpus_sha256": corpus_info["sha256"], "text_index_sha256": text_info["sha256"],
            "n_recipes": corpus.n_recipes, "n_slots": len(corpus.flat), "n_vocab": corpus.n_vocab,
            "min_ingredients": int(sizes.min()), "max_ingredients": int(sizes.max()),
            "singletons": int(np.count_nonzero(sizes == 1)),
            "coverage": coverage, "coverage_definitions": CONTENT_COVERAGE_DEFINITIONS,
            "sources": {
                source: {**audit, "inputs": [_report_file(info) for info in audit["inputs"]]}
                for source, audit in source_audit.items()
            },
            "inputs": [_report_file(info) for info in inputs.values()],
            "code": [_report_file(info) for info in code.values()],
            "verification": verification,
            "build": {
                "started_at": started_at, "completed_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "sqlite_version": sqlite3.sqlite_version, "batch_size": batch_size,
                "sqlite_threads": 2, "text_arrow_use_threads": False,
                "arrow_cpu_threads": pa.cpu_count(), "arrow_io_threads": pa.io_thread_count(),
                "require_metadata_match": require_metadata_match,
                "artifact_name": output.name,
                "privacy": "local-only; report contains no source text or filesystem paths",
                "join": "exact v2 text fingerprint plus normalized ingredient identity",
            },
        }
        metadata.update(
            partial=False, coverage=coverage, coverage_definitions=CONTENT_COVERAGE_DEFINITIONS,
            provenance={
                "sources": report["sources"], "inputs": report["inputs"], "code": report["code"],
                "build": report["build"]}, verification=verification)
        _set_metadata(connection, metadata)
        connection.close()
        connection = None
        load_catalog_metadata(staged)
        with staged.open("rb") as stream:
            os.fsync(stream.fileno())
        report["build"]["output_bytes"] = staged.stat().st_size
        os.link(staged, output)
        directory = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        progress({"phase": "published", "rows": index, "bytes": output.stat().st_size})
        return report
    finally:
        parquet.close()
        if connection is not None:
            connection.close()
        staged.unlink(missing_ok=True)
        for suffix in ("-journal", "-wal", "-shm"):
            Path(str(staged) + suffix).unlink(missing_ok=True)
        pa.set_cpu_count(arrow_threads[0])
        pa.set_io_thread_count(arrow_threads[1])


def load_catalog_metadata(path: Path) -> dict:
    """Read and validate complete catalog metadata without creating/writing a DB."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        metadata = {
            key: json.loads(value) for key, value in connection.execute("SELECT key,value FROM metadata")
        }
    finally:
        connection.close()
    if not REQUIRED_METADATA.issubset(metadata):
        raise RuntimeError("catalog lacks required metadata")
    if type(metadata["schema_version"]) is not int or metadata["schema_version"] != SCHEMA_VERSION:
        raise RuntimeError("unsupported catalog schema")
    if metadata["partial"] is not False:
        raise RuntimeError("incomplete catalogs cannot be used for search")
    for name in ("corpus_sha256", "text_index_sha256"):
        if not isinstance(metadata[name], str) or not re.fullmatch(r"[0-9a-f]{64}", metadata[name]):
            raise RuntimeError(f"invalid catalog {name}")
    vocabulary, frequency = metadata["vocabulary"], metadata["ingredient_frequency"]
    if (not isinstance(vocabulary, list) or not vocabulary
            or any(not isinstance(value, str) for value in vocabulary)
            or len(set(vocabulary)) != len(vocabulary)
            or not isinstance(frequency, list) or len(frequency) != len(vocabulary)
            or type(metadata["n_recipes"]) is not int or metadata["n_recipes"] <= 0
            or type(metadata["n_slots"]) is not int or metadata["n_slots"] < metadata["n_recipes"]
            or any(type(value) is not int or not 0 <= value <= metadata["n_recipes"]
                   for value in frequency)
            or sum(frequency) != metadata["n_slots"] or not isinstance(metadata["coverage"], dict)):
        raise RuntimeError("catalog vocabulary/frequency/count metadata is invalid")
    return metadata
