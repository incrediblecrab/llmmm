"""Immutable, private, read-only metadata prefilter for a recipe catalog.

The adjacent ``<catalog filename>.metadata`` directory contains a fixed manifest
and six non-pickled NumPy arrays. It is a derived performance cache, never a
replacement for final SQL/Python constraint checks. Only source-provided totals
are copied; unknown numeric values are NaN, never zero or inferred durations.

``filter_ids`` always requires a stripped title, at least one stripped instruction
fragment, and ``text_status != 'unparsed_steps'``. Optional constraints intersect
this mask, preserving input order and duplicates. Invalid IDs/constraints,
partial catalogs, changed inputs, and corrupt caches raise rather than falling
back. Missing numeric values cannot satisfy an explicit numeric constraint.
The query's maximum-time bound may be zero (yielding no matches); stored source
totals and the minimum-servings bound must remain strictly positive.

Publication is atomic and no-replace, including against a competing empty output
directory. Catalog connections are read-only; active SQLite journal/WAL sidecars
are rejected because a main-file digest cannot bind their contents. Catalog and
array hashes are cached per process only while path/inode/size/mtime/ctime agree.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import numbers
import os
import re
import shutil
import sqlite3
import stat
import struct
import sys
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .._hashing import file_sha256
from .recipe_catalog import has_usable_steps, load_catalog_metadata

FORMAT_VERSION = 1
MANIFEST_FILE = "manifest.json"
MAX_LANGUAGES = 256
LANGUAGE_WIDTH = 32
BATCH_SIZE = 8192
ARRAY_DTYPES = {
    "total_minutes.npy": np.dtype("<f8"),
    "servings.npy": np.dtype("<f8"),
    "language_codes.npy": np.dtype("<u2"),
    "ingredient_counts.npy": np.dtype("<u2"),
    "readable.npy": np.dtype("bool"),
    "language_names.npy": np.dtype(f"<U{LANGUAGE_WIDTH}"),
}
PREDICATES = {
    "total_minutes": "finite positive source_total; NaN for SQL NULL; no derived sums",
    "servings": "finite positive source_servings; NaN for SQL NULL",
    "readable": (
        "bool((title or '').strip()) and "
        "any(fragment.strip() for fragment in (steps or '').split('\\x1f')) "
        "and text_status != 'unparsed_steps'"),
}
_LANGUAGE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,31}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_HASH_CACHE: OrderedDict[tuple, str] = OrderedDict()
_HASH_LOCK = threading.RLock()


class SearchMetadataError(ValueError):
    """An invalid catalog/cache must not be used to prefilter candidates."""


def default_search_metadata_path(catalog: Path) -> Path:
    catalog = Path(catalog)
    return catalog.with_name(catalog.name + ".metadata")


def _stamp(path: Path) -> tuple:
    resolved = path.resolve(strict=True)
    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode):
        raise SearchMetadataError("metadata inputs must be regular files")
    return (str(resolved), info.st_dev, info.st_ino, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _catalog_stamp(path: Path) -> tuple:
    resolved = path.resolve(strict=True)
    if any(os.path.lexists(str(resolved) + suffix) for suffix in ("-wal", "-shm", "-journal")):
        raise SearchMetadataError("catalog has an active or uncheckpointed SQLite sidecar")
    return _stamp(path)


def _cached_sha256(path: Path) -> tuple[str, tuple]:
    with _HASH_LOCK:
        before = _stamp(path)
        cached = _HASH_CACHE.get(before)
        if cached is not None:
            _HASH_CACHE.move_to_end(before)
            return cached, before
        digest = file_sha256(Path(before[0]))
        if _stamp(path) != before:
            raise SearchMetadataError("file changed while verifying its hash")
        _HASH_CACHE[before] = digest
        while len(_HASH_CACHE) > 64:
            _HASH_CACHE.popitem(last=False)
        return digest, before


def _json_hash(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")).hexdigest()


def _catalog_binding(catalog: Path):
    before = _catalog_stamp(catalog)
    digest, hashed_stamp = _cached_sha256(catalog)
    try:
        metadata = load_catalog_metadata(catalog)
    except (sqlite3.Error, RuntimeError, ValueError) as error:
        raise SearchMetadataError("catalog metadata is corrupt, partial, or invalid") from error
    if len(metadata["vocabulary"]) > 65536:
        raise SearchMetadataError("catalog vocabulary exceeds uint16 ingredient IDs")
    if before != hashed_stamp or _catalog_stamp(catalog) != before:
        raise SearchMetadataError("catalog changed while reading its metadata")
    try:
        identity = {
            "sha256": digest, "bytes": before[3],
            "schema_version": metadata["schema_version"],
            "metadata_sha256": _json_hash(metadata),
            "corpus_sha256": metadata["corpus_sha256"],
            "text_index_sha256": metadata["text_index_sha256"],
            "vocabulary_sha256": _json_hash(metadata["vocabulary"]),
            "ingredient_frequency_sha256": _json_hash(metadata["ingredient_frequency"]),
            "n_recipes": metadata["n_recipes"], "n_slots": metadata["n_slots"],
            "n_vocab": len(metadata["vocabulary"]),
        }
    except (ValueError, TypeError) as error:
        raise SearchMetadataError("catalog metadata is not finite JSON") from error
    return identity, before, metadata


def _read_connection(catalog: Path):
    connection = sqlite3.connect(catalog.resolve().as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-32768")
    connection.execute("PRAGMA query_only=ON")
    return connection


def _close_arrays(arrays):
    for array in arrays.values():
        mapping = getattr(array, "_mmap", None)
        if mapping is not None:
            mapping.close()


def _numeric(value, status, expected_status, name):
    if not isinstance(status, str):
        raise SearchMetadataError(f"{name} status must be a string")
    if value is None:
        if status == expected_status:
            raise SearchMetadataError(f"{name} claims a source value but is NULL")
        return np.nan
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or value <= 0 or status != expected_status):
        raise SearchMetadataError(f"{name} is not a finite positive {expected_status} value")
    return float(value)


def _publish_directory(staged: Path, output: Path) -> None:
    """Atomic no-replace rename; ordinary POSIX rename can clobber an empty dir."""
    if os.name == "nt":
        os.rename(staged, output)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        result = rename(os.fsencode(staged), os.fsencode(output), 0x00000004)
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                           ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        result = rename(-100, os.fsencode(staged), -100, os.fsencode(output), 1)
    else:
        raise SearchMetadataError("this platform lacks an atomic no-replace directory rename")
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(output))


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_search_metadata(catalog: Path, output: Path | None = None) -> dict:
    """Stream a complete catalog into a new private cache; never mutate SQLite."""
    started = time.perf_counter()
    catalog = Path(catalog)
    output = Path(output) if output is not None else default_search_metadata_path(catalog)
    if os.path.lexists(output):
        raise FileExistsError(f"{output}: metadata indexes are never overwritten")
    identity, catalog_stamp, metadata = _catalog_binding(catalog)
    n_recipes, n_vocab = identity["n_recipes"], identity["n_vocab"]
    connection = _read_connection(catalog)
    arrays = {}
    staged = None
    try:
        if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            raise SearchMetadataError("catalog SQLite integrity check failed")
        actual = connection.execute("SELECT COUNT(*),MIN(id),MAX(id) FROM recipes").fetchone()
        if actual != (n_recipes, 0, n_recipes - 1):
            raise SearchMetadataError("catalog IDs/counts are incomplete or noncontiguous")
        output.parent.mkdir(parents=True, exist_ok=True)
        staged = output.with_name(f".{output.name}.{uuid.uuid4().hex}.building")
        staged.mkdir(mode=0o700)
        for filename, dtype in ARRAY_DTYPES.items():
            if filename == "language_names.npy":
                continue
            arrays[filename] = np.lib.format.open_memmap(
                staged / filename, mode="w+", dtype=dtype, shape=(n_recipes,), version=(1, 0))
            os.chmod(staged / filename, 0o600)
        language_ids = {}
        source_counts = {}
        frequency = [0] * n_vocab
        counts = {"readable": 0, "source_totals": 0, "servings": 0, "ingredient_slots": 0}
        index = 0
        cursor = connection.execute("""
            SELECT id,source,language,ingredient_ids,title,steps,text_status,
                total_minutes,time_status,servings,servings_status
            FROM recipes ORDER BY id
        """)
        while rows := cursor.fetchmany(BATCH_SIZE):
            for (row_id, source, language, blob, title, steps, text_status,
                 total, time_status, servings, servings_status) in rows:
                if row_id != index or index >= n_recipes:
                    raise SearchMetadataError("catalog IDs are noncontiguous or duplicated")
                if not isinstance(source, str) or not source or len(source) > 256:
                    raise SearchMetadataError("catalog source identity is invalid")
                if not isinstance(language, str) or not _LANGUAGE.fullmatch(language):
                    raise SearchMetadataError("catalog language is invalid or too long")
                if language not in language_ids:
                    if len(language_ids) >= MAX_LANGUAGES:
                        raise SearchMetadataError("catalog exceeds the bounded language vocabulary")
                    language_ids[language] = len(language_ids)
                if not isinstance(blob, bytes) or not blob or len(blob) % 2:
                    raise SearchMetadataError("ingredient IDs must be a nonempty uint16 BLOB")
                n_ingredients = len(blob) // 2
                if n_ingredients > min(65535, n_vocab):
                    raise SearchMetadataError("ingredient count exceeds uint16 or vocabulary bounds")
                previous = -1
                for (ingredient,) in struct.iter_unpack("<H", blob):
                    if ingredient <= previous or ingredient >= n_vocab:
                        raise SearchMetadataError("ingredient IDs are unsorted, duplicated, or out of range")
                    frequency[ingredient] += 1
                    previous = ingredient
                if any(value is not None and not isinstance(value, str)
                       for value in (title, steps, text_status)):
                    raise SearchMetadataError("catalog content fields must be strings or NULL")
                total = _numeric(total, time_status, "source_total", "total_minutes")
                servings = _numeric(servings, servings_status, "source_servings", "servings")
                readable = bool(title and title.strip()) and has_usable_steps(steps) and (
                    text_status != "unparsed_steps")
                arrays["total_minutes.npy"][index] = total
                arrays["servings.npy"][index] = servings
                arrays["language_codes.npy"][index] = language_ids[language]
                arrays["ingredient_counts.npy"][index] = n_ingredients
                arrays["readable.npy"][index] = readable
                counts["readable"] += readable
                counts["source_totals"] += not math.isnan(total)
                counts["servings"] += not math.isnan(servings)
                counts["ingredient_slots"] += n_ingredients
                source_counts[source] = source_counts.get(source, 0) + 1
                index += 1
        if index != n_recipes or counts["ingredient_slots"] != identity["n_slots"]:
            raise SearchMetadataError("catalog row/ingredient-slot coverage is incomplete")
        if frequency != metadata["ingredient_frequency"]:
            raise SearchMetadataError("catalog ingredient document frequencies are misaligned")
        recorded_sources = metadata["coverage"].get("by_source")
        if recorded_sources is not None and {
                key: value.get("n_recipes") for key, value in recorded_sources.items()} != source_counts:
            raise SearchMetadataError("catalog source row counts are misaligned")
        arrays["language_names.npy"] = np.lib.format.open_memmap(
            staged / "language_names.npy", mode="w+", dtype=ARRAY_DTYPES["language_names.npy"],
            shape=(len(language_ids),), version=(1, 0))
        arrays["language_names.npy"][:] = list(language_ids)
        os.chmod(staged / "language_names.npy", 0o600)
        descriptions = {}
        for filename, array in arrays.items():
            array.flush()
            with (staged / filename).open("rb") as stream:
                os.fsync(stream.fileno())
            descriptions[filename] = {
                "sha256": file_sha256(staged / filename),
                "bytes": (staged / filename).stat().st_size,
                "dtype": array.dtype.str, "shape": list(array.shape),
            }
        _close_arrays(arrays)
        arrays.clear()
        connection.close()
        connection = None
        if _catalog_stamp(catalog) != catalog_stamp:
            raise SearchMetadataError("catalog changed during metadata-index construction")
        manifest = {
            "format_version": FORMAT_VERSION, "partial": False,
            "catalog": identity, "n_languages": len(language_ids),
            "arrays": descriptions, "predicates": PREDICATES,
            "coverage": counts, "source_counts": source_counts,
            "build": {
                "elapsed_seconds": round(time.perf_counter() - started, 6),
                "batch_size": BATCH_SIZE, "python": sys.version.split()[0],
                "numpy": np.__version__, "builder_sha256": file_sha256(Path(__file__)),
                "catalog_modified": False, "all_record_ids_checked": True,
                "all_ingredient_ids_and_frequencies_checked": True,
            },
        }
        descriptor = os.open(staged / MANIFEST_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        checked = RecipeSearchMetadata.load(catalog, staged)
        checked.close()
        _fsync_directory(staged)
        if _catalog_stamp(catalog) != catalog_stamp:
            raise SearchMetadataError("catalog changed before metadata publication")
        output_bytes = sum((staged / filename).stat().st_size
                           for filename in (*ARRAY_DTYPES, MANIFEST_FILE))
        _publish_directory(staged, output)
        staged = None
        _fsync_directory(output.parent)
        return {
            **manifest, "artifact_name": output.name, "output_bytes": output_bytes,
            "verified": True, "elapsed_seconds": round(time.perf_counter() - started, 6),
        }
    except sqlite3.Error as error:
        raise SearchMetadataError("catalog SQL records are corrupt or invalid") from error
    finally:
        _close_arrays(arrays)
        if connection is not None:
            connection.close()
        if staged is not None:
            shutil.rmtree(staged)


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SearchMetadataError("duplicate keys in metadata manifest")
        result[key] = value
    return result


def _finite_bound(value, name, *, allow_zero=False):
    if value is None:
        return None
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
            or not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero)):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {qualifier} number")
    return float(value)


@dataclass(frozen=True)
class RecipeSearchMetadata:
    catalog: Path
    path: Path
    n_recipes: int
    total_minutes: np.ndarray
    servings: np.ndarray
    language_codes: np.ndarray
    language_names: np.ndarray
    ingredient_counts: np.ndarray
    readable: np.ndarray
    _catalog_stamp: tuple = field(repr=False)
    _file_stamps: dict = field(repr=False)
    _language_ids: dict = field(repr=False)
    _closed: bool = field(default=False, repr=False)

    @classmethod
    def load(cls, catalog: Path, path: Path | None = None) -> "RecipeSearchMetadata":
        catalog = Path(catalog)
        path = Path(path) if path is not None else default_search_metadata_path(catalog)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_symlink() or not path.is_dir():
            raise SearchMetadataError("metadata index must be a real directory")
        expected_files = {*ARRAY_DTYPES, MANIFEST_FILE}
        if {entry.name for entry in path.iterdir()} != expected_files:
            raise SearchMetadataError("metadata index has missing or unexpected files")
        if any((path / name).is_symlink() for name in expected_files):
            raise SearchMetadataError("metadata files must not be symlinks")
        if (path / MANIFEST_FILE).stat().st_size > 64 * 1024:
            raise SearchMetadataError("metadata manifest exceeds its bounded size")
        manifest_hash, manifest_stamp = _cached_sha256(path / MANIFEST_FILE)
        try:
            manifest = json.loads((path / MANIFEST_FILE).read_text(encoding="utf-8"),
                                  object_pairs_hook=_unique_json_object)
        except (ValueError, UnicodeError) as error:
            raise SearchMetadataError("metadata manifest is not valid JSON") from error
        if (not isinstance(manifest, dict) or type(manifest.get("format_version")) is not int
                or manifest["format_version"] != FORMAT_VERSION or manifest.get("partial") is not False
                or manifest.get("predicates") != PREDICATES):
            raise SearchMetadataError("unsupported or partial metadata manifest")
        if not isinstance(manifest.get("arrays"), dict) or set(manifest["arrays"]) != set(ARRAY_DTYPES):
            raise SearchMetadataError("manifest must contain only the fixed array filenames")
        identity, catalog_stamp, _metadata = _catalog_binding(catalog)
        if manifest.get("catalog") != identity:
            raise SearchMetadataError("metadata index does not match the exact catalog identity")
        n_languages = manifest.get("n_languages")
        if type(n_languages) is not int or not 1 <= n_languages <= min(
                MAX_LANGUAGES, identity["n_recipes"]):
            raise SearchMetadataError("invalid bounded language vocabulary")
        arrays, stamps = {}, {MANIFEST_FILE: manifest_stamp}
        try:
            for filename, dtype in ARRAY_DTYPES.items():
                shape = (n_languages if filename == "language_names.npy" else identity["n_recipes"],)
                descriptor = manifest["arrays"][filename]
                if (not isinstance(descriptor, dict)
                        or set(descriptor) != {"sha256", "bytes", "dtype", "shape"}
                        or descriptor["shape"] != list(shape) or descriptor["dtype"] != dtype.str
                        or not isinstance(descriptor["sha256"], str)
                        or not _SHA256.fullmatch(descriptor["sha256"])
                        or type(descriptor["bytes"]) is not int
                        or not shape[0] * dtype.itemsize < descriptor["bytes"] <= shape[0] * dtype.itemsize + 1024):
                    raise SearchMetadataError("invalid array shape, dtype, or hash descriptor")
                digest, stamp = _cached_sha256(path / filename)
                if stamp[3] != descriptor["bytes"] or digest != descriptor["sha256"]:
                    raise SearchMetadataError(f"metadata array hash/size mismatch: {filename}")
                try:
                    array = np.load(path / filename, allow_pickle=False, mmap_mode="r", max_header_size=1024)
                except (ValueError, OSError, EOFError) as error:
                    raise SearchMetadataError(f"invalid safe NumPy array: {filename}") from error
                if not isinstance(array, np.memmap):
                    if hasattr(array, "close"):
                        array.close()
                    raise SearchMetadataError("metadata files must be individual NumPy arrays")
                arrays[filename] = array
                if (array.shape != shape or array.dtype != dtype or array.flags.writeable
                        or array.offset + array.nbytes != stamp[3]):
                    raise SearchMetadataError(f"array header or layout mismatch: {filename}")
                if _stamp(path / filename) != stamp:
                    raise SearchMetadataError("metadata array changed while loading")
                stamps[filename] = stamp
            names = arrays["language_names.npy"].tolist()
            if len(set(names)) != n_languages or any(not _LANGUAGE.fullmatch(name) for name in names):
                raise SearchMetadataError("language names are invalid or duplicated")
            counted = {"readable": 0, "source_totals": 0, "servings": 0, "ingredient_slots": 0}
            for start in range(0, identity["n_recipes"], 65536):
                selection = slice(start, start + 65536)
                total, servings = (arrays[name][selection] for name in ("total_minutes.npy", "servings.npy"))
                if (np.any(np.isinf(total)) or np.any(total <= 0)
                        or np.any(np.isinf(servings)) or np.any(servings <= 0)):
                    raise SearchMetadataError("cached numeric metadata must be positive finite values or NaN")
                codes = arrays["language_codes.npy"][selection]
                ingredients = arrays["ingredient_counts.npy"][selection]
                readable = arrays["readable.npy"][selection]
                if (np.any(codes >= n_languages) or np.any(ingredients == 0)
                        or np.any(ingredients > min(identity["n_vocab"], 65535))
                        or np.any(readable.view(np.uint8) > 1)):
                    raise SearchMetadataError("cached codes, counts, or readability flags are invalid")
                counted["readable"] += int(np.count_nonzero(readable))
                counted["source_totals"] += int(np.count_nonzero(~np.isnan(total)))
                counted["servings"] += int(np.count_nonzero(~np.isnan(servings)))
                counted["ingredient_slots"] += int(ingredients.sum(dtype=np.uint64))
            if counted != manifest.get("coverage") or counted["ingredient_slots"] != identity["n_slots"]:
                raise SearchMetadataError("cached aggregate coverage is inconsistent")
            if (_catalog_stamp(catalog) != catalog_stamp
                    or _stamp(path / MANIFEST_FILE) != manifest_stamp
                    or _cached_sha256(path / MANIFEST_FILE)[0] != manifest_hash):
                raise SearchMetadataError("catalog or manifest changed during loading")
            return cls(
                catalog=catalog.absolute(), path=path.absolute(), n_recipes=identity["n_recipes"],
                total_minutes=arrays["total_minutes.npy"], servings=arrays["servings.npy"],
                language_codes=arrays["language_codes.npy"], language_names=arrays["language_names.npy"],
                ingredient_counts=arrays["ingredient_counts.npy"], readable=arrays["readable.npy"],
                _catalog_stamp=catalog_stamp, _file_stamps=stamps,
                _language_ids={name: index for index, name in enumerate(names)})
        except BaseException:
            _close_arrays(arrays)
            raise

    def filter_ids(
            self, ids: Sequence[int], *, max_total_minutes=None, min_servings=None,
            language=None, max_ingredients=None) -> list[int]:
        """Return readable, feasible IDs in input order, retaining duplicates."""
        if self._closed:
            raise SearchMetadataError("metadata index is closed")
        if _catalog_stamp(self.catalog) != self._catalog_stamp:
            raise SearchMetadataError("catalog changed after metadata initialization")
        if any(_stamp(self.path / name) != stamp for name, stamp in self._file_stamps.items()):
            raise SearchMetadataError("metadata index changed after initialization")
        max_total_minutes = _finite_bound(max_total_minutes, "max_total_minutes", allow_zero=True)
        min_servings = _finite_bound(min_servings, "min_servings")
        if language is not None and (not isinstance(language, str) or not _LANGUAGE.fullmatch(language)):
            raise ValueError("language must be a bounded language-code string")
        if max_ingredients is not None and (
                isinstance(max_ingredients, (bool, np.bool_))
                or not isinstance(max_ingredients, (int, np.integer)) or max_ingredients <= 0):
            raise ValueError("max_ingredients must be a positive integer")
        if isinstance(ids, np.ndarray):
            if ids.ndim != 1 or ids.dtype.kind not in "iu":
                raise ValueError("recipe IDs must be a one-dimensional integer sequence")
            indices = ids
        else:
            if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)):
                raise ValueError("recipe IDs must be an integer sequence")
            if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                   for value in ids):
                raise ValueError("recipe IDs must be integers, not booleans or coercible strings")
            try:
                indices = np.asarray(ids, dtype=np.int64)
            except (ValueError, OverflowError) as error:
                raise ValueError("recipe IDs are outside supported integer bounds") from error
        if np.any(indices < 0) or np.any(indices >= self.n_recipes):
            raise ValueError("recipe ID is outside the catalog")
        indices = indices.astype(np.int64, copy=False)
        keep = self.readable[indices].copy()
        if max_total_minutes is not None:
            keep &= self.total_minutes[indices] <= max_total_minutes
        if min_servings is not None:
            keep &= self.servings[indices] >= min_servings
        if max_ingredients is not None and max_ingredients < 65535:
            keep &= self.ingredient_counts[indices] <= max_ingredients
        if language is not None:
            code = self._language_ids.get(language)
            if code is None:
                return []
            keep &= self.language_codes[indices] == code
        return indices[keep].tolist()

    def close(self) -> None:
        if not self._closed:
            _close_arrays({name: getattr(self, name.removesuffix(".npy")) for name in ARRAY_DTYPES})
            object.__setattr__(self, "_closed", True)
