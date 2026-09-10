"""Verify full ingredient-only coverage and compare browser retrieval with Python."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ingredient_model._hashing import file_sha256
from ingredient_model.ingredient_catalog import load_ingredient_catalog, public_source_url
from ingredient_model.ingredient_demo import load_ingredient_policy
from ingredient_model.recipe_demo import browser_policy
from ingredient_model.recipe_ingredients import CanonicalIngredientIndex
from ingredient_model.recipe_ranker import candidate_features, heuristic_scores

QUERIES = [
    {"available_ingredients": ["chicken", "rice", "broccoli"], "max_total_minutes": 30, "max_missing": 2},
    {"available_ingredients": ["tomato", "basil", "olive_oil", "salt", "garlic"], "max_missing": 2},
    {"available_ingredients": ["egg", "milk", "flour", "butter", "salt"], "max_missing": 2, "min_servings": 2},
    {"available_ingredients": ["potato", "onion", "salt", "oil"], "must_use": ["potato"],
     "exclude": ["milk"], "max_total_minutes": 30, "max_missing": 2},
    {"available_ingredients": ["lentil", "carrot", "onion", "water"], "language": "en",
     "max_missing": 3, "max_total_minutes": 60},
    {"available_ingredients": ["egg"], "max_total_minutes": 0, "max_missing": 0},
    {"available_ingredients": ["rice"], "max_missing": None, "max_total_minutes": 30},
    {"available_ingredients": ["water"], "max_missing": 0},
    {"available_ingredients": ["tomato", "basil", "olive_oil", "salt", "garlic"],
     "max_missing": 2, "require_source_url": True},
    {"available_ingredients": ["chicken", "rice", "broccoli"], "max_total_minutes": 30,
     "max_missing": 2, "require_source_url": True},
]


def reference_search(metadata: dict, arrays: dict, query: dict, policy, *, max_candidates: int = 2000) -> dict:
    lookup = {name: position for position, name in enumerate(metadata["vocabulary"])}
    offsets = np.r_[0, np.cumsum(arrays["lengths"], dtype=np.int64)]
    available = {lookup[name] for name in query["available_ingredients"]}
    required = {lookup[name] for name in query.get("must_use", [])}
    excluded = {lookup[name] for name in query.get("exclude", [])}
    searchable = (available - excluded) | required
    n_vocab = len(lookup)

    def counts(ids):
        membership = np.zeros(n_vocab, dtype=np.int8)
        membership[list(ids)] = 1
        return np.add.reduceat(membership[arrays["ingredients"]], offsets[:-1], dtype=np.int32)

    overlap = counts(available)
    eligible = counts(searchable) > 0
    if query.get("require_source_url", False):
        eligible &= arrays["has_source_url"] != 0
    if required:
        eligible &= counts(required) == len(required)
    if excluded:
        eligible &= counts(excluded) == 0
    missing = query.get("max_missing", 2)
    if missing is not None:
        eligible &= arrays["lengths"].astype(np.int32) - overlap <= missing
    if query.get("max_total_minutes") is not None:
        eligible &= arrays["total_minutes"] <= query["max_total_minutes"]
    if query.get("min_servings") is not None:
        eligible &= arrays["servings"] >= query["min_servings"]
    if query.get("language") is not None:
        code = metadata["language_names"].index(query["language"]) if query["language"] in metadata["language_names"] else -1
        eligible &= arrays["language_codes"] == code
    ids = np.flatnonzero(eligible)
    priority = np.empty(len(ids), dtype=np.float64)
    frequency = np.asarray(metadata["ingredient_frequency"])

    def features(selected):
        sets = [arrays["ingredients"][offsets[value]:offsets[value + 1]] for value in selected]
        times = [float(arrays["total_minutes"][value]) if np.isfinite(arrays["total_minutes"][value]) else None
                 for value in selected]
        return candidate_features(sorted(available), sets, ingredient_frequency=frequency,
                                  n_recipes=metadata["n_recipes"], total_minutes=times,
                                  max_total_minutes=query.get("max_total_minutes"))

    for first in range(0, len(ids), 8192):
        priority[first:first + 8192] = heuristic_scores(features(ids[first:first + 8192]))
    order = np.lexsort((ids, -priority))[:max_candidates]
    selected = np.sort(ids[order])
    selected_features = features(selected) if len(selected) else np.empty((0, 20), dtype=np.float32)
    result = {"feasible_count": len(ids), "candidates_scored": len(selected)}
    for name, scores in (("learned", policy.score(selected_features)), ("heuristic", heuristic_scores(selected_features))):
        top = np.lexsort((selected, -scores))[:query.get("top_k", 5)]
        result[name] = {
            "ids": selected[top].tolist(),
            "scores": scores[top].tolist(),
        }
    return result


def verify(index_directory: Path, corpus_path: Path, repository: Path, *, ipv4: bool = False) -> dict:
    torch.set_num_threads(4)
    metadata, arrays = load_ingredient_catalog(index_directory)
    canonical = CanonicalIngredientIndex.load(
        corpus_path, corpus_sha256=metadata["identity"]["corpus_sha256"],
        n_recipes=metadata["n_recipes"], n_slots=metadata["n_slots"],
        n_vocab=len(metadata["vocabulary"]))
    if (not np.array_equal(arrays["ingredients"], canonical.flat)
            or not np.array_equal(arrays["lengths"], np.diff(canonical.offsets))):
        raise ValueError("the exported index is not exactly the complete canonical corpus")
    for field, measured in (
            ("records", len(arrays["lengths"])), ("ingredient_slots", len(arrays["ingredients"])),
            ("min_ingredients", int(arrays["lengths"].min())),
            ("max_ingredients", int(arrays["lengths"].max())),
            ("singletons", int(np.count_nonzero(arrays["lengths"] == 1))),
            ("source_total_times", int(np.count_nonzero(np.isfinite(arrays["total_minutes"])))),
            ("source_servings", int(np.count_nonzero(np.isfinite(arrays["servings"]))))):
        if metadata["coverage"][field] != measured:
            raise ValueError(f"declared {field} coverage differs from the actual array")
    url_count = 0
    for record in metadata["url_shards"]:
        path = index_directory / record["file"]
        if file_sha256(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError("a source URL shard failed its compressed hash check")
        with gzip.open(path, "rb") as stream:
            raw = stream.read(record["raw_bytes"] + 1)
        if len(raw) != record["raw_bytes"] or hashlib.sha256(raw).hexdigest() != record["raw_sha256"]:
            raise ValueError("a source URL shard failed its decompressed integrity check")
        shard = json.loads(raw)
        if (set(shard) != {"first_id", "urls"} or shard["first_id"] != record["first_id"]
                or len(shard["urls"]) != record["rows"]):
            raise ValueError("a source URL shard contains prose fields or misaligned records")
        for position, url in enumerate(shard["urls"]):
            if bool(arrays["has_source_url"][shard["first_id"] + position]) != (url is not None):
                raise ValueError("a URL availability flag differs from the actual recorded link")
            if url is not None:
                normalized, status = public_source_url(url)
                if normalized != url or status != "source_url":
                    raise ValueError("a source URL failed the public-link policy")
                url_count += 1
    if url_count != metadata["coverage"]["url_statuses"]["source_url"]:
        raise ValueError("source URL coverage differs from its recorded count")
    _, policy, model_source = load_ingredient_policy(repository, metadata, ipv4=ipv4)
    process = subprocess.run(
        ["node", str(repository / "model/demo/test/full_catalog_bridge.js")],
        input=json.dumps({"directory": str(index_directory), "policy": browser_policy(policy, model_source),
                          "queries": QUERIES}), text=True, capture_output=True, check=True, timeout=120)
    browser = json.loads(process.stdout)
    if browser["records"] != metadata["n_recipes"] or browser["slots"] != metadata["n_slots"]:
        raise ValueError("the browser did not verify the entire ingredient population")
    if len(browser["results"]) != len(QUERIES) * 2:
        raise ValueError("browser search results are incomplete")
    checked = []
    max_score_error = 0.0
    for position, query in enumerate(QUERIES):
        reference = reference_search(metadata, arrays, query, policy)
        timings = {}
        for offset, ranking in enumerate(("learned", "heuristic")):
            observed = browser["results"][position * 2 + offset]
            if (observed["query"] != query or observed["ranking"] != ranking
                    or observed["scanned"] != metadata["n_recipes"]
                    or observed["feasible_count"] != reference["feasible_count"]
                    or observed["candidates_scored"] != reference["candidates_scored"]
                    or [match["id"] for match in observed["matches"]] != reference[ranking]["ids"]):
                raise ValueError(f"browser/Python shortlist or ordered results differ for {query!r} ({ranking})")
            differences = np.abs(np.asarray([match["score"] for match in observed["matches"]])
                                 - np.asarray(reference[ranking]["scores"]))
            error = float(differences.max()) if len(differences) else 0.0
            if error > 1e-4:
                raise ValueError("browser/Python scores differ beyond the declared tolerance")
            max_score_error = max(max_score_error, error)
            timings[ranking] = observed["elapsed_ms"]
        checked.append({"query": query, "feasible_records": reference["feasible_count"],
                        "candidates_scored": reference["candidates_scored"], "node_search_ms": timings})
    canonical.check_unchanged()
    return {
        "schema_version": 1, "status": "verified_local_complete_ingredient_index",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "index_sha256": file_sha256(index_directory / "ingredient-index.json"),
        "corpus_sha256": metadata["identity"]["corpus_sha256"],
        "catalog_sha256": metadata["identity"]["catalog_sha256"],
        "model_revision": model_source["revision"],
        "model_input_identity_verified": True,
        "records_compared": metadata["n_recipes"], "ingredient_slots_compared": metadata["n_slots"],
        "source_urls_checked": url_count, "url_records_checked": metadata["n_recipes"],
        "source_urls_unavailable": metadata["n_recipes"] - url_count,
        "coverage": metadata["coverage"], "index_bytes": metadata["bytes"],
        "initial_download_bytes": metadata["bytes"]["initial_compressed_download"],
        "python_browser_searches_matched": len(browser["results"]),
        "max_absolute_score_error": max_score_error, "score_tolerance": 1e-4,
        "checks": checked, "node_cold_load_ms": browser["load_ms"],
        "node_process_memory_bytes": browser["memory"],
        "timing_scope": "Local Node process; excludes network, source-URL loading and UI. Not a service guarantee.",
        "prose_fields_in_export": False, "quality_benchmark": False, "public_upload": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("model/data/recipes/recipe_ids.npz"))
    parser.add_argument("--report", type=Path, required=True, help="new aggregate report path")
    parser.add_argument("--ipv4", action="store_true")
    args = parser.parse_args()
    if args.report.exists():
        parser.error("choose a new verification report; existing evidence is not overwritten")
    report = verify(args.index.resolve(), args.corpus.resolve(), Path(__file__).resolve().parents[2], ipv4=args.ipv4)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({name: report[name] for name in (
        "records_compared", "ingredient_slots_compared", "source_urls_checked",
        "python_browser_searches_matched", "max_absolute_score_error", "public_upload")}, indent=2))


if __name__ == "__main__":
    main()
