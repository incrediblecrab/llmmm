"""Verify full recipe-card index coverage and compare browser retrieval with Python."""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ingredient_model._hashing import file_sha256
from ingredient_model.data.recipe_search_metadata import _catalog_binding, _catalog_stamp, _read_connection
from ingredient_model.ingredient_catalog import (
    TEXT_COVERAGE, ingredient_lines, load_ingredient_catalog, load_text_shards, public_source_url,
    read_text_shard, recipe_title,
)
from ingredient_model.ingredient_dataset import _read_link_shard, _read_url_shard
from ingredient_model.ingredient_demo import load_ingredient_policy
from ingredient_model.recipe_demo import browser_policy
from ingredient_model.recipe_ingredients import CanonicalIngredientIndex
from ingredient_model.recipe_links import ARCHIVE, LINK_STATUSES, SOURCE, recipe_link
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
     "max_missing": 2, "require_link": True},
    {"available_ingredients": ["chicken", "rice", "broccoli"], "max_total_minutes": 30,
     "max_missing": 2, "require_link": True},
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
    if query.get("require_link", False):
        eligible &= np.isin(arrays["link_status"], (SOURCE, ARCHIVE))
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


def compare_with_catalog(index_directory: Path, metadata: dict, arrays: dict, catalog_path: Path) -> Counter:
    """Recompute every exported URL, card link, title and ingredient line from the private catalog, in ID order."""
    identity, stamp, _ = _catalog_binding(catalog_path)
    if identity["sha256"] != metadata["identity"]["catalog_sha256"]:
        raise ValueError("the private catalog is not the one this index was exported from")
    url_records, text_records = metadata["url_shards"], load_text_shards(index_directory, metadata)
    link_records = metadata["link_shards"]
    url_size, text_size = metadata["rows_per_url_shard"], metadata["rows_per_text_shard"]
    counts, link_statuses = Counter(), Counter()
    connection = _read_connection(catalog_path)
    try:
        cursor = connection.execute("SELECT id, source, url, title, raw_ingredients FROM recipes ORDER BY id")
        while batch := cursor.fetchmany(8192):
            for recipe_id, source, url, title, raw in batch:
                if recipe_id != counts["rows"] or recipe_id >= metadata["n_recipes"]:
                    raise ValueError("catalog record IDs are not complete, ordered and contiguous")
                if recipe_id % url_size == 0:
                    urls = _read_url_shard(index_directory, url_records[recipe_id // url_size])
                    links = _read_link_shard(index_directory, link_records[recipe_id // url_size])
                if recipe_id % text_size == 0:
                    titles, lines = read_text_shard(index_directory, text_records[recipe_id // text_size])
                exported = urls[recipe_id % url_size]
                if exported is not None:
                    normalized, status = public_source_url(exported)
                    if normalized != exported or status != "source_url":
                        raise ValueError("a source URL failed the public-link policy")
                    counts["source_urls"] += 1
                if exported != public_source_url(url)[0]:
                    raise ValueError(f"recipe {recipe_id}: the exported link differs from the catalog")
                link, link_status = recipe_link(public_source_url(url)[0], recipe_title(title))
                if links[recipe_id % url_size] != link or int(arrays["link_status"][recipe_id]) != link_status:
                    raise ValueError(f"recipe {recipe_id}: the card link differs from the per-site link rule")
                link_statuses[LINK_STATUSES[link_status]] += 1
                exported_title, exported_lines = titles[recipe_id % text_size], lines[recipe_id % text_size]
                if exported_title != recipe_title(title) or exported_lines != ingredient_lines(raw, source):
                    raise ValueError(f"recipe {recipe_id}: exported card text differs from the catalog")
                counts["recipe_titles"] += exported_title is not None
                counts["ingredient_line_records"] += exported_lines is not None
                counts["ingredient_lines"] += len(exported_lines or ())
                counts["rows"] += 1
    finally:
        connection.close()
    if _catalog_stamp(catalog_path) != stamp:
        raise ValueError("the private catalog changed during comparison")
    if counts["rows"] != metadata["n_recipes"]:
        raise ValueError("the catalog comparison did not cover every exported record")
    if counts["source_urls"] != metadata["coverage"]["url_statuses"]["source_url"]:
        raise ValueError("source URL coverage differs from its recorded count")
    if dict(link_statuses) != metadata["coverage"]["link_statuses"]:
        raise ValueError("card link coverage differs from its recorded counts")
    if any(counts[name] != metadata["coverage"][name] for name in TEXT_COVERAGE):
        raise ValueError("declared card-text coverage differs from the catalog")
    counts["link_statuses"] = dict(link_statuses)
    return counts


def verify(index_directory: Path, corpus_path: Path, catalog_path: Path, repository: Path, *,
           ipv4: bool = False) -> dict:
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
    compared = compare_with_catalog(index_directory, metadata, arrays, catalog_path)
    url_count = compared["source_urls"]
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
        "schema_version": 2, "status": "verified_local_complete_recipe_card_index",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "index_sha256": file_sha256(index_directory / "ingredient-index.json"),
        "corpus_sha256": metadata["identity"]["corpus_sha256"],
        "catalog_sha256": metadata["identity"]["catalog_sha256"],
        "model_revision": model_source["revision"],
        "model_input_identity_verified": True,
        "records_compared": metadata["n_recipes"], "ingredient_slots_compared": metadata["n_slots"],
        "source_urls_checked": url_count, "url_records_checked": metadata["n_recipes"],
        "source_urls_unavailable": metadata["n_recipes"] - url_count,
        "card_links_rederived": compared["rows"], "link_statuses": compared["link_statuses"],
        "catalog_records_compared": compared["rows"],
        "recipe_titles_compared": compared["recipe_titles"],
        "ingredient_line_records_compared": compared["ingredient_line_records"],
        "ingredient_lines_compared": compared["ingredient_lines"],
        "catalog_comparison_scope": (
            "Every exported URL, card link, title and ingredient line was recomputed from the private catalog in "
            "ID order with the export's own normalization functions and per-site link rule. This checks "
            "alignment, completeness and integrity; unit tests and the link-health sample, not this comparison, "
            "check the rules themselves."),
        "coverage": metadata["coverage"], "index_bytes": metadata["bytes"],
        "initial_download_bytes": metadata["bytes"]["initial_compressed_download"],
        "python_browser_searches_matched": len(browser["results"]),
        "max_absolute_score_error": max_score_error, "score_tolerance": 1e-4,
        "checks": checked, "node_cold_load_ms": browser["load_ms"],
        "node_process_memory_bytes": browser["memory"],
        "timing_scope": "Local Node process; excludes network, source-URL loading and UI. Not a service guarantee.",
        "fields_in_export": metadata["fields_included"],
        "cooking_instructions_in_export": False, "quality_benchmark": False, "public_upload": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("model/data/recipes/recipe_ids.npz"))
    parser.add_argument("--catalog", type=Path, default=Path("model/data/recipes/recipe_search.sqlite"))
    parser.add_argument("--report", type=Path, required=True, help="new aggregate report path")
    parser.add_argument("--ipv4", action="store_true")
    args = parser.parse_args()
    if args.report.exists():
        parser.error("choose a new verification report; existing evidence is not overwritten")
    report = verify(args.index.resolve(), args.corpus.resolve(), args.catalog.resolve(),
                    Path(__file__).resolve().parents[2], ipv4=args.ipv4)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({name: report[name] for name in (
        "records_compared", "ingredient_slots_compared", "source_urls_checked", "card_links_rederived",
        "link_statuses", "recipe_titles_compared",
        "ingredient_lines_compared", "python_browser_searches_matched", "max_absolute_score_error",
        "public_upload")}, indent=2))


if __name__ == "__main__":
    main()
