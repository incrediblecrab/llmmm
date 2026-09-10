"""Evaluate constrained retrieval on separately generated queries over a known catalog."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ingredient_model.config import PATHS
from ingredient_model._hashing import file_sha256
from ingredient_model.recipe_search import RecipeFinder, RecipeQuery
from train_recipe_ranker import pantry_signature, query_partition


def _query_cases(finder: RecipeFinder, count: int, seed: int,
                 partition: str) -> list[tuple[int, RecipeQuery]]:
    if partition not in ("validation", "test"):
        raise ValueError("live evaluation must use a held-out pantry partition")
    rng = np.random.default_rng(seed)
    cases = []
    seen = set()
    seen_pantries = set()
    with closing(sqlite3.connect(
            finder.catalog_path.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        for _ in range(max(200, count * 8)):
            selected = rng.integers(0, finder.n_recipes, size=512).tolist()
            placeholders = ",".join("?" for _ in selected)
            rows = connection.execute(
                f"SELECT id, ingredient_ids, total_minutes, servings, language, title, steps "
                f"FROM recipes WHERE id IN ({placeholders}) "
                "AND length(trim(title)) > 0 AND length(trim(steps)) > 0 "
                "AND coalesce(text_status, '') != 'unparsed_steps' ORDER BY id", selected).fetchall()
            rng.shuffle(rows)
            for row in rows:
                if (row["id"] in seen or not row["title"].strip()
                        or not any(step.strip() for step in row["steps"].split("\x1f"))):
                    continue
                ingredients = np.frombuffer(row["ingredient_ids"], dtype="<u2")
                if not 2 <= len(ingredients) <= 25:
                    continue
                timed = len(cases) % 2 == 0
                if timed and row["total_minutes"] is None:
                    continue
                hidden = min(int(rng.integers(1, 3)), len(ingredients) - 1)
                available = set(int(value) for value in rng.choice(
                    ingredients, len(ingredients) - hidden, replace=False))
                extra_count = int(rng.integers(0, 3))
                extras = [int(value) for value in rng.permutation(len(finder.vocabulary))
                          if value not in ingredients][:extra_count]
                available.update(extras)
                signature = pantry_signature(np.asarray(sorted(available), dtype=np.uint16))
                if query_partition(signature) != partition or signature in seen_pantries:
                    continue
                must_use = [int(rng.choice(sorted(available & set(ingredients))))]
                forbidden = [int(value) for value in rng.permutation(len(finder.vocabulary))
                             if value not in ingredients and value not in available][:1]
                query = RecipeQuery(
                    [finder.vocabulary[value] for value in sorted(available)],
                    must_use=[finder.vocabulary[value] for value in must_use],
                    exclude=[finder.vocabulary[value] for value in forbidden],
                    max_total_minutes=float(row["total_minutes"]) if timed else None,
                    max_missing=hidden,
                    min_servings=(min(2.0, float(row["servings"]))
                                  if row["servings"] is not None and row["servings"] >= 1 else None),
                    language=row["language"] if len(cases) % 3 == 0 else None,
                    top_k=10)
                seen.add(row["id"])
                seen_pantries.add(signature)
                cases.append((row["id"], query))
                if len(cases) == count:
                    return cases
    raise ValueError(f"only {len(cases)} eligible query targets found, expected {count}")


def _summary(rows: list[dict]) -> dict:
    ranks = np.array([row["rank"] for row in rows], dtype=np.float64)
    latency = np.array([row["latency_ms"] for row in rows])
    return {
        "queries": len(rows),
        "source_set_recall_at_1": float(np.mean(ranks <= 1)),
        "source_set_recall_at_5": float(np.mean(ranks <= 5)),
        "source_set_recall_at_10": float(np.mean(ranks <= 10)),
        "mrr_at_10": float(np.mean(1 / ranks)),
        "source_set_in_shortlist": float(np.mean([row["shortlist_hit"] for row in rows])),
        "empty_results": sum(row["empty"] for row in rows),
        "truncated_retrievals": sum(row["truncated"] for row in rows),
        "timeouts": sum(row["timeout"] for row in rows),
        "constraint_violations": sum(row["constraint_violations"] for row in rows),
        "latency_ms": {"median": float(np.median(latency)),
                       "p95": float(np.quantile(latency, 0.95)),
                       "maximum": float(latency.max())},
    }


def _paired_interval(left: list[dict], right: list[dict], seed: int) -> dict:
    if not left or len(left) != len(right):
        raise ValueError("paired comparisons require aligned nonempty query results")
    difference = np.array([
        float(a["rank"] <= 5) - float(b["rank"] <= 5)
        for a, b in zip(left, right)], dtype=np.float64)
    rng = np.random.default_rng(seed)
    bootstrap = np.array([
        difference[rng.integers(0, len(difference), len(difference))].mean()
        for _ in range(2000)])
    return {"difference_in_recall_at_5": float(difference.mean()),
            "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist()}


def _operational_pass(metrics: dict, timeout_seconds: float) -> bool:
    return (metrics["queries"] > 0 and metrics["constraint_violations"] == 0
            and metrics["timeouts"] == 0
            and metrics["empty_results"] / metrics["queries"] <= 0.05
            and metrics["latency_ms"]["maximum"] <= timeout_seconds * 1000)


def _case_fingerprint(cases: list[tuple[int, RecipeQuery]]) -> str:
    digest = hashlib.sha256()
    for target, query in cases:
        digest.update(json.dumps(
            [target, asdict(query)], sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _evaluate_cases(finders: dict[str, RecipeFinder], cases: list[tuple[int, RecipeQuery]],
                    flat: np.ndarray, offsets: np.ndarray, *, phase: str) -> dict[str, list[dict]]:
    data = {name: [] for name in finders}
    reference = finders["heuristic"]
    names = list(finders)
    for index, (target, query) in enumerate(cases):
        expected_ids = flat[offsets[target]:offsets[target + 1]]
        expected_names = {reference.vocabulary[int(value)] for value in expected_ids}
        expected_shortlist = None
        for shift in range(len(names)):
            name = names[(index + shift) % len(names)]
            tick = time.perf_counter()
            try:
                result = finders[name].search(query, include_candidate_ids=True)
            except TimeoutError:
                data[name].append({
                    "rank": float("inf"), "latency_ms": (time.perf_counter() - tick) * 1000,
                    "shortlist_hit": False, "empty": True, "truncated": True,
                    "timeout": True, "constraint_violations": 0})
                continue
            shortlist = result.candidate_recipe_ids
            if expected_shortlist is not None and shortlist != expected_shortlist:
                raise ValueError("ranking policies were given different retrieval shortlists")
            expected_shortlist = shortlist
            hit = any(np.array_equal(
                flat[offsets[item]:offsets[item + 1]], expected_ids) for item in shortlist or ())
            rank = next((position for position, recipe in enumerate(result.recipes, 1)
                         if set(recipe.ingredients) == expected_names), float("inf"))
            violations = 0
            for recipe in result.recipes:
                ingredients = set(recipe.ingredients)
                if (not set(query.must_use) <= ingredients or set(query.exclude) & ingredients
                        or len(ingredients - set(query.available_ingredients)) > query.max_missing
                        or (query.max_total_minutes is not None and (
                            recipe.total_minutes is None or recipe.total_minutes > query.max_total_minutes))
                        or (query.min_servings is not None and (
                            recipe.servings is None or recipe.servings < query.min_servings))
                        or (query.language is not None and recipe.language != query.language)
                        or not recipe.title.strip() or not recipe.steps
                        or not all(step.strip() for step in recipe.steps)):
                    violations += 1
            data[name].append({
                "rank": rank, "latency_ms": result.elapsed_ms, "shortlist_hit": hit,
                "empty": not result.recipes, "truncated": result.retrieval_truncated,
                "timeout": False, "constraint_violations": violations})
        if (index + 1) % 50 == 0:
            print(f"{phase}: {index + 1}/{len(cases)} live-catalog queries evaluated", flush=True)
    return data


def _select_policy(data: dict[str, list[dict]], *, timeout: float, seed: int) -> tuple[str, list[dict]]:
    selected = "heuristic"
    comparisons = []
    for name in list(data)[1:]:
        comparison = _paired_interval(data[name], data[selected], seed)
        operational = _operational_pass(_summary(data[name]), timeout)
        promoted = operational and comparison["ci95"][0] > 0
        comparisons.append({
            "candidate": name, "incumbent": selected, **comparison,
            "validation_operational_gate_passed": operational, "promoted": promoted})
        if promoted:
            selected = name
    return selected, comparisons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=PATHS.recipes / "recipe_search.sqlite")
    parser.add_argument("--corpus", type=Path, default=PATHS.recipes / "recipe_ids.npz")
    parser.add_argument("--policy", action="append", default=[], metavar="NAME=DIR")
    parser.add_argument("--queries", type=int, default=200, help="queries per validation/test partition")
    parser.add_argument("--validation-only", action="store_true",
                        help="debug retrieval without generating or scoring any test queries")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--max-candidates", type=int, default=2000)
    parser.add_argument("--scan-limit", type=int, default=50_000)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.queries < 10:
        parser.error("--queries must be at least 10 per partition")
    if args.out.exists():
        raise FileExistsError(f"{args.out}: choose a new evaluation report path")
    source = Path(__file__).resolve().parents[1]
    source_paths = {
        "code_sha256": Path(__file__),
        "serving_code_sha256": source / "ingredient_model" / "recipe_search.py",
        "metadata_index_code_sha256": source / "ingredient_model" / "data" / "recipe_search_metadata.py",
        "ingredient_index_code_sha256": source / "ingredient_model" / "recipe_ingredients.py",
        "ranker_code_sha256": source / "ingredient_model" / "recipe_ranker.py",
        "query_split_code_sha256": source / "scripts" / "train_recipe_ranker.py",
    }
    source_hashes = {name: file_sha256(path) for name, path in source_paths.items()}
    catalog_hash = file_sha256(args.catalog)
    options = dict(max_candidates=args.max_candidates, scan_limit=args.scan_limit,
                   timeout_seconds=args.timeout)
    startup_tick = time.perf_counter()
    finders = {"heuristic": RecipeFinder(
        args.catalog, ranking="heuristic", corpus_path=args.corpus, **options)}
    initialization_seconds = {"heuristic": time.perf_counter() - startup_tick}
    policy_fingerprints = {}
    policy_paths = {}
    for specification in args.policy:
        name, separator, path = specification.partition("=")
        if not separator or not name or name in finders:
            parser.error("--policy requires a unique NAME=DIR")
        policy_paths[name] = Path(path)
        policy_fingerprints[name] = {}
        for file in sorted(Path(path).iterdir()):
            if file.suffix in (".json", ".safetensors") and file.is_file():
                policy_fingerprints[name][file.name] = file_sha256(file)
        startup_tick = time.perf_counter()
        finders[name] = RecipeFinder.from_directory(
            path, catalog_path=args.catalog, corpus_path=args.corpus, **options)
        initialization_seconds[name] = time.perf_counter() - startup_tick
        if any(file_sha256(Path(path) / file) != expected
               for file, expected in policy_fingerprints[name].items()):
            raise ValueError("policy files changed while loading")
    reference = finders["heuristic"]
    order = ["heuristic", *[name for name in ("supervised", "reinforce") if name in finders],
             *sorted(set(finders) - {"heuristic", "supervised", "reinforce"})]
    finders = {name: finders[name] for name in order}
    validation_cases = _query_cases(reference, args.queries, args.seed, "validation")
    corpus_path = args.corpus
    if file_sha256(corpus_path) != reference.metadata["corpus_sha256"]:
        raise ValueError("canonical corpus bytes differ from the search catalog identity")
    with np.load(corpus_path, allow_pickle=False) as corpus:
        flat, offsets = corpus["flat"], corpus["offsets"]
    if len(offsets) - 1 != reference.n_recipes:
        raise ValueError("canonical corpus and search catalog have different populations")
    started = time.perf_counter()
    names = list(finders)
    validation_data = _evaluate_cases(
        finders, validation_cases, flat, offsets, phase="validation")
    validation = {name: _summary(rows) for name, rows in validation_data.items()}
    selected, selection_comparisons = _select_policy(
        validation_data, timeout=args.timeout, seed=args.seed)
    selection_fingerprint = hashlib.sha256(json.dumps({
        "selected": selected, "policy_files": policy_fingerprints, "retrieval": options,
        "validation_cases": _case_fingerprint(validation_cases),
    }, sort_keys=True).encode()).hexdigest()
    print(f"Selection frozen on validation: {selected}", flush=True)
    test_cases = ([] if args.validation_only else
                  _query_cases(reference, args.queries, args.seed + 1, "test"))
    test_data = ({} if args.validation_only else
                 _evaluate_cases(finders, test_cases, flat, offsets, phase="test"))
    test = {name: _summary(rows) for name, rows in test_data.items()}
    cases = [*validation_cases, *test_cases]
    if any(file_sha256(path) != source_hashes[name] for name, path in source_paths.items()):
        raise ValueError("source code changed during evaluation; rerun with frozen files")
    if file_sha256(args.catalog) != catalog_hash:
        raise ValueError("source catalog changed during evaluation")
    if any(file_sha256(policy_paths[name] / file) != expected
           for name, files in policy_fingerprints.items() for file, expected in files.items()):
        raise ValueError("policy files changed during evaluation")
    report = {
        "schema_version": 1,
        "protocol": "held-out pantry-hash synthetic queries over a known catalog; not human-preference evaluation",
        "scope": "source ingredient-set recovery with strict constraints; not cooking quality",
        "corpus_sha256": reference.metadata["corpus_sha256"],
        "catalog_sha256": catalog_hash,
        "catalog_recipes": reference.n_recipes,
        "seed": args.seed, "queries_per_partition": args.queries,
        "release_evaluation": (
            not args.validation_only and args.queries >= 200
            and set(names) == {"heuristic", "supervised", "reinforce"}
            and all(finder._metadata_index is not None and finder._ingredient_index is not None
                    for finder in finders.values())),
        "initialization_seconds": initialization_seconds,
        "request_latency_excludes_initialization": True,
        "case_fingerprints": {
            "validation": _case_fingerprint(validation_cases),
            "test": _case_fingerprint(test_cases) if test_cases else None,
        },
        "test_scored": not args.validation_only,
        "timed_query_fraction": sum(query.max_total_minutes is not None
                                    for _, query in cases) / len(cases),
        "query_targets_unique_by_row_within_partition": True,
        "query_pantries_unique": True,
        "training_query_partition_overlap": 0,
        "query_partition": {
            "method": "BLAKE2b-128 sorted unique uint16 little-endian pantry IDs; llmmm-query-v1",
            "bucket": "first eight digest bytes, little-endian, modulo ten",
            "training": list(range(8)), "validation": 8, "test": 9,
        },
        "target_eligibility": {
            "ingredient_count_range": [2, 25], "nonempty_title_and_steps": True,
            "half_require_known_source_total_time": True,
        },
        "recipe_families_disjoint_from_training": False,
        "policy_files_sha256": policy_fingerprints,
        "retrieval": options,
        "validation": validation, "test": test,
        "selected_on_validation": selected,
        "selection_frozen_before_test_scoring": True,
        "selection_fingerprint": selection_fingerprint,
        "selection_rule": "start with heuristic; promote only on positive lower paired validation recall@5 CI95 and a passed validation operational gate",
        "selection_comparisons": selection_comparisons,
        "paired_test_against_heuristic": {
            name: _paired_interval(rows, test_data["heuristic"], args.seed)
            for name, rows in test_data.items() if name != "heuristic"
        },
        "operational_limits": {
            "max_timeouts": 0, "max_constraint_violations": 0,
            "max_empty_fraction_on_known_feasible_queries": 0.05,
            "max_request_seconds": args.timeout,
        },
        "operational_gate_passed": (_operational_pass(validation[selected], args.timeout)
                                    and (args.validation_only or
                                         _operational_pass(test[selected], args.timeout))),
        "duration_s": time.perf_counter() - started,
        **source_hashes,
        "limitations": [
            "Synthetic source recovery is a proxy, not observed user satisfaction.",
            "Catalog items are known; the ingredient predictor was trained on all canonical records.",
            "Validation/test pantry hashes are excluded from policy training; recipe families are not held out.",
            "Equivalent ingredient sets count as positives even if instructions differ.",
            "Recall includes retrieval misses; source_set_in_shortlist measures the retrieval ceiling.",
            "Recipe times are source-reported rather than independently timed cooking measurements.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    phase = "validation" if args.validation_only else "test"
    print(json.dumps({"selected": selected, phase: report[phase][selected],
                      "operational_gate_passed": report["operational_gate_passed"]}, indent=2))
    return 0 if report["operational_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
