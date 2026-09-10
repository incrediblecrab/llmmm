"""Build a public-sample browser demo without opening the private recipe corpus."""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
import torch

from ._hashing import file_sha256
from .data.recipe_catalog import _SCHEMA
from .recipe_ranker import (
    CONFIG_FILENAME, FEATURE_NAMES, FEATURE_VERSION, WEIGHTS_FILENAME,
    RecipeRankingPolicy, _json_object, candidate_features, heuristic_scores,
)
from .recipe_search import RecipeFinder, RecipeQuery

MODEL_REPOSITORY = "incrediblecrab/llmmm-recipes"
DATASET_REPOSITORY = "incrediblecrab/llmmm-recipe-sample"
SPACE_REPOSITORY = "incrediblecrab/llmmm-recipes-demo"
WEB_FILES = ("index.html", "styles.css", "app.js", "ranker.js", "search.js")
DATASET_FILES = ("README.md", "recipes.jsonl", "sources.json")
SPACE_FILES = (*WEB_FILES, "catalog.json", "policy.json", "manifest.json", "README.md")
SOURCE_FILES = (
    *(f"model/demo/{name}" for name in WEB_FILES),
    "model/demo/test/python_bridge.js",
    "model/demo/e2e/demo.spec.js",
    "model/demo/playwright.config.js",
    "model/demo/package.json",
    "model/demo/package-lock.json",
    "model/ingredient_model/recipe_demo.py",
    "model/ingredient_model/recipe_ranker.py",
    "model/ingredient_model/recipe_search.py",
    "model/ingredient_model/data/recipe_catalog.py",
    "model/ingredient_model/_hashing.py",
    "model/scripts/build_recipe_demo.py",
    "model/scripts/publish_recipe_demo.py",
    *(f"model/demo_data/{name}" for name in DATASET_FILES),
    "model/scripts/curate_recipe_demo.py",
    "model/results/huggingface_recipe_search_release.json",
)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _string_list(value: object, *, nonempty: bool = False) -> bool:
    return (isinstance(value, list) and (bool(value) or not nonempty)
            and all(isinstance(item, str) and item.strip() for item in value))


def _reject_json_constant(value: str):
    raise ValueError(f"non-finite JSON constant in public sample: {value}")


def _https_url(value: object, *, wikibooks: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError("source links must be HTTPS strings")
    parts = urlsplit(value)
    if (parts.scheme != "https" or not parts.hostname or parts.username or parts.password
            or (wikibooks and parts.hostname != "en.wikibooks.org")):
        raise ValueError(f"unexpected source URL: {value!r}")
    return value


def load_sample(path: Path, vocabulary: list[str]) -> list[dict]:
    """Accept only the bounded, explicitly attributed public demonstration schema."""
    if path.stat().st_size > 20 * 1024 * 1024:
        raise ValueError("a public demo input must be at most 20 MiB, not a source-corpus dump")
    known = set(vocabulary)
    rows, seen = [], set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line_number > 1000:
                raise ValueError("a demo catalog must contain at most 1000 records")
            row = json.loads(line, object_pairs_hook=_json_object, parse_constant=_reject_json_constant)
            if not isinstance(row, dict):
                raise ValueError(f"public recipe row {line_number} must be an object")
            for name in ("id", "title", "language", "source_title", "retrieved_at",
                         "attribution", "changes", "license"):
                if not isinstance(row.get(name), str) or not row[name].strip():
                    raise ValueError(f"row {line_number}: missing {name}")
            if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", row["id"]) or row["id"] in seen:
                raise ValueError(f"row {line_number}: recipe IDs must be unique ASCII slugs")
            if row["language"] != "en":
                raise ValueError("this public sample release contains only explicitly English recipes")
            for name in ("canonical_ingredients", "raw_ingredients", "instructions"):
                if not _string_list(row.get(name), nonempty=True):
                    raise ValueError(f"{row['id']}: {name} must contain nonempty strings")
            if not _string_list(row.get("unmapped_ingredients")):
                raise ValueError(f"{row['id']}: unmapped ingredients must be explicitly recorded")
            canonical = row["canonical_ingredients"]
            if len(set(canonical)) != len(canonical) or not set(canonical) <= known:
                raise ValueError(f"{row['id']}: ingredient names must be unique and in the public vocabulary")
            if row["license"].lower() not in ("cc-by-sa-3.0", "cc-by-sa-4.0"):
                raise ValueError(f"{row['id']}: unexpected recipe-text license")
            for name, evidence in (("total_minutes", "time_evidence"), ("servings", "servings_evidence")):
                if name not in row:
                    raise ValueError(f"{row['id']}: unknown {name} must be explicit null")
                value = row[name]
                if value is not None:
                    if (type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                            or not isinstance(row.get(evidence), str) or not row[evidence].strip()):
                        raise ValueError(f"{row['id']}: positive source {name} needs explicit evidence")
            _https_url(row["source_url"], wikibooks=True)
            _https_url(row["attribution_url"], wikibooks=True)
            revision = str(row.get("source_revision_id", ""))
            pinned = parse_qs(urlsplit(row["source_url"]).query).get("oldid", [])
            if not revision.isascii() or not revision.isdigit() or pinned != [revision]:
                raise ValueError(f"{row['id']}: source URL must pin its exact Wikibooks revision")
            datetime.fromisoformat(row["retrieved_at"].replace("Z", "+00:00"))
            seen.add(row["id"])
            rows.append(row)
    if not rows:
        raise ValueError("the public sample is empty")
    return rows


def make_catalog(rows: list[dict], vocabulary: list[str]) -> dict:
    index = {name: value for value, name in enumerate(vocabulary)}
    frequency = [0] * len(vocabulary)
    for row in rows:
        for name in row["canonical_ingredients"]:
            frequency[index[name]] += 1
    return {
        "schema_version": 1, "statistics_scope": "public-sample",
        "vocabulary": vocabulary, "ingredient_frequency": frequency,
        "n_recipes": len(rows), "recipes": rows,
    }


def verify_sample_provenance(repository: Path) -> dict:
    result = subprocess.run(
        [sys.executable, str(repository / "model/scripts/curate_recipe_demo.py"), "--verify"],
        cwd=repository, text=True, capture_output=True, timeout=60)
    if result.returncode:
        raise ValueError(f"public sample provenance verification failed:\n{result.stdout}\n{result.stderr}")
    return json.loads((repository / "model/demo_data/sources.json").read_text(encoding="utf-8"))


def browser_policy(policy: RecipeRankingPolicy, source: dict) -> dict:
    return {
        "schema_version": 1, "model_type": "recipe-ranking-mlp-browser",
        "feature_version": FEATURE_VERSION, "feature_names": list(FEATURE_NAMES),
        "activation": "tanh", "hidden_dim": policy.hidden_dim,
        "time_features_enabled": policy.time_features_enabled,
        "learned_parameter_count": sum(value.numel() for value in policy.parameters()),
        "source": source,
        "tensors": {name: value.detach().cpu().tolist() for name, value in policy.state_dict().items()},
    }


def load_public_policy(repository: Path, *, ipv4: bool = False) -> tuple[list[str], RecipeRankingPolicy, dict]:
    from huggingface_hub import hf_hub_download

    if ipv4:
        import httpx
        from huggingface_hub import set_client_factory

        set_client_factory(lambda: httpx.Client(
            transport=httpx.HTTPTransport(local_address="0.0.0.0"),
            follow_redirects=True, timeout=60))
    receipt = json.loads((repository / "model/results/huggingface_recipe_search_release.json").read_text())
    if receipt["repo_id"] != MODEL_REPOSITORY or receipt["selected_policy"] != "supervised":
        raise ValueError("the demo requires the recorded public supervised release")
    if not re.fullmatch(r"[0-9a-f]{40}", receipt["revision"]):
        raise ValueError("the model must be pinned to an immutable commit")
    files = {
        "config.json": receipt["preserved_native_files"]["config.json"],
        **{f"recipe_policies/supervised/{name}": receipt["files"][f"recipe_policies/supervised/{name}"]
           for name in (CONFIG_FILENAME, WEIGHTS_FILENAME)},
    }
    local = {}
    for name, expected in files.items():
        path = Path(hf_hub_download(
            MODEL_REPOSITORY, name, revision=receipt["revision"], token=False,
            cache_dir=repository / "model/data/hf_cache"))
        if path.stat().st_size != expected["bytes"] or file_sha256(path) != expected["sha256"]:
            raise ValueError(f"public model file {name} differs from its committed release receipt")
        local[name] = path
    vocabulary = json.loads(local["config.json"].read_text())["vocabulary"]
    if (not _string_list(vocabulary, nonempty=True) or len(set(vocabulary)) != len(vocabulary)
            or len(vocabulary) > 65_536):
        raise ValueError("public model vocabulary is invalid")
    model = RecipeRankingPolicy.load(local[f"recipe_policies/supervised/{CONFIG_FILENAME}"].parent)
    return vocabulary, model, {
        "repository": MODEL_REPOSITORY, "revision": receipt["revision"], "tag": receipt["tag"],
        "policy": "supervised", "files": files,
    }


def example_pantries(catalog: dict) -> list[dict]:
    rows = sorted(catalog["recipes"], key=lambda row: (
        row["total_minutes"] is None, row["total_minutes"] or math.inf,
        len(row["canonical_ingredients"]), row["id"]))
    base = rows[:min(3, len(rows))]
    pantry = sorted(set().union(*(set(row["canonical_ingredients"]) for row in base)))
    examples = [{
        "label": "A stocked pantry", "available_ingredients": pantry,
        "max_total_minutes": None, "max_missing": 2,
    }]
    for row in rows[:2]:
        ingredients = row["canonical_ingredients"]
        examples.append({
            "label": row["title"], "available_ingredients": ingredients[:-1] if len(ingredients) > 2 else ingredients,
            "max_total_minutes": row["total_minutes"], "max_missing": 1,
        })
    return examples


def _reference_catalog(path: Path, catalog: dict) -> None:
    index = {name: position for position, name in enumerate(catalog["vocabulary"])}
    metadata = {name: catalog[name] for name in ("n_recipes", "vocabulary", "ingredient_frequency")}
    metadata.update(schema_version=1, partial=False, catalog_scope="public-sample")
    with sqlite3.connect(path) as connection:
        connection.executescript(_SCHEMA)
        connection.executemany("INSERT INTO metadata VALUES (?,?)",
                               [(name, json.dumps(value)) for name, value in metadata.items()])
        for position, row in enumerate(catalog["recipes"]):
            ids = sorted(index[name] for name in row["canonical_ingredients"])
            values = {
                "id": position, "source": "public_sample_wikibooks", "language": row["language"],
                "title": row["title"], "url": row["source_url"],
                "ingredient_ids": np.asarray(ids, dtype="<u2").tobytes(),
                "raw_ingredients": "\x1f".join(row["raw_ingredients"]),
                "steps": "\x1f".join(row["instructions"]), "ingredient_quantities": "[]",
                "quantity_status": "", "text_status": "", "total_minutes": row["total_minutes"],
                "servings": row["servings"], "time_status": "unknown" if row["total_minutes"] is None else "source_total",
                "servings_status": "unknown" if row["servings"] is None else "source_servings",
                "metadata_status": "public_sample_source", "metadata_match_count": 1,
            }
            connection.execute(
                f"INSERT INTO recipes ({','.join(values)}) VALUES ({','.join('?' for _ in values)})",
                list(values.values()))
            connection.execute("INSERT INTO recipe_fts(rowid,ingredient_tokens) VALUES (?,?)",
                               (position, " ".join(f"i{value}" for value in ids)))


def verify_browser(repository: Path, catalog: dict, policy: RecipeRankingPolicy,
                   exported_policy: dict, *, seed: int = 42, cases: int = 128) -> dict:
    """Compare the browser kernel and complete searches with the existing Python implementation."""
    rng = np.random.default_rng(seed)
    vocabulary = catalog["vocabulary"]
    frequency = np.asarray(catalog["ingredient_frequency"])
    feature_cases, expected = [], []
    times = [None, 0, 0.25, 10, 30, 120, 10_080, 20_000]
    for number in range(cases):
        available = rng.choice(len(vocabulary), size=min(1 + number % 100, len(vocabulary)), replace=False).tolist()
        candidates = [rng.choice(len(vocabulary), size=min(1 + (number + offset) % 30, len(vocabulary)),
                                 replace=False).tolist() for offset in range(8)]
        if number % 3 == 0:
            candidates[0] = available + available[:1]
        minutes = [times[(number + offset) % len(times)] for offset in range(8)]
        budget = [None, 0.5, 15, 30, 120, 10_080][number % 6]
        features = candidate_features(available, candidates, ingredient_frequency=frequency,
                                      n_recipes=catalog["n_recipes"], total_minutes=minutes,
                                      max_total_minutes=budget)
        feature_cases.append({"available": available, "candidates": candidates,
                              "total_minutes": minutes, "max_total_minutes": budget})
        expected.append((features, policy.score(features), heuristic_scores(features)))
    queries = []
    for row in catalog["recipes"]:
        ingredients = row["canonical_ingredients"]
        queries.extend([
            {"available_ingredients": ingredients, "max_missing": 0, "top_k": 100},
            {"available_ingredients": ingredients, "max_missing": 2, "max_total_minutes": row["total_minutes"],
             "min_servings": row["servings"], "top_k": 100},
            {"available_ingredients": ingredients, "must_use": ingredients[:1], "max_missing": 2, "top_k": 100},
            {"available_ingredients": ingredients, "max_total_minutes": 0, "top_k": 100},
        ])
        if len(ingredients) > 1:
            queries.append({"available_ingredients": ingredients, "exclude": ingredients[-1:],
                            "max_missing": None, "top_k": 100})
            queries.append({"available_ingredients": ingredients[:-1], "max_missing": 1, "top_k": 100})
    for example in example_pantries(catalog):
        queries.append({name: value for name, value in example.items() if name != "label"} | {"top_k": 100})
    result = subprocess.run(
        ["node", str(repository / "model/demo/test/python_bridge.js")],
        input=json.dumps({"catalog": catalog, "policy": exported_policy,
                          "feature_cases": feature_cases, "queries": queries}, allow_nan=False),
        text=True, capture_output=True, check=True, timeout=60)
    actual = json.loads(result.stdout)
    if len(actual["features"]) != len(expected) or len(actual["searches"]) != len(queries):
        raise ValueError("the browser verifier returned an incomplete result")
    feature_error = score_error = baseline_error = 0.0
    for browser, (features, scores, baseline) in zip(actual["features"], expected, strict=True):
        for observed, reference, tolerance in (
                (browser["features"], features, 1e-7), (browser["learned"], scores, 1e-4),
                (browser["heuristic"], baseline, 1e-6)):
            np.testing.assert_allclose(observed, reference, atol=tolerance, rtol=0)
        feature_error = max(feature_error, float(np.max(np.abs(np.asarray(browser["features"]) - features))))
        score_error = max(score_error, float(np.max(np.abs(np.asarray(browser["learned"]) - scores))))
        baseline_error = max(baseline_error, float(np.max(np.abs(np.asarray(browser["heuristic"]) - baseline))))
    matched_queries = 0
    with tempfile.TemporaryDirectory(prefix="llmmm-public-reference-") as temporary:
        path = Path(temporary) / "public_sample.sqlite"
        _reference_catalog(path, catalog)
        for ranking in ("learned", "heuristic"):
            finder = RecipeFinder(path, policy=policy, ranking=ranking,
                                  max_candidates=1000, scan_limit=1000, timeout_seconds=5)
            for query, browser in zip(queries, actual["searches"], strict=True):
                reference = finder.search(RecipeQuery(**query))
                observed = browser[ranking]
                expected_ids = [catalog["recipes"][recipe.recipe_id]["id"] for recipe in reference.recipes]
                observed_ids = [recipe["id"] for recipe in observed["matches"]]
                if expected_ids != observed_ids or observed["feasible_count"] != reference.feasible_candidates:
                    raise ValueError(f"browser/Python search order or feasibility differs: {query!r}")
                if reference.retrieval_truncated:
                    raise ValueError("reference sample search unexpectedly hit a retrieval budget")
                for match, recipe in zip(observed["matches"], reference.recipes, strict=True):
                    if (match["matched_ingredients"] != list(recipe.matched_ingredients)
                            or match["missing_ingredients"] != list(recipe.missing_ingredients)):
                        raise ValueError("browser/Python ingredient result fields differ")
                    if abs(match["score"] - recipe.score) > 1e-4:
                        raise ValueError("browser/Python search score differs beyond the declared tolerance")
                matched_queries += 1
    return {
        "seed": seed, "synthetic_feature_contexts": cases, "feature_rows": cases * 8,
        "max_absolute_feature_error": feature_error, "feature_tolerance": 1e-7,
        "max_absolute_score_error": score_error, "score_tolerance": 1e-4,
        "max_absolute_baseline_error": baseline_error, "baseline_tolerance": 1e-6,
        "python_finder_searches_matched": matched_queries,
        "exact_match_fields": ["ordered_recipe_ids", "feasible_count", "matched_ingredients", "missing_ingredients"],
        "quality_benchmark": False, "private_corpus_used": False,
    }


def verify_source_revision(repository: Path, revision: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("source revision must be a full Git commit")
    identities = {}
    for name in SOURCE_FILES:
        expected = subprocess.run(["git", "show", f"{revision}:{name}"], cwd=repository,
                                  check=True, capture_output=True).stdout
        current = (repository / name).read_bytes()
        if expected != current:
            raise ValueError(f"{name} differs from the pinned source commit")
        identities[name] = hashlib.sha256(current).hexdigest()
    return identities


def space_card(catalog: dict, provenance: dict) -> str:
    licenses = ", ".join(sorted({row["license"].upper() for row in catalog["recipes"]}))
    return f"""---
title: llmmm Recipe Finder
colorFrom: green
colorTo: yellow
sdk: static
app_file: index.html
models:
  - {MODEL_REPOSITORY}
datasets:
  - {DATASET_REPOSITORY}
---

# llmmm recipe finder

Try the released supervised ranker on **{catalog['n_recipes']} openly licensed sample recipes**.
Enter pantry ingredients, a total-time limit and optional constraints. The app filters
the complete small sample before ranking. It does not generate recipes.

Everything runs in the browser, including the trained model's forward pass.
No GPU, backend, private corpus, account or API key is needed to use the demo.
Pantry inputs are not sent to a server or stored between visits.

The sample is separately sourced from Wikibooks, not extracted from the private
training corpus. It is not a held-out benchmark, and the full-catalog quality
measurements do not apply to it. Document frequencies are recomputed from this
sample; the full published vocabulary and supervised weights are retained.
Time and servings are source reports, not independently measured values. Unknown
metadata cannot pass a corresponding limit. Canonical exclusions are not an
allergy-safety check, and servings do not scale quantities.

Recipe text is licensed under {licenses}; see the [dataset](https://huggingface.co/datasets/{DATASET_REPOSITORY})
and individual recipes for source revisions, contributor-history links and changes.
The model weights retain their [separate terms](https://huggingface.co/{MODEL_REPOSITORY}/blob/{provenance['model_revision']}/recipe_release_policy.json);
the recipe-text license does not relicense the weights or project code.

- Model commit: `{provenance['model_revision']}` (supervised).
- Dataset commit: `{provenance['dataset_revision'] or 'local preview; not published'}`.
- Source commit: `{provenance['source_revision'] or 'local preview; not committed'}`.
- [Source and reproduction instructions](https://github.com/incrediblecrab/llmmm).

`manifest.json` records the exact application assets and their SHA256 digests.
The app verifies the catalog and policy digests before enabling search. The Hub
renders this README as HTML; its raw source hash is recorded separately under
`documentation_files` and verified through the pinned Hub repository.
"""


def add_demo_links(card: str) -> str:
    start, end = "<!-- PUBLIC-DEMO:START -->", "<!-- PUBLIC-DEMO:END -->"
    section = f"""{start}
## Try the public demo

[Open the browser demo](https://huggingface.co/spaces/{SPACE_REPOSITORY}) or
[download the public recipe sample](https://huggingface.co/datasets/{DATASET_REPOSITORY}).
The demo runs the released supervised ranker on a small, separately sourced
Wikibooks catalog. It needs no private dataset, API key or login.

This is not the full training catalog or a new quality benchmark. The demo uses
sample-specific ingredient frequencies; the full-catalog measurements below do
not apply to it. Recipe-text licenses, source revisions and attribution are
recorded with the sample. The model weights retain their existing terms.
{end}
"""
    if start in card or end in card:
        if card.count(start) != 1 or card.count(end) != 1 or card.index(start) > card.index(end):
            raise ValueError("malformed existing demo-link section")
        before, remaining = card.split(start, 1)
        _, after = remaining.split(end, 1)
        return before + section.rstrip() + after
    heading = "## Finding recipes"
    if card.count(heading) != 1:
        raise ValueError("the model card does not have its expected recipe-search section")
    return card.replace(heading, section + "\n" + heading, 1)


def build_demo(repository: Path, output: Path, *, source_revision: str | None = None,
               dataset_revision: str | None = None, ipv4: bool = False) -> dict:
    if output.exists():
        raise FileExistsError(f"{output}: demo builds never overwrite an existing directory")
    if dataset_revision is not None and not re.fullmatch(r"[0-9a-f]{40}", dataset_revision):
        raise ValueError("dataset revision must be an immutable Hub commit")
    source_files = verify_source_revision(repository, source_revision) if source_revision else None
    sources = verify_sample_provenance(repository)
    torch.set_num_threads(4)
    vocabulary, policy, model_source = load_public_policy(repository, ipv4=ipv4)
    sample_path = repository / "model/demo_data/recipes.jsonl"
    rows = load_sample(sample_path, vocabulary)
    by_id = {source["id"]: source for source in sources["sources"]}
    if len(by_id) != len(rows) or set(by_id) != {row["id"] for row in rows}:
        raise ValueError("the source provenance does not cover this exact public sample")
    rows = [{**row, "source_limitations": by_id[row["id"]]["source_limitations"]} for row in rows]
    catalog = make_catalog(rows, vocabulary)
    exported = browser_policy(policy, model_source)
    verification = verify_browser(repository, catalog, policy, exported)
    provenance = {
        "model_repository": MODEL_REPOSITORY, "model_revision": model_source["revision"],
        "dataset_repository": DATASET_REPOSITORY, "dataset_revision": dataset_revision,
        "source_revision": source_revision, "sample_sha256": file_sha256(sample_path),
        "source_manifest_sha256": file_sha256(repository / "model/demo_data/sources.json"),
    }
    bundle = {"catalog": catalog, "examples": example_pantries(catalog), "provenance": provenance}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".llmmm-demo-build-", dir=output.parent) as temporary:
        staged = Path(temporary) / "space"
        staged.mkdir()
        for name in WEB_FILES:
            shutil.copyfile(repository / "model/demo" / name, staged / name)
        (staged / "catalog.json").write_bytes(_json_bytes(bundle))
        (staged / "policy.json").write_bytes(_json_bytes(exported))
        (staged / "README.md").write_text(space_card(catalog, provenance), encoding="utf-8")
        files = {name: {"bytes": (staged / name).stat().st_size, "sha256": file_sha256(staged / name)}
                 for name in (*WEB_FILES, "catalog.json", "policy.json")}
        manifest = {
            "schema_version": 1, "built_at": datetime.now(timezone.utc).isoformat(),
            "files": files, "provenance": provenance, "source_files_sha256": source_files,
            "documentation_files": {
                "README.md": {"bytes": (staged / "README.md").stat().st_size,
                              "sha256": file_sha256(staged / "README.md")},
            },
            "verification": verification,
            "catalog_summary": {
                "recipes": len(rows), "source_total_times": sum(row["total_minutes"] is not None for row in rows),
                "source_serving_counts": sum(row["servings"] is not None for row in rows),
                "recipes_with_unmapped_ingredients": sum(bool(row["unmapped_ingredients"]) for row in rows),
                "observed_ingredients": sum(count > 0 for count in catalog["ingredient_frequency"]),
                "vocabulary_size": len(vocabulary),
                "statistics_scope": "public-sample",
                "licenses": sorted({row["license"].lower() for row in rows}),
            },
        }
        (staged / "manifest.json").write_bytes(_json_bytes(manifest))
        if set(path.name for path in staged.iterdir()) != set(SPACE_FILES):
            raise ValueError("unexpected file in the static Space build")
        if source_revision:
            verify_source_revision(repository, source_revision)
        staged.rename(output)
    return manifest
