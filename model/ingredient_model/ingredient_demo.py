"""Assemble a local browser preview of the complete ingredient-only catalog."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ._hashing import file_sha256
from .data.recipe_search_metadata import _publish_directory
from .ingredient_catalog import TEXT_MANIFEST, load_ingredient_catalog, load_text_shards, read_text_shard
from .ingredient_dataset import _read_url_shard
from .recipe_demo import (
    DATASET_REPOSITORY, MODEL_REPOSITORY, SOURCE_FILES, SPACE_REPOSITORY,
    WEB_FILES, _json_bytes, browser_policy, load_public_policy, verify_source_revision,
)
from .recipe_ranker import RecipeRankingPolicy

INGREDIENT_DATASET_REPOSITORY = "incrediblecrab/llmmm-recipe-ingredients"
INGREDIENT_SOURCE_FILES = tuple(dict.fromkeys((
    *SOURCE_FILES,
    "model/ingredient_model/ingredient_catalog.py",
    "model/ingredient_model/ingredient_dataset.py",
    "model/ingredient_model/ingredient_demo.py",
    "model/ingredient_model/recipe_ingredients.py",
    "model/ingredient_model/data/recipe_search_metadata.py",
    "model/demo/test/full_catalog_bridge.js",
    "model/demo/e2e/ingredient-demo.spec.js",
    "model/demo/ingredient-playwright.config.js",
    "model/scripts/build_ingredient_catalog.py",
    "model/scripts/build_ingredient_dataset.py",
    "model/scripts/build_ingredient_demo.py",
    "model/scripts/verify_ingredient_catalog.py",
    "model/scripts/publish_ingredient_demo.py",
    "model/results/ingredient_catalog_verification.json",
    "model/results/ingredient_catalog_publication_scope.json",
)))
EXAMPLE_PANTRIES = (
    {"label": "Tomato and basil", "available_ingredients": ["tomato", "basil", "olive_oil", "salt", "garlic"],
     "max_total_minutes": None, "max_missing": 2},
    {"label": "Chicken, rice, broccoli", "available_ingredients": ["chicken", "rice", "broccoli"],
     "max_total_minutes": 30, "max_missing": 2},
    {"label": "Eggs and potatoes", "available_ingredients": ["egg", "potato", "onion", "salt", "oil"],
     "max_total_minutes": 30, "max_missing": 2},
)
MAX_CANDIDATES = 2000


def load_ingredient_policy(
    repository: Path, metadata: dict, *, ipv4: bool = False,
) -> tuple[list[str], RecipeRankingPolicy, dict]:
    from huggingface_hub import hf_hub_download

    vocabulary, policy, source = load_public_policy(repository, ipv4=ipv4)
    receipt = json.loads((repository / "model/results/huggingface_recipe_search_release.json").read_text())
    expected = receipt["files"]["recipe_search_config.json"]
    path = Path(hf_hub_download(
        MODEL_REPOSITORY, "recipe_search_config.json", revision=source["revision"],
        token=False, cache_dir=repository / "model/data/hf_cache"))
    if path.stat().st_size != expected["bytes"] or file_sha256(path) != expected["sha256"]:
        raise ValueError("the released recipe-search configuration failed its integrity check")
    configuration = json.loads(path.read_text())
    if (configuration["schema_version"] != 1 or configuration["repo_id"] != MODEL_REPOSITORY
            or configuration["selected_policy"] != "supervised"
            or metadata["vocabulary"] != vocabulary
            or metadata["identity"]["corpus_sha256"] != configuration["corpus_sha256"]
            or metadata["identity"]["catalog_sha256"] != configuration["evaluated_catalog_sha256"]):
        raise ValueError("ingredient index vocabulary, corpus or catalog differs from the released search model")
    return vocabulary, policy, {
        **source, "files": {**source["files"], "recipe_search_config.json": expected},
        "corpus_sha256": configuration["corpus_sha256"],
        "catalog_sha256": configuration["evaluated_catalog_sha256"],
    }


def example_query(example: dict) -> dict:
    """The query app.js sends when an example pantry is chosen with the default controls."""
    return {
        "available_ingredients": list(example["available_ingredients"]), "must_use": [], "exclude": [],
        "max_total_minutes": example["max_total_minutes"], "min_servings": None,
        "max_missing": example["max_missing"], "top_k": 5, "require_source_url": True,
    }


def precompute_examples(repository: Path, index_directory: Path, metadata: dict, arrays: dict,
                        text_records: list[dict], exported_policy: dict) -> list[dict]:
    """Rank each example pantry with the browser's own search code, then attach its card text and link."""
    queries = [example_query(example) for example in EXAMPLE_PANTRIES]
    process = subprocess.run(
        ["node", str(repository / "model/demo/test/full_catalog_bridge.js")],
        input=json.dumps({"directory": str(index_directory), "policy": exported_policy,
                          "queries": queries, "max_candidates": MAX_CANDIDATES}),
        text=True, capture_output=True, check=True, timeout=600)
    browser = json.loads(process.stdout)
    if browser["records"] != metadata["n_recipes"] or len(browser["results"]) != 2 * len(queries):
        raise ValueError("the example searches did not cover the complete index")
    url_size, text_size = metadata["rows_per_url_shard"], metadata["rows_per_text_shard"]
    urls, texts = {}, {}
    examples = []
    for position, (example, query) in enumerate(zip(EXAMPLE_PANTRIES, queries)):
        result = browser["results"][2 * position]
        if (result["query"] != query or result["ranking"] != "learned"
                or result["scanned"] != metadata["n_recipes"] or not result["matches"]):
            raise ValueError(f"{example['label']}: the precomputed search is incomplete")
        matches = []
        for match in result["matches"]:
            recipe_id = match["id"]
            if recipe_id // url_size not in urls:
                urls[recipe_id // url_size] = _read_url_shard(index_directory, metadata["url_shards"][recipe_id // url_size])
            if recipe_id // text_size not in texts:
                texts[recipe_id // text_size] = read_text_shard(index_directory, text_records[recipe_id // text_size])
            url = urls[recipe_id // url_size][recipe_id % url_size]
            titles, lines = texts[recipe_id // text_size]
            if url is None or not arrays["has_source_url"][recipe_id]:
                raise ValueError(f"{example['label']}: a link-only result has no source link")
            matches.append({**match, "source_url": url, "title": titles[recipe_id % text_size],
                            "ingredient_lines": lines[recipe_id % text_size]})
        summary = {name: result[name] for name in (
            "ranking", "feasible_count", "scanned", "candidates_scored", "retrieval_truncated", "shortlist_method")}
        examples.append({**example, "results": {**summary, "matches": matches}})
    return examples


def ingredient_space_card(metadata: dict, provenance: dict) -> str:
    return f"""---
title: llmmm Recipe Finder
colorFrom: green
colorTo: yellow
sdk: static
app_file: index.html
models:
  - {MODEL_REPOSITORY}
datasets:
  - {INGREDIENT_DATASET_REPOSITORY}
  - {DATASET_REPOSITORY}
---

# llmmm recipe finder

Search **{metadata['n_recipes']:,} canonical ingredient records** with pantry ingredients, source-reported time and serving limits, required/excluded ingredients, and a missing-item allowance. The same independently trained supervised weights from [llmmm-recipes](https://huggingface.co/{MODEL_REPOSITORY}) rank the results.

This is retrieval, not recipe generation. Result cards show each recipe's recorded title and original ingredient lines, quantities included: **{metadata['coverage']['recipe_titles']:,} records have titles** and **{metadata['coverage']['ingredient_line_records']:,} have ingredient lines**. Cooking instructions, descriptions and photos are not copied; open the recorded source link for the steps. There are **{metadata['coverage']['url_statuses'].get('source_url', 0):,} records with links**; the link-only filter is on by default and can be disabled. A record is not necessarily a unique or complete recipe.

The example pantries show precomputed results immediately. Loading the **{metadata['bytes']['initial_compressed_download'] / 1024**2:.1f} MiB** index is optional; search and model inference then run in your browser, and recipe text and source links for the results shown download as needed. This is a free Static Space, with no server-side model, paid hardware, API key or per-search inference fee. Pantry inputs are not sent to a server or stored between visits.

All records are checked against the hard constraints. The baseline retains at most 2,000 feasible records for learned ranking; the app discloses truncation. Those results are not guaranteed global learned top-k. The private finder's recovery scores do not evaluate this browser retrieval pipeline.

Missing time/serving metadata cannot pass its corresponding constraint. Values are source reports, not independently measured. Canonical exclusions are not an allergy-safety check; serving limits do not scale quantities.

- [Ingredient dataset and source provenance](https://huggingface.co/datasets/{INGREDIENT_DATASET_REPOSITORY})
- [Separate twelve-recipe Wikibooks sample, with complete instructions](https://huggingface.co/datasets/{DATASET_REPOSITORY})
- [Source and reproduction](https://github.com/incrediblecrab/llmmm/tree/{provenance['source_revision'] or 'main'}/model/demo)
- Model revision: `{provenance['model_revision']}`.
- Dataset revision: `{provenance['dataset_revision'] or 'local preview'}`.
- Source revision: `{provenance['source_revision'] or 'local preview'}`.

The maintainer confirmed permission to publish this extract, including the recorded titles and ingredient lines. Cooking instructions and other recipe-page content stay on the source sites under their own terms, and the model weights keep their separate terms. `manifest.json` pins the application assets and dataset index by SHA256.
"""


def add_ingredient_demo_links(card: str) -> str:
    start, end = "<!-- PUBLIC-DEMO:START -->", "<!-- PUBLIC-DEMO:END -->"
    if card.count(start) != 1 or card.count(end) != 1 or card.index(start) > card.index(end):
        raise ValueError("the model card must contain its existing single demo-link section")
    section = f"""{start}
## Try the public demo

[Open the browser demo](https://huggingface.co/spaces/{SPACE_REPOSITORY}) or [download all 4,653,430 ingredient records](https://huggingface.co/datasets/{INGREDIENT_DATASET_REPOSITORY}). The demo uses the released supervised weights and full-corpus ingredient frequencies. Example pantries show precomputed results at once; an optional 36.0 MiB index download enables local browser search, with no login, API key or paid inference service.

The dataset contains normalized ingredient names, source-reported times and servings, source identifiers, recorded original links, and each recipe's recorded title and ingredient lines for the result cards. It does not copy cooking instructions, descriptions or images. The link-only filter is enabled by default; unlinked records remain available when disabled.

All records are checked against constraints, then at most 2,000 baseline-selected candidates receive learned scores. This is not a new quality benchmark or a guaranteed global learned top-k. The private catalog's recovery scores below do not measure this browser retrieval pipeline.

The [separately sourced twelve-recipe sample](https://huggingface.co/datasets/{DATASET_REPOSITORY}) remains available with complete Wikibooks instructions and attribution. Model weights and their terms are unchanged.
{end}"""
    before, remaining = card.split(start, 1)
    _, after = remaining.split(end, 1)
    updated = before + section + after
    # \s+ matches both the joined card and the hard-wrapped one already published on the Hub.
    return re.sub(
        r"\*\*Recipe search requires an authorized local catalog, its verified metadata\s+index, and the canonical",
        "**The text-backed Python finder requires an authorized local catalog, its verified metadata index, and the canonical",
        updated,
        count=1,
    )


def build_ingredient_demo(
    repository: Path, index_directory: Path, output: Path, *, ipv4: bool = False,
    source_revision: str | None = None, dataset_revision: str | None = None,
) -> dict:
    if any(value is not None and not re.fullmatch(r"[0-9a-f]{40}", value)
           for value in (source_revision, dataset_revision)):
        raise ValueError("source and dataset revisions must be immutable full commits")
    if dataset_revision is not None and source_revision is None:
        raise ValueError("a public dataset requires a pinned source revision")
    if os.path.lexists(output):
        raise FileExistsError(f"{output}: browser previews never overwrite existing outputs")
    identities = verify_source_revision(repository, source_revision, files=INGREDIENT_SOURCE_FILES) if source_revision else None
    metadata, arrays = load_ingredient_catalog(index_directory)
    text_records = load_text_shards(index_directory, metadata)
    _, policy, source = load_ingredient_policy(repository, metadata, ipv4=ipv4)
    exported_policy = browser_policy(policy, source)
    examples = precompute_examples(repository, index_directory, metadata, arrays, text_records, exported_policy)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ingredient-browser-", dir=output.parent) as temporary:
        staged = Path(temporary) / "space"
        staged.mkdir(mode=0o700)
        for filename in WEB_FILES:
            shutil.copyfile(repository / "model/demo" / filename, staged / filename)
        index_path = index_directory / "ingredient-index.json"
        index_location = f"https://huggingface.co/datasets/{INGREDIENT_DATASET_REPOSITORY}/resolve/{dataset_revision}/index/ingredient-index.json"
        if dataset_revision is None:
            data_directory = staged / "ingredient-data"
            data_directory.mkdir()
            shards = {record["file"]: record for record in (*metadata["url_shards"], *text_records)}
            data_files = ["ingredient-index.json", TEXT_MANIFEST,
                          *(record["file"] for record in metadata["arrays"].values()), *shards]
            for filename in data_files:
                source_path, destination = index_directory / filename, data_directory / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                record = shards.get(filename)
                if record and (source_path.stat().st_size != record["bytes"] or file_sha256(source_path) != record["sha256"]):
                    raise ValueError(f"{filename}: a URL or text shard differs from its index declaration")
                shutil.copyfile(source_path, destination)
            index_location = "ingredient-data/ingredient-index.json"
        bundle = {
            "mode": "ingredient-only", "catalog": metadata,
            "index": {
                "path": index_location, "bytes": index_path.stat().st_size,
                "sha256": file_sha256(index_path),
            },
            "examples": examples,
            "provenance": {
                "model_repository": MODEL_REPOSITORY, "model_revision": source["revision"],
                "dataset_repository": INGREDIENT_DATASET_REPOSITORY if dataset_revision else None,
                "dataset_revision": dataset_revision, "source_revision": source_revision,
                "ingredient_corpus_sha256": metadata["identity"]["corpus_sha256"],
                "publication_status": "pinned_public_dataset" if dataset_revision else "local_preview_only",
            },
        }
        (staged / "README.md").write_text(ingredient_space_card(metadata, bundle["provenance"]), encoding="utf-8")
        (staged / "catalog.json").write_bytes(_json_bytes(bundle))
        (staged / "policy.json").write_bytes(_json_bytes(exported_policy))
        manifest = {
            "schema_version": 1, "mode": "ingredient-only",
            "built_at": datetime.now(timezone.utc).isoformat(),
            "files": {
                filename: {"bytes": (staged / filename).stat().st_size,
                           "sha256": file_sha256(staged / filename)}
                for filename in (*WEB_FILES, "catalog.json", "policy.json")
            },
            "index_manifest": bundle["index"], "provenance": bundle["provenance"],
            "source_files_sha256": identities,
            "documentation_files": {
                "README.md": {"bytes": (staged / "README.md").stat().st_size,
                              "sha256": file_sha256(staged / "README.md")},
            },
            "catalog_summary": metadata["coverage"], "index_bytes": metadata["bytes"],
            "scoring": {
                "policy": "supervised", "max_candidates": MAX_CANDIDATES,
                "all_records_checked_for_constraints": True,
                "shortlist": "top deterministic-baseline scores, then original record ID",
                "learned_global_top_k_guaranteed": False,
                "existing_live_benchmark_applies": False,
                "example_results": "precomputed by model/demo/test/full_catalog_bridge.js with the browser search code",
            },
        }
        (staged / "manifest.json").write_bytes(_json_bytes(manifest))
        _publish_directory(staged, output)
    return manifest
