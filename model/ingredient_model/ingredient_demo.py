"""Assemble a local browser preview of the complete ingredient-only catalog."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ._hashing import file_sha256
from .data.recipe_search_metadata import _publish_directory
from .ingredient_catalog import load_ingredient_catalog
from .recipe_demo import (
    MODEL_REPOSITORY, WEB_FILES, _json_bytes, browser_policy, load_public_policy,
)
from .recipe_ranker import RecipeRankingPolicy


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


def build_ingredient_demo(repository: Path, index_directory: Path, output: Path, *, ipv4: bool = False) -> dict:
    if os.path.lexists(output):
        raise FileExistsError(f"{output}: browser previews never overwrite existing outputs")
    metadata, _ = load_ingredient_catalog(index_directory)
    _, policy, source = load_ingredient_policy(repository, metadata, ipv4=ipv4)
    exported_policy = browser_policy(policy, source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ingredient-browser-", dir=output.parent) as temporary:
        staged = Path(temporary) / "space"
        staged.mkdir(mode=0o700)
        for filename in WEB_FILES:
            shutil.copyfile(repository / "model/demo" / filename, staged / filename)
        data_directory = staged / "ingredient-data"
        data_directory.mkdir()
        data_files = ["ingredient-index.json",
                      *(record["file"] for record in metadata["arrays"].values()),
                      *(record["file"] for record in metadata["url_shards"])]
        for filename in data_files:
            source_path, destination = index_directory / filename, data_directory / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            if filename != "ingredient-index.json":
                record = next((item for item in metadata["url_shards"] if item["file"] == filename), None)
                if record and (source_path.stat().st_size != record["bytes"] or file_sha256(source_path) != record["sha256"]):
                    raise ValueError(f"{filename}: a URL shard differs from its index declaration")
            shutil.copyfile(source_path, destination)
        index_path = data_directory / "ingredient-index.json"
        bundle = {
            "mode": "ingredient-only", "catalog": metadata,
            "index": {
                "path": "ingredient-data/ingredient-index.json", "bytes": index_path.stat().st_size,
                "sha256": file_sha256(index_path),
            },
            "examples": [
                {"label": "Tomato and basil", "available_ingredients": ["tomato", "basil", "olive_oil", "salt", "garlic"],
                 "max_total_minutes": None, "max_missing": 2},
                {"label": "Chicken, rice, broccoli", "available_ingredients": ["chicken", "rice", "broccoli"],
                 "max_total_minutes": 30, "max_missing": 2},
                {"label": "Eggs and potatoes", "available_ingredients": ["egg", "potato", "onion", "salt", "oil"],
                 "max_total_minutes": 30, "max_missing": 2},
            ],
            "provenance": {
                "model_repository": MODEL_REPOSITORY, "model_revision": source["revision"],
                "dataset_repository": None, "dataset_revision": None, "source_revision": None,
                "ingredient_corpus_sha256": metadata["identity"]["corpus_sha256"],
                "publication_status": "local_preview_only",
            },
        }
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
            "catalog_summary": metadata["coverage"], "index_bytes": metadata["bytes"],
            "scoring": {
                "policy": "supervised", "max_candidates": 2000,
                "all_records_checked_for_constraints": True,
                "shortlist": "top deterministic-baseline scores, then original record ID",
                "learned_global_top_k_guaranteed": False,
                "existing_live_benchmark_applies": False,
            },
        }
        (staged / "manifest.json").write_bytes(_json_bytes(manifest))
        _publish_directory(staged, output)
    return manifest
