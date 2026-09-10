"""Publish the complete ingredient-only dataset and a free, verified Static Space."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download, set_client_factory
from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError

from ingredient_model._hashing import file_sha256
from ingredient_model.ingredient_catalog import load_ingredient_catalog
from ingredient_model.ingredient_demo import (
    INGREDIENT_DATASET_REPOSITORY, INGREDIENT_SOURCE_FILES, add_ingredient_demo_links,
    build_ingredient_demo,
)
from ingredient_model.recipe_demo import (
    DATASET_REPOSITORY, SPACE_FILES, SPACE_REPOSITORY, WEB_FILES, verify_source_revision,
)
from publish_recipe_demo import browser_check, link_model_card, preview_server, wait_for_space, write_report

ROOT = Path(__file__).resolve().parents[2]
PARQUET_COLUMNS = (
    "id", "ingredient_ids", "ingredients", "source", "language", "total_minutes", "servings", "source_url",
)
FULL_BROWSER_CASES = [
    "anonymous_trained_inference", "optional_large_download", "complete_population",
    "original_source_links", "no_copied_recipe_prose", "zero_time_limit",
    "required_excluded_and_missing_constraints", "explicit_baseline_comparison",
    "source_link_filter", "shortlist_disclosure", "input_privacy_and_text_rendering",
    "corrupt_index_rejected", "failed_download_rejected", "responsive_layout",
]


def inventory(directory: Path) -> dict[str, dict]:
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("publication trees may not contain symlinks")
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = {
                "bytes": path.stat().st_size, "sha256": file_sha256(path),
            }
    return result


def verified_dataset_files(directory: Path) -> dict[str, dict]:
    metadata, _ = load_ingredient_catalog(directory / "index")
    manifest = json.loads((directory / "dataset-manifest.json").read_text())
    files = inventory(directory)
    declared = manifest["files"]
    if set(files) != set(declared) | {"dataset-manifest.json"}:
        raise ValueError("the public dataset has files outside its verified inventory")
    for name, expected in declared.items():
        if files[name] != {"bytes": expected["bytes"], "sha256": expected["sha256"]}:
            raise ValueError(f"{name}: dataset file differs from its recorded bytes")
    required = {"README.md", "dataset-manifest.json", "index/ingredient-index.json"}
    required.update("index/" + record["file"] for record in metadata["arrays"].values())
    required.update("index/" + record["file"] for record in metadata["url_shards"])
    parquet_files = sorted(set(files) - required)
    if not parquet_files or parquet_files != [
            f"data/train-{position:05d}-of-{len(parquet_files):05d}.parquet"
            for position in range(len(parquet_files))]:
        raise ValueError("the public dataset must contain only the declared index and contiguous Parquet shards")
    return files


def verify_dataset_content(directory: Path, original_index: Path) -> dict:
    metadata, arrays = load_ingredient_catalog(original_index)
    released = json.loads((directory / "index/ingredient-index.json").read_text())
    if {k: v for k, v in released.items() if k != "publication"} != {
            k: v for k, v in metadata.items() if k != "publication"}:
        raise ValueError("public index changed data or statistics from the complete verified input")
    offsets = np.r_[0, np.cumsum(arrays["lengths"], dtype=np.int64)]
    vocabulary = pa.array(metadata["vocabulary"])
    sources, languages = pa.array(metadata["source_names"]), pa.array(metadata["language_names"])
    seen = slots = links = times = servings = 0
    cached_shard = -1
    urls = None
    for path in sorted((directory / "data").glob("*.parquet")):
        parquet = pq.ParquetFile(path)
        if tuple(parquet.schema_arrow.names) != PARQUET_COLUMNS:
            raise ValueError("a Parquet shard includes unexpected or missing fields")
        for batch in parquet.iter_batches(batch_size=metadata["rows_per_url_shard"]):
            last = seen + len(batch)
            if last > metadata["n_recipes"]:
                raise ValueError("Parquet contains extra ingredient records")
            columns = {name: batch.column(name) for name in PARQUET_COLUMNS}
            flat = pc.list_flatten(columns["ingredient_ids"])
            if (not np.array_equal(columns["id"].to_numpy(), np.arange(seen, last, dtype=np.uint32))
                    or not np.array_equal(pc.list_value_length(columns["ingredient_ids"]).to_numpy(),
                                          arrays["lengths"][seen:last])
                    or not np.array_equal(flat.to_numpy(), arrays["ingredients"][offsets[seen]:offsets[last]])
                    or not pc.list_flatten(columns["ingredients"]).equals(pc.take(vocabulary, flat))
                    or not np.array_equal(pc.list_value_length(columns["ingredients"]).to_numpy(),
                                          arrays["lengths"][seen:last])):
                raise ValueError("Parquet IDs, ingredient sets or vocabulary names differ from canonical input")
            for name, expected in (
                    ("source", pc.take(sources, pa.array(arrays["source_codes"][seen:last]))),
                    ("language", pc.take(languages, pa.array(arrays["language_codes"][seen:last])))):
                if not columns[name].equals(expected):
                    raise ValueError(f"Parquet {name} differs from the source metadata")
            for name in ("total_minutes", "servings"):
                values = arrays[name][seen:last]
                if not columns[name].equals(pa.array(values, mask=np.isnan(values))):
                    raise ValueError(f"Parquet {name} invented, removed or changed a source value")
            expected_urls = []
            position = seen
            while position < last:
                shard_id = position // metadata["rows_per_url_shard"]
                record = metadata["url_shards"][shard_id]
                if shard_id != cached_shard:
                    data = (original_index / record["file"]).read_bytes()
                    if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
                        raise ValueError("an original URL shard failed its integrity check")
                    raw = gzip.decompress(data)
                    if len(raw) != record["raw_bytes"] or hashlib.sha256(raw).hexdigest() != record["raw_sha256"]:
                        raise ValueError("an original URL shard failed its raw integrity check")
                    shard = json.loads(raw)
                    if (set(shard) != {"first_id", "urls"} or shard["first_id"] != record["first_id"]
                            or len(shard["urls"]) != record["rows"]):
                        raise ValueError("an original URL shard is not aligned")
                    urls, cached_shard = shard["urls"], shard_id
                end = min(last, record["first_id"] + record["rows"])
                expected_urls.extend(urls[position - record["first_id"]:end - record["first_id"]])
                position = end
            if not columns["source_url"].equals(pa.array(expected_urls, type=pa.string())):
                raise ValueError("Parquet source URLs differ from the original recorded links")
            slots += len(flat)
            links += len(batch) - columns["source_url"].null_count
            times += len(batch) - columns["total_minutes"].null_count
            servings += len(batch) - columns["servings"].null_count
            seen = last
    if (seen != metadata["n_recipes"] or slots != metadata["n_slots"]
            or links != metadata["coverage"]["url_statuses"]["source_url"]
            or times != metadata["coverage"]["source_total_times"]
            or servings != metadata["coverage"]["source_servings"]):
        raise ValueError("public Parquet does not cover every canonical record and source fact")
    return {
        "records_compared": seen, "ingredient_slots_compared": slots,
        "source_urls_compared": links, "source_total_times_compared": times,
        "source_serving_counts_compared": servings,
        "exact_fields": list(PARQUET_COLUMNS), "private_prose_exported": False,
        "input_index_sha256": file_sha256(original_index / "ingredient-index.json"),
        "public_index_sha256": file_sha256(directory / "index/ingredient-index.json"),
    }


def downloaded_file(repository: str, kind: str, revision: str, name: str) -> Path:
    return Path(hf_hub_download(
        repository, name, repo_type=kind, revision=revision, token=False,
        cache_dir=ROOT / "model/data/hf_cache"))


def verify_public_tree(api: HfApi, repository: str, kind: str, revision: str, files: dict) -> None:
    info = api.repo_info(repository, repo_type=kind, revision=revision, token=False)
    if info.private or getattr(info, "gated", False):
        raise ValueError("the released repository is not public and ungated")
    if {file.rfilename for file in info.siblings} - {".gitattributes"} != set(files):
        raise ValueError("the public repository has an unexpected file inventory")

    def verify(name):
        path = downloaded_file(repository, kind, revision, name)
        if path.stat().st_size != files[name]["bytes"] or file_sha256(path) != files[name]["sha256"]:
            raise ValueError(f"{repository}/{name}: anonymous download differs from the release")

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(verify, sorted(files)))


def publish_tree(
    api: HfApi, repository: str, kind: str, directory: Path, files: dict, *, update: bool,
) -> str:
    created = False
    try:
        existing = api.repo_info(repository, repo_type=kind)
    except RepositoryNotFoundError as error:
        if error.response.status_code != 404:
            raise
        api.create_repo(
            repository, repo_type=kind, private=False, exist_ok=False,
            **({"space_sdk": "static"} if kind == "space" else {}))
        existing = api.repo_info(repository, repo_type=kind)
        created = True
    if existing.private or getattr(existing, "gated", False):
        raise ValueError("refusing to repurpose a private or gated repository")
    if kind == "space" and getattr(existing, "sdk", None) != "static":
        raise ValueError("only the free static Space SDK is permitted")
    names = {file.rfilename for file in existing.siblings} - {".gitattributes"}
    extras = names - set(files)
    if extras and not (created and kind == "space" and extras == {"style.css"}):
        raise ValueError(f"refusing to delete unrelated remote files: {sorted(extras)}")
    same = names == set(files)
    if same:
        for name in sorted(files):
            path = downloaded_file(repository, kind, existing.sha, name)
            if path.stat().st_size != files[name]["bytes"] or file_sha256(path) != files[name]["sha256"]:
                same = False
                break
    if same:
        verify_public_tree(api, repository, kind, existing.sha, files)
        return existing.sha
    if not created:
        if not update:
            raise FileExistsError("the destination has different bytes; inspect it before --update")
        marker = "manifest.json" if kind == "space" else "dataset-manifest.json"
        if marker not in names:
            raise ValueError("the existing destination is not a recognized llmmm release")
        previous = json.loads(downloaded_file(repository, kind, existing.sha, marker).read_text())
        if kind == "space" and previous.get("provenance", {}).get("dataset_repository") not in {
                DATASET_REPOSITORY, INGREDIENT_DATASET_REPOSITORY}:
            raise ValueError("the existing Space belongs to a different dataset")
    commit = api.create_commit(
        repository, repo_type=kind, parent_commit=existing.sha,
        operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=directory / name)
                    for name in sorted(files)]
        + [CommitOperationDelete(path_in_repo=name) for name in sorted(extras)],
        commit_message="Publish the complete ingredient-only dataset" if kind == "dataset"
        else "Deploy full ingredient-only browser search without paid hardware")
    verify_public_tree(api, repository, kind, commit.oid, files)
    return commit.oid


def immutable_tag(api: HfApi, repository: str, kind: str, tag: str, revision: str) -> None:
    try:
        existing = api.repo_info(repository, repo_type=kind, revision=tag, token=False)
    except RevisionNotFoundError:
        api.create_tag(repository, repo_type=kind, tag=tag, revision=revision)
    else:
        if existing.sha != revision:
            raise ValueError(f"{tag} already points elsewhere; existing tags must not move")


def verify_dataset_viewer(client: httpx.Client, directory: Path, *, timeout: int = 300) -> dict:
    first_file = sorted((directory / "data").glob("*.parquet"))[0]
    first = next(pq.ParquetFile(first_file).iter_batches(batch_size=20)).to_pylist()
    if not first or any(set(row) != set(PARQUET_COLUMNS) for row in first):
        raise ValueError("viewer comparison requires the exact ingredient-only Parquet schema")
    expected_rows = json.loads((directory / "index/ingredient-index.json").read_text())["n_recipes"]
    deadline = time.monotonic() + timeout
    last_status = None
    while time.monotonic() < deadline:
        response = client.get("https://datasets-server.huggingface.co/first-rows", params={
            "dataset": INGREDIENT_DATASET_REPOSITORY, "config": "default", "split": "train"})
        last_status = response.status_code
        if response.status_code == 200:
            rows = response.json()["rows"][:len(first)]
            if (len(rows) != len(first) or any(item.get("truncated_cells") for item in rows)
                    or [item["row"] for item in rows] != first):
                raise ValueError("the public dataset viewer differs from the verified Parquet rows")
            size_response = client.get("https://datasets-server.huggingface.co/size", params={
                "dataset": INGREDIENT_DATASET_REPOSITORY})
            last_status = size_response.status_code
            if size_response.status_code == 200:
                size = size_response.json()
                if not size.get("partial") and not size.get("pending") and not size.get("failed"):
                    splits = size["size"]["splits"]
                    if (size["size"]["dataset"]["num_rows"] != expected_rows or len(splits) != 1
                            or splits[0]["config"] != "default" or splits[0]["split"] != "train"
                            or splits[0]["num_rows"] != expected_rows
                            or splits[0]["num_columns"] != len(PARQUET_COLUMNS)):
                        raise ValueError("the public viewer does not cover the full declared population and schema")
                    return {
                        "status": "verified", "rows_compared": len(first), "total_rows": expected_rows,
                        "exact_fields": list(PARQUET_COLUMNS),
                    }
            elif size_response.status_code not in {404, 500, 502, 503, 504}:
                size_response.raise_for_status()
        elif response.status_code not in {404, 500, 502, 503, 504}:
            response.raise_for_status()
        time.sleep(15)
    raise TimeoutError(f"public dataset viewer was not ready (last HTTP status {last_status})")


def main() -> None:
    from ingredient_model.ingredient_dataset import build_ingredient_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--public", action="store_true", required=True)
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--dataset-tag", default="v0.1.0")
    parser.add_argument("--ipv4", action="store_true")
    args = parser.parse_args()
    output, report_path = args.out.resolve(), args.report.resolve()
    if os.path.lexists(output) or os.path.lexists(report_path):
        parser.error("choose new publication artifact and receipt paths")
    authorization = json.loads((ROOT / "model/results/ingredient_catalog_publication_scope.json").read_text())
    if authorization["status"] != "owner_confirmed" or not authorization["decision"]["public_bulk_scope_established"]:
        raise ValueError("this release requires the recorded owner publication confirmation")
    index = args.index.resolve()
    if file_sha256(index / "ingredient-index.json") != authorization["index_sha256"]:
        raise ValueError("the publication input differs from the verified, authorized ingredient index")
    identities = verify_source_revision(ROOT, args.source_revision, files=INGREDIENT_SOURCE_FILES)
    if args.ipv4:
        set_client_factory(lambda: httpx.Client(
            transport=httpx.HTTPTransport(local_address="0.0.0.0"), follow_redirects=True, timeout=90))
    transport = httpx.HTTPTransport(local_address="0.0.0.0") if args.ipv4 else httpx.HTTPTransport()
    with httpx.Client(transport=transport, follow_redirects=True, timeout=90) as client:
        api = HfApi()
        if api.whoami()["name"] != "incrediblecrab":
            raise ValueError("publication requires the declared repository owner")
        for name in WEB_FILES:
            response = client.get(f"https://raw.githubusercontent.com/incrediblecrab/llmmm/{args.source_revision}/model/demo/{name}")
            response.raise_for_status()
            if hashlib.sha256(response.content).hexdigest() != identities[f"model/demo/{name}"]:
                raise ValueError("public source assets differ from the committed application")
        output.mkdir(parents=True, exist_ok=False)
        dataset = output / "dataset"
        build_ingredient_dataset(index, dataset, source_revision=args.source_revision)
        content_check = verify_dataset_content(dataset, index)
        write_report(output / "dataset-content-verification.json", content_check)
        dataset_files = verified_dataset_files(dataset)
        dataset_revision = publish_tree(
            api, INGREDIENT_DATASET_REPOSITORY, "dataset", dataset, dataset_files, update=args.update)
        dataset_receipt = {
            "repository": INGREDIENT_DATASET_REPOSITORY, "revision": dataset_revision,
            "url": f"https://huggingface.co/datasets/{INGREDIENT_DATASET_REPOSITORY}",
            "tag": args.dataset_tag, "files": dataset_files, "content_verification": content_check,
        }
        write_report(output / "dataset-upload.json", dataset_receipt)
        previous = json.loads((ROOT / "model/results/huggingface_recipe_demo_release.json").read_text())
        immutable_tag(api, SPACE_REPOSITORY, "space", "v0.1.0-sample", previous["space"]["revision"])
        build = output / "space"
        manifest = build_ingredient_demo(
            ROOT, dataset / "index", build, source_revision=args.source_revision,
            dataset_revision=dataset_revision, ipv4=args.ipv4)
        manifest_hash = file_sha256(build / "manifest.json")
        with preview_server(build) as local_url:
            local_browser = browser_check(
                local_url, output / "browser-local.json", manifest_hash,
                script="test:ingredients", minimum_expected=6, cases=FULL_BROWSER_CASES)
        space_files = inventory(build)
        if set(space_files) != set(SPACE_FILES):
            raise ValueError("the full Space must contain only the static application allowlist, not a private corpus")
        space_revision = publish_tree(
            api, SPACE_REPOSITORY, "space", build, space_files, update=args.update)
        info = api.space_info(SPACE_REPOSITORY, token=False)
        space_url = info.host.rstrip("/")
        address = urlsplit(space_url)
        if (address.scheme != "https" or not address.hostname or address.username
                or not address.hostname.endswith(".static.hf.space")):
            raise ValueError("the Hub did not report a public Static Space origin")
        write_report(output / "space-upload.json", {
            "repository": SPACE_REPOSITORY, "revision": space_revision, "app_url": space_url,
            "manifest_sha256": manifest_hash})
        wait_for_space(client, space_url, (build / "manifest.json").read_bytes())
        public_browser = browser_check(
            space_url, output / "browser-public.json", manifest_hash,
            script="test:ingredients", minimum_expected=6, cases=FULL_BROWSER_CASES)
        info = api.space_info(SPACE_REPOSITORY, token=False)
        if (info.sdk != "static" or info.private or getattr(info.runtime, "hardware", None)
                or getattr(info.runtime, "requested_hardware", None)):
            raise ValueError("the live demo must remain public, static and without paid hardware")
        viewer = verify_dataset_viewer(client, dataset)
        model_update = link_model_card(
            api, transform=add_ingredient_demo_links,
            commit_message="Link the full public ingredient dataset and browser-only recipe finder")
        immutable_tag(api, INGREDIENT_DATASET_REPOSITORY, "dataset", args.dataset_tag, dataset_revision)
        immutable_tag(api, SPACE_REPOSITORY, "space", "v0.2.0-ingredient-search", space_revision)
        verify_source_revision(ROOT, args.source_revision, files=INGREDIENT_SOURCE_FILES)
        release = {
            "schema_version": 1, "status": "verified_public_ingredient_search",
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "source_code_revision": args.source_revision, "source_files_sha256": identities,
            "dataset": dataset_receipt, "dataset_viewer": viewer,
            "space": {
                "repository": SPACE_REPOSITORY, "revision": space_revision,
                "url": f"https://huggingface.co/spaces/{SPACE_REPOSITORY}", "app_url": space_url,
                "sdk": "static", "requested_paid_hardware": False, "files": space_files,
                "preserved_sample_tag": "v0.1.0-sample", "tag": "v0.2.0-ingredient-search",
            },
            "catalog": manifest["catalog_summary"], "model": manifest["provenance"],
            "local_browser": local_browser, "anonymous_public_browser": public_browser,
            "model_card_update": model_update, "original_recipe_prose_uploaded": False,
            "new_quality_benchmark_claimed": False,
        }
        write_report(report_path, release)
        print(json.dumps({
            "dataset": dataset_receipt["url"], "space": release["space"]["url"],
            "records": content_check["records_compared"], "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
