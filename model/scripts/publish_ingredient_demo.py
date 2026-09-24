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
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import parse_ratelimit_headers

from ingredient_model._hashing import file_sha256
from ingredient_model.ingredient_catalog import load_ingredient_catalog, load_text_shards
from ingredient_model.ingredient_demo import (
    INGREDIENT_DATASET_REPOSITORY, INGREDIENT_SOURCE_FILES, add_ingredient_demo_links,
    build_ingredient_demo,
)
from ingredient_model.recipe_demo import SPACE_FILES, SPACE_REPOSITORY, WEB_FILES, verify_source_revision
from ingredient_model.recipe_links import LINK_STATUSES
from publish_recipe_demo import (
    _browser_tests, browser_check, link_model_card, preview_server, wait_for_space, write_report,
)

ROOT = Path(__file__).resolve().parents[2]
PARQUET_COLUMNS = (
    "id", "ingredient_ids", "ingredients", "source", "language", "total_minutes", "servings", "source_url",
    "recipe_link", "link_status",
)
FULL_BROWSER_CASES = [
    "precomputed_examples_before_download", "recipe_titles_and_ingredient_lines",
    "precomputed_equal_live_results", "anonymous_trained_inference", "optional_large_download",
    "complete_population", "recipe_card_links", "no_instruction_steps_rendered", "zero_time_limit",
    "required_excluded_and_missing_constraints", "explicit_baseline_comparison",
    "link_filter", "shortlist_disclosure", "canonical_match_details", "show_100_latency_recorded",
    "input_privacy_and_text_rendering", "corrupt_index_rejected", "failed_download_rejected",
    "corrupt_recipe_text_rejected", "dropped_download_retried", "responsive_layout",
]
# Five tests in ingredient-demo.spec.js, each in the desktop and mobile projects.
FULL_BROWSER_RESULTS = 10
# https://huggingface.co/docs/hub/storage-limits: "If you commit manually, keep around 50-100 files per commit."
MAX_COMMIT_FILES = 99
# Committed last, so a repository describes its previous release until every file it names is present.
CONTROL_FILES = {
    "dataset": {"README.md", "dataset-manifest.json", "index/ingredient-index.json"},
    "space": {"README.md", "index.html", "manifest.json"},
}
RELEASE_REPOSITORIES = {"dataset": INGREDIENT_DATASET_REPOSITORY, "space": SPACE_REPOSITORY}
# Measured September 24, 2026: the Hub allows 3,000 anonymous resolve requests per IP per fixed 5-minute window.
# Verifying a 2,872-file release anonymously can leave too few for a browser check, which then fails with HTTP 429.
BROWSER_RESOLVER_RESERVE = 1_500


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
    required = {"README.md", "dataset-manifest.json", "index/ingredient-index.json",
                "index/" + metadata["text_manifest"]["file"]}
    required.update("index/" + record["file"] for record in metadata["arrays"].values())
    required.update("index/" + record["file"] for record in (*metadata["url_shards"], *metadata["link_shards"]))
    required.update("index/" + record["file"] for record in load_text_shards(directory / "index", metadata))
    if not required <= set(files):
        raise ValueError("the public dataset is missing declared index files")
    parquet_files = sorted(set(files) - required)
    if not parquet_files or parquet_files != [
            f"data/train-{position:05d}-of-{len(parquet_files):05d}.parquet"
            for position in range(len(parquet_files))]:
        raise ValueError("the public dataset must contain only the declared index and contiguous Parquet shards")
    return files


def _original_shard(original_index: Path, record: dict, field: str) -> list:
    data = (original_index / record["file"]).read_bytes()
    if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError(f"an original {field} shard failed its integrity check")
    raw = gzip.decompress(data)
    if len(raw) != record["raw_bytes"] or hashlib.sha256(raw).hexdigest() != record["raw_sha256"]:
        raise ValueError(f"an original {field} shard failed its raw integrity check")
    shard = json.loads(raw)
    if set(shard) != {"first_id", field} or shard["first_id"] != record["first_id"] or len(shard[field]) != record["rows"]:
        raise ValueError(f"an original {field} shard is not aligned")
    return shard[field]


def verify_dataset_content(directory: Path, original_index: Path) -> dict:
    metadata, arrays = load_ingredient_catalog(original_index)
    released = json.loads((directory / "index/ingredient-index.json").read_text())
    if {k: v for k, v in released.items() if k != "publication"} != {
            k: v for k, v in metadata.items() if k != "publication"}:
        raise ValueError("public index changed data or statistics from the complete verified input")
    offsets = np.r_[0, np.cumsum(arrays["lengths"], dtype=np.int64)]
    vocabulary = pa.array(metadata["vocabulary"])
    sources, languages = pa.array(metadata["source_names"]), pa.array(metadata["language_names"])
    link_names = pa.array(LINK_STATUSES)
    seen = slots = links = card_links = times = servings = 0
    cached_shard = -1
    urls = card = None
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
            expected_urls, expected_links = [], []
            position = seen
            while position < last:
                shard_id = position // metadata["rows_per_url_shard"]
                record = metadata["url_shards"][shard_id]
                if shard_id != cached_shard:
                    urls = _original_shard(original_index, record, "urls")
                    card = _original_shard(original_index, metadata["link_shards"][shard_id], "links")
                    cached_shard = shard_id
                end = min(last, record["first_id"] + record["rows"])
                expected_urls.extend(urls[position - record["first_id"]:end - record["first_id"]])
                expected_links.extend(card[position - record["first_id"]:end - record["first_id"]])
                position = end
            if not columns["source_url"].equals(pa.array(expected_urls, type=pa.string())):
                raise ValueError("Parquet source URLs differ from the original recorded links")
            if (not columns["recipe_link"].equals(pa.array(expected_links, type=pa.string()))
                    or not columns["link_status"].equals(pc.take(link_names, pa.array(arrays["link_status"][seen:last])))):
                raise ValueError("Parquet card links or link statuses differ from the verified index")
            slots += len(flat)
            links += len(batch) - columns["source_url"].null_count
            card_links += len(batch) - columns["recipe_link"].null_count
            times += len(batch) - columns["total_minutes"].null_count
            servings += len(batch) - columns["servings"].null_count
            seen = last
    if (seen != metadata["n_recipes"] or slots != metadata["n_slots"]
            or links != metadata["coverage"]["url_statuses"]["source_url"]
            or card_links != sum(metadata["coverage"]["link_statuses"].get(name, 0) for name in ("source", "archive"))
            or times != metadata["coverage"]["source_total_times"]
            or servings != metadata["coverage"]["source_servings"]):
        raise ValueError("public Parquet does not cover every canonical record and source fact")
    text_shards = load_text_shards(directory / "index", released)
    for record in text_shards:
        path = directory / "index" / record["file"]
        if path.is_symlink() or path.stat().st_size != record["bytes"] or file_sha256(path) != record["sha256"]:
            raise ValueError(f"{record['file']}: dataset card text differs from the verified index")
    return {
        "records_compared": seen, "ingredient_slots_compared": slots,
        "source_urls_compared": links, "card_links_compared": card_links, "source_total_times_compared": times,
        "source_serving_counts_compared": servings,
        "text_shards_compared": len(text_shards),
        "text_shard_bytes_compared": sum(record["bytes"] for record in text_shards),
        "exact_fields": list(PARQUET_COLUMNS), "cooking_instructions_exported": False,
        "input_index_sha256": file_sha256(original_index / "ingredient-index.json"),
        "public_index_sha256": file_sha256(directory / "index/ingredient-index.json"),
    }


def downloaded_file(repository: str, kind: str, revision: str, name: str) -> Path:
    return Path(hf_hub_download(
        repository, name, repo_type=kind, revision=revision, token=False,
        cache_dir=ROOT / "model/data/hf_cache"))


def remote_files(api: HfApi, repository: str, kind: str, revision: str, *, token=None) -> dict[str, RepoFile]:
    return {entry.path: entry for entry in api.list_repo_tree(
        repository, repo_type=kind, revision=revision, recursive=True, token=token)
        if isinstance(entry, RepoFile) and entry.path != ".gitattributes"}


def matches_remote(path: Path, expected: dict, remote: RepoFile) -> bool:
    """Compare a local file with Hub tree metadata: LFS SHA256, or the git blob ID of a regular file."""
    if remote.lfs is not None:
        return remote.lfs.size == expected["bytes"] and remote.lfs.sha256 == expected["sha256"]
    data = path.read_bytes()
    return remote.size == len(data) and hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == remote.blob_id


def verify_public_tree(api: HfApi, repository: str, kind: str, revision: str, files: dict) -> None:
    info = api.repo_info(repository, repo_type=kind, revision=revision, token=False)
    if info.private or getattr(info, "gated", False):
        raise ValueError("the released repository is not public and ungated")
    if set(remote_files(api, repository, kind, revision, token=False)) != set(files):
        raise ValueError("the public repository has an unexpected file inventory")

    def verify(name):
        path = downloaded_file(repository, kind, revision, name)
        if path.stat().st_size != files[name]["bytes"] or file_sha256(path) != files[name]["sha256"]:
            raise ValueError(f"{repository}/{name}: anonymous download differs from the release")

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(verify, sorted(files)))


def wait_for_resolver_quota(client: httpx.Client, url: str, *, sleep=time.sleep) -> None:
    """Browser checks download the index anonymously; wait for a new window if too little of the current one remains."""
    for _ in range(3):
        quota = parse_ratelimit_headers(client.head(url, follow_redirects=False).headers)
        if quota is None or quota.remaining >= BROWSER_RESOLVER_RESERVE:
            return
        print(f"waiting {quota.reset_in_seconds + 1} s: {quota.remaining} anonymous Hub {quota.resource_type} requests remain")
        sleep(quota.reset_in_seconds + 1)
    raise RuntimeError("the Hub's anonymous request quota did not recover")


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
    remote = remote_files(api, repository, kind, existing.sha)
    marker = "manifest.json" if kind == "space" else "dataset-manifest.json"
    extras = set(remote) - set(files)
    unrelated = extras
    if extras and not created and marker in remote:
        # A file the previous release's own manifest lists is retired with it; any other extra is not ours to delete.
        listed = json.loads(downloaded_file(repository, kind, existing.sha, marker).read_text()).get("files", {})
        unrelated = extras - set(listed)
    if unrelated and not (created and kind == "space" and unrelated == {"style.css"}):
        raise ValueError(f"refusing to delete unrelated remote files: {sorted(unrelated)}")
    changed = [name for name in sorted(files)
               if name not in remote or not matches_remote(directory / name, files[name], remote[name])]
    if not changed and not extras:
        verify_public_tree(api, repository, kind, existing.sha, files)
        return existing.sha
    if not created:
        if not update:
            raise FileExistsError("the destination has different bytes; inspect it before --update")
        if marker not in remote:
            raise ValueError("the existing destination is not a recognized llmmm release")
        previous = json.loads(downloaded_file(repository, kind, existing.sha, marker).read_text())
        if kind == "space" and previous.get("provenance", {}).get("dataset_repository") != INGREDIENT_DATASET_REPOSITORY:
            raise ValueError("the existing Space belongs to a different dataset")
    if len(changed) + len(extras) <= MAX_COMMIT_FILES:
        batches = [changed]
    else:
        data = sorted((name for name in changed if name not in CONTROL_FILES[kind]), key=lambda name: name in remote)
        batches = [data[start:start + MAX_COMMIT_FILES] for start in range(0, len(data), MAX_COMMIT_FILES)]
        final = [name for name in changed if name in CONTROL_FILES[kind]]
        if final or extras:
            batches.append(final)
    message = ("Publish ingredient records with measured recipe-card links" if kind == "dataset"
               else "Deploy recipe cards with measured links, without paid hardware")
    parent = existing.sha
    for position, batch in enumerate(batches, 1):
        deletions = sorted(extras) if position == len(batches) else []
        commit = api.create_commit(
            repository, repo_type=kind, parent_commit=parent,
            operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=directory / name) for name in batch]
            + [CommitOperationDelete(path_in_repo=name) for name in deletions],
            commit_message=message if len(batches) == 1 else f"{message} ({position}/{len(batches)})")
        parent = commit.oid
    verify_public_tree(api, repository, kind, parent, files)
    return parent


def immutable_tag(api: HfApi, repository: str, kind: str, tag: str, revision: str) -> None:
    try:
        existing = api.repo_info(repository, repo_type=kind, revision=tag, token=False)
    except RevisionNotFoundError:
        api.create_tag(repository, repo_type=kind, tag=tag, revision=revision)
    else:
        if existing.sha != revision:
            raise ValueError(f"{tag} already points elsewhere; existing tags must not move")


def repository_tags(api: HfApi, repository: str, kind: str) -> dict[str, str]:
    return {ref.name: ref.target_commit for ref in api.list_repo_refs(repository, repo_type=kind).tags}


def unused_release_tags(api: HfApi, dataset_tag: str, space_tag: str) -> dict[str, dict[str, str]]:
    """Snapshot every existing tag before publishing, refusing a requested tag that already exists."""
    tags = {kind: repository_tags(api, repository, kind) for kind, repository in RELEASE_REPOSITORIES.items()}
    if dataset_tag in tags["dataset"] or space_tag in tags["space"]:
        raise ValueError("choose new release tags; existing tags are never moved or reused")
    return tags


def check_tags_preserved(api: HfApi, tags_before: dict[str, dict[str, str]]) -> None:
    for kind, repository in RELEASE_REPOSITORIES.items():
        tags = repository_tags(api, repository, kind)
        if any(tags.get(name) != target for name, target in tags_before[kind].items()):
            raise ValueError(f"an existing {kind} tag moved or disappeared during publication")


def show_100_milliseconds(report_path: Path) -> dict[str, int]:
    """Time from clicking "show 100" to 100 rendered cards, per browser viewport, from one run."""
    return {test["projectName"]: int(annotation["description"])
            for test in _browser_tests(json.loads(report_path.read_text()))
            for annotation in test.get("annotations", []) if annotation.get("type") == "show-100-ms"}


def verify_dataset_viewer(client: httpx.Client, directory: Path, *, timeout: int = 900) -> dict:
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
                # After an update, the viewer serves the previous revision's rows until it re-indexes.
                last_status = "200 with rows that differ from this release"
                time.sleep(15)
                continue
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
    parser.add_argument("--dataset-tag", required=True, help="new, unused dataset tag")
    parser.add_argument("--space-tag", required=True, help="new, unused Space tag")
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
        tags_before = unused_release_tags(api, args.dataset_tag, args.space_tag)
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
        quota_url = (f"https://huggingface.co/datasets/{INGREDIENT_DATASET_REPOSITORY}/resolve/"
                     f"{dataset_revision}/dataset-manifest.json")
        wait_for_resolver_quota(client, quota_url)
        with preview_server(build) as local_url:
            local_browser = browser_check(
                local_url, output / "browser-local.json", manifest_hash,
                script="test:ingredients", minimum_expected=FULL_BROWSER_RESULTS, cases=FULL_BROWSER_CASES)
        local_browser["show_100_ms"] = show_100_milliseconds(output / "browser-local.json")
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
        wait_for_resolver_quota(client, quota_url)
        public_browser = browser_check(
            space_url, output / "browser-public.json", manifest_hash,
            script="test:ingredients", minimum_expected=FULL_BROWSER_RESULTS, cases=FULL_BROWSER_CASES)
        public_browser["show_100_ms"] = show_100_milliseconds(output / "browser-public.json")
        info = api.space_info(SPACE_REPOSITORY, token=False)
        if (info.sdk != "static" or info.private or getattr(info.runtime, "hardware", None)
                or getattr(info.runtime, "requested_hardware", None)):
            raise ValueError("the live demo must remain public, static and without paid hardware")
        viewer = verify_dataset_viewer(client, dataset)
        model_update = link_model_card(
            api, transform=add_ingredient_demo_links,
            commit_message="Describe the demo's recipe-card links; drop the retired sample dataset link")
        immutable_tag(api, INGREDIENT_DATASET_REPOSITORY, "dataset", args.dataset_tag, dataset_revision)
        immutable_tag(api, SPACE_REPOSITORY, "space", args.space_tag, space_revision)
        check_tags_preserved(api, tags_before)
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
                "preserved_sample_tag": "v0.1.0-sample", "tag": args.space_tag,
            },
            "preserved_tags": tags_before,
            "catalog": manifest["catalog_summary"], "model": manifest["provenance"],
            "local_browser": local_browser, "anonymous_public_browser": public_browser,
            "model_card_update": model_update, "recipe_titles_and_ingredient_lines_uploaded": True,
            "cooking_instructions_uploaded": False, "new_quality_benchmark_claimed": False,
        }
        write_report(report_path, release)
        print(json.dumps({
            "dataset": dataset_receipt["url"], "space": release["space"]["url"],
            "records": content_check["records_compared"], "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
