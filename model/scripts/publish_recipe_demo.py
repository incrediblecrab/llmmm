"""Publish the licensed sample and a free Static Space, with anonymous browser verification.

Requires a pushed source commit, Node, and the demo's declared Playwright tooling.
Never requests paid hardware, changes existing model weights, or uploads a private corpus.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download, set_client_factory
from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError

from ingredient_model._hashing import file_sha256
from ingredient_model.recipe_demo import (
    DATASET_FILES, DATASET_REPOSITORY, MODEL_REPOSITORY,
    SPACE_FILES, SPACE_REPOSITORY, WEB_FILES, add_demo_links, build_demo,
    load_public_policy, load_sample, verify_sample_provenance, verify_source_revision,
)

ROOT = Path(__file__).resolve().parents[2]


def write_report(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def public_files(api: HfApi, repository: str, kind: str, revision: str, expected: dict) -> None:
    info = api.repo_info(repository, repo_type=kind, revision=revision, token=False)
    if info.private or getattr(info, "gated", False):
        raise ValueError(f"{repository} is not public and ungated")
    names = {file.rfilename for file in info.siblings} - {".gitattributes"}
    if names != set(expected):
        raise ValueError(f"{repository} has an unexpected public file inventory: {names ^ set(expected)}")
    for name, data in expected.items():
        path = Path(hf_hub_download(
            repository, name, repo_type=kind, revision=revision, token=False,
            cache_dir=ROOT / "model/data/hf_cache"))
        if path.read_bytes() != data:
            raise ValueError(f"anonymous download differs from the intended release: {repository}/{name}")


def publish_files(api: HfApi, repository: str, kind: str, files: dict,
                  *, update: bool) -> str:
    initialized_extras = set()
    try:
        existing = api.repo_info(repository, repo_type=kind)
    except RepositoryNotFoundError as error:
        if error.response.status_code != 404:
            raise
        existing = None
    if existing is None:
        arguments = {"space_sdk": "static"} if kind == "space" else {}
        api.create_repo(repository, repo_type=kind, private=False, exist_ok=False, **arguments)
        existing = api.repo_info(repository, repo_type=kind)
        initialized_extras = {file.rfilename for file in existing.siblings} - {".gitattributes"} - set(files)
        if initialized_extras - ({"style.css"} if kind == "space" else set()):
            raise ValueError(f"the new repository contains unexpected starter files: {initialized_extras}")
    else:
        if existing.private or getattr(existing, "gated", False):
            raise ValueError(f"refusing to repurpose a private or gated repository: {repository}")
        names = {file.rfilename for file in existing.siblings} - {".gitattributes"}
        if names - set(files):
            raise ValueError(f"refusing to overwrite a repository with unrelated files: {repository}")
        if kind == "space" and getattr(existing, "sdk", None) != "static":
            raise ValueError("an existing Space must already use the free static SDK")
        same = names == set(files)
        if same:
            for name, data in files.items():
                path = Path(hf_hub_download(repository, name, repo_type=kind, revision=existing.sha,
                                            token=False, cache_dir=ROOT / "model/data/hf_cache"))
                if path.read_bytes() != data:
                    same = False
                    break
        if same:
            public_files(api, repository, kind, existing.sha, files)
            return existing.sha
        if not update:
            raise FileExistsError(f"{repository} already exists with different bytes; inspect it before --update")
        if kind == "space" and "manifest.json" in names:
            old = json.loads(Path(hf_hub_download(
                repository, "manifest.json", repo_type=kind, revision=existing.sha,
                token=False, cache_dir=ROOT / "model/data/hf_cache")).read_text())
            if old.get("provenance", {}).get("dataset_repository") != DATASET_REPOSITORY:
                raise ValueError("the existing Space is not this public-sample demo")
    commit = api.create_commit(
        repository, repo_type=kind, parent_commit=existing.sha,
        operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=data)
                    for name, data in sorted(files.items())]
        + [CommitOperationDelete(path_in_repo=name) for name in sorted(initialized_extras)],
        commit_message="Publish the public recipe sample" if kind == "dataset"
        else "Deploy the verified browser-only recipe finder",
    )
    public_files(api, repository, kind, commit.oid, files)
    return commit.oid


class QuietHandler(SimpleHTTPRequestHandler):
    def log_request(self, code="-", size="-"):
        if str(code).isdigit() and int(code) >= 400:
            super().log_request(code, size)


@contextmanager
def preview_server(directory: Path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(directory)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
        if thread.is_alive():
            raise RuntimeError("the temporary demo server did not shut down")


def _browser_tests(report: dict) -> list[dict]:
    tests = []
    for suite in report.get("suites", []):
        tests.extend(_browser_tests(suite))
    for spec in report.get("specs", []):
        tests.extend(spec["tests"])
    return tests


def browser_check(
    url: str, report_path: Path, manifest_sha256: str, *, script: str = "test:e2e",
    minimum_expected: int = 8, cases: list[str] | None = None,
) -> dict:
    environment = {**os.environ, "LLMMM_DEMO_URL": url, "LLMMM_DEMO_REPORT": str(report_path)}
    subprocess.run(["npm", "--prefix", str(ROOT / "model/demo"), "run", script],
                   cwd=ROOT, env=environment, check=True, timeout=600)
    report = json.loads(report_path.read_text())
    stats = report["stats"]
    tests = _browser_tests(report)
    if (report.get("errors") or stats["expected"] < minimum_expected or stats["unexpected"]
            or stats["flaky"] or stats["skipped"] or len(tests) != stats["expected"]):
        raise ValueError("the browser run was incomplete or contained failures, retries or skips")
    for test in tests:
        if (test["status"] != "expected" or not test["results"]
                or any(result["status"] != "passed" for result in test["results"])
                or not any(annotation.get("type") == "demo-manifest-sha256"
                           and annotation.get("description") == manifest_sha256
                           for annotation in test.get("annotations", []))):
            raise ValueError("a browser result is not bound to the intended serving manifest")
    return {
        "url": url, "passed": len(tests), "failed": 0, "skipped": 0, "retries": 0,
        "served_manifest_sha256": manifest_sha256, "report_sha256": file_sha256(report_path),
        "host_html_bootstrap_removed_before_source_hash_check": any(
            annotation.get("type") == "static-html-bootstrap-removed"
            and annotation.get("description") == "true"
            for test in tests for annotation in test.get("annotations", [])),
        "viewports": ["1360x1000", "390x844"],
        "cases": cases if cases is not None else ["anonymous_trained_inference", "original_recipe_text", "source_limitations_displayed", "zero_time_limit",
                  "ingredient_time_serving_constraints", "explicit_baseline_comparison",
                  "input_privacy_and_text_rendering", "corrupt_policy_rejected",
                  "failed_download_rejected", "responsive_layout"],
    }


def wait_for_space(client: httpx.Client, url: str, expected_manifest: bytes, *, timeout: int = 300) -> None:
    deadline = time.monotonic() + timeout
    last_error = "not yet fetched"
    while time.monotonic() < deadline:
        try:
            response = client.get(url + "/manifest.json")
            if response.status_code == 200 and response.content == expected_manifest:
                return
            last_error = f"HTTP {response.status_code}, serving manifest not current"
        except httpx.TransportError as error:
            last_error = str(error)
        time.sleep(10)
    raise TimeoutError(f"the public Space did not serve the intended release: {last_error}")


def link_model_card(api: HfApi, *, transform=add_demo_links,
                    commit_message: str = "Link the public recipe demo and attributed sample dataset") -> dict:
    receipt = json.loads((ROOT / "model/results/huggingface_recipe_search_release.json").read_text())
    original = {**receipt["files"], **receipt["preserved_native_files"]}
    before = api.model_info(MODEL_REPOSITORY, token=False)
    if before.private or before.gated:
        raise ValueError("the model must remain public and ungated")
    if {file.rfilename for file in before.siblings} - {".gitattributes"} != set(original):
        raise ValueError("the model's file inventory changed; inspect before adding demo links")
    tags_before = {ref.name: ref.target_commit for ref in api.list_repo_refs(MODEL_REPOSITORY).tags}
    before_files = {}
    for name, expected in original.items():
        path = Path(hf_hub_download(
            MODEL_REPOSITORY, name, revision=before.sha, token=False,
            cache_dir=ROOT / "model/data/hf_cache"))
        if name != "README.md" and file_sha256(path) != expected["sha256"]:
            raise ValueError(f"model artifact {name} changed since the recorded release")
        before_files[name] = path.read_bytes()
    card = transform(before_files["README.md"].decode("utf-8")).encode("utf-8")
    if card != before_files["README.md"]:
        commit = api.create_commit(
            MODEL_REPOSITORY, parent_commit=before.sha,
            operations=[CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card)],
            commit_message=commit_message,
        )
        revision = commit.oid
    else:
        revision = before.sha
    public_files(api, MODEL_REPOSITORY, "model", revision, {**before_files, "README.md": card})
    tags_after = {ref.name: ref.target_commit for ref in api.list_repo_refs(MODEL_REPOSITORY).tags}
    if tags_before != tags_after:
        raise ValueError("model tags changed during the documentation-only update")
    return {
        "previous_revision": before.sha, "revision": revision, "changed_file": "README.md",
        "readme_sha256": hashlib.sha256(card).hexdigest(),
        "non_readme_artifact_sha256": {name: hashlib.sha256(data).hexdigest()
                                    for name, data in before_files.items() if name != "README.md"},
        "preserved_tags": tags_after,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True, help="exact source commit, already pushed publicly")
    parser.add_argument("--out", type=Path, required=True, help="new artifact directory for this publication attempt")
    parser.add_argument("--report", type=Path, required=True, help="new final public release receipt")
    parser.add_argument("--dataset-tag", default="v0.1.0")
    parser.add_argument("--public", action="store_true", required=True)
    parser.add_argument("--update", action="store_true", help="allow inspected changes to existing demo repositories")
    parser.add_argument("--ipv4", action="store_true")
    args = parser.parse_args()
    output, report_path = args.out.resolve(), args.report.resolve()
    if output.exists() or report_path.exists():
        parser.error("publication artifacts and receipt paths must both be new")
    identities = verify_source_revision(ROOT, args.source_revision)
    transport = httpx.HTTPTransport(local_address="0.0.0.0") if args.ipv4 else httpx.HTTPTransport()
    with httpx.Client(transport=transport, follow_redirects=True, timeout=60) as client:
        if args.ipv4:
            set_client_factory(lambda: httpx.Client(
                transport=httpx.HTTPTransport(local_address="0.0.0.0"),
                follow_redirects=True, timeout=60))
        api = HfApi()
        if api.whoami()["name"] != "incrediblecrab":
            raise ValueError("publishing requires the owner of the declared model, dataset and Space")
        for name in WEB_FILES:
            response = client.get(
                f"https://raw.githubusercontent.com/incrediblecrab/llmmm/{args.source_revision}/model/demo/{name}")
            response.raise_for_status()
            if hashlib.sha256(response.content).hexdigest() != identities[f"model/demo/{name}"]:
                raise ValueError("the anonymous public Git revision does not match the serving code")
        vocabulary, _, _ = load_public_policy(ROOT, ipv4=args.ipv4)
        load_sample(ROOT / "model/demo_data/recipes.jsonl", vocabulary)
        verify_sample_provenance(ROOT)
        output.mkdir(parents=True, exist_ok=False)
        dataset_files = {name: (ROOT / "model/demo_data" / name).read_bytes() for name in DATASET_FILES}
        dataset_revision = publish_files(api, DATASET_REPOSITORY, "dataset", dataset_files, update=args.update)
        try:
            tagged = api.dataset_info(DATASET_REPOSITORY, revision=args.dataset_tag, token=False)
        except RevisionNotFoundError:
            api.create_tag(DATASET_REPOSITORY, repo_type="dataset", tag=args.dataset_tag, revision=dataset_revision)
        else:
            if tagged.sha != dataset_revision:
                raise ValueError("the dataset tag points elsewhere; choose a new tag rather than moving it")
        dataset_receipt = {
            "repository": DATASET_REPOSITORY, "revision": dataset_revision, "tag": args.dataset_tag,
            "url": f"https://huggingface.co/datasets/{DATASET_REPOSITORY}",
            "files": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                      for name, data in dataset_files.items()},
        }
        write_report(output / "dataset-upload.json", dataset_receipt)
        build = output / "space"
        manifest = build_demo(ROOT, build, source_revision=args.source_revision,
                              dataset_revision=dataset_revision, ipv4=args.ipv4)
        manifest_hash = file_sha256(build / "manifest.json")
        with preview_server(build) as local_url:
            local_browser = browser_check(local_url, output / "browser-local.json", manifest_hash)
        files = {name: (build / name).read_bytes() for name in SPACE_FILES}
        space_revision = publish_files(api, SPACE_REPOSITORY, "space", files, update=args.update)
        space_url = api.space_info(SPACE_REPOSITORY, token=False).host.rstrip("/")
        address = urlsplit(space_url)
        if (address.scheme != "https" or not address.hostname
                or not address.hostname.endswith(".static.hf.space") or address.username):
            raise ValueError("the Hub did not report a recognized public Static Space origin")
        write_report(output / "space-upload.json", {
            "repository": SPACE_REPOSITORY, "revision": space_revision, "app_url": space_url,
            "manifest_sha256": manifest_hash,
        })
        wait_for_space(client, space_url, files["manifest.json"])
        public_browser = browser_check(space_url, output / "browser-public.json", manifest_hash)
        if api.space_info(SPACE_REPOSITORY, token=False).sdk != "static":
            raise ValueError("the deployed Space is not a static application")
        public_files(api, DATASET_REPOSITORY, "dataset", dataset_revision, dataset_files)
        model_links = link_model_card(api)
        verify_source_revision(ROOT, args.source_revision)
        release = {
            "schema_version": 1, "status": "verified_public_browser_demo",
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "source_code_revision": args.source_revision, "source_files_sha256": identities,
            "dataset": dataset_receipt,
            "space": {
                "repository": SPACE_REPOSITORY, "revision": space_revision, "sdk": "static",
                "url": f"https://huggingface.co/spaces/{SPACE_REPOSITORY}", "app_url": space_url,
                "files": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                          for name, data in files.items()},
                "requested_paid_hardware": False,
            },
            "catalog": manifest["catalog_summary"], "model": manifest["provenance"],
            "numeric_and_search_parity": manifest["verification"],
            "local_browser": local_browser, "anonymous_public_browser": public_browser,
            "model_card_update": model_links,
            "private_recipe_data_uploaded": False,
            "new_quality_benchmark_claimed": False,
        }
        write_report(report_path, release)
        print(json.dumps({"space": release["space"]["url"], "dataset": dataset_receipt["url"],
                          "recipes": release["catalog"]["recipes"], "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
