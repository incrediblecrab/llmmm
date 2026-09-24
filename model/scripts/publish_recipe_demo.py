"""Hub publishing helpers shared by publish_ingredient_demo.py: browser checks, Space readiness, model card links.

This module used to publish the twelve-recipe Wikibooks sample dataset and its Space. That path is retired:
incrediblecrab/llmmm-recipe-ingredients is the project's only public dataset, and publish_ingredient_demo.py
is the only publisher. The sample's tracked copy remains in model/demo_data for local previews.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from ingredient_model._hashing import file_sha256
from ingredient_model.recipe_demo import MODEL_REPOSITORY

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


def link_model_card(api: HfApi, *, transform, commit_message: str) -> dict:
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
