"""Publish a verified all-record package privately, preserving older model tags."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import httpx
from huggingface_hub import (CommitOperationAdd, CommitOperationDelete, HfApi,
                             hf_hub_download, set_client_factory)
from huggingface_hub.utils import disable_progress_bars

from ingredient_model.config import PATHS
from ingredient_model.hub import IngredientPredictor

FILES = {
    "README.md", "config.json", "model.safetensors", "training_manifest.json",
    "training_verification.json", "release_policy.json",
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_package(folder: Path) -> dict:
    if {path.name for path in folder.iterdir()} != FILES:
        raise ValueError("the export must contain only the six allowed model-package files")
    report = json.loads((folder / "training_verification.json").read_text())
    if (report["evaluation_status"] != "not_run"
            or report["training"]["status"] != "verified_all_record_training"):
        raise ValueError("the package lacks verified all-record training evidence")
    for name, expected in {
        "model.safetensors": report["model_file"]["sha256"],
        "config.json": report["config_sha256"],
        "training_manifest.json": report["training"]["artifact_sha256"]["manifest.json"],
    }.items():
        if digest(folder / name) != expected:
            raise ValueError(f"{name}: package bytes differ from the verified export")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        default=PATHS.results / "huggingface_all_record_release.json")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"{args.out}: refusing to overwrite a publication receipt")
    package = read_package(args.folder)
    repo, tag = package["repo_id"], package["tag"]
    disable_progress_bars()
    set_client_factory(lambda: httpx.Client(
        transport=httpx.HTTPTransport(local_address="0.0.0.0"),
        follow_redirects=True, timeout=60.0))
    api = HfApi()
    before = api.model_info(repo)
    if not before.private:
        raise ValueError("this publisher requires an existing private model repository")
    tags = {ref.name: ref.target_commit for ref in api.list_repo_refs(repo).tags}
    if tag in tags:
        raise ValueError(f"{tag}: version tags are immutable; choose a new version")
    tag_revisions = {name: api.model_info(repo, revision=name).sha for name in tags}
    previous_files = set(api.list_repo_files(repo, revision=before.sha))
    if previous_files - FILES - {".gitattributes", "evaluation.json"}:
        raise ValueError("the remote repository contains unexpected files; inspect it before publishing")
    local = IngredientPredictor.from_pretrained(args.folder, local_files_only=True)
    vocabulary = local.vocabulary
    contexts = [vocabulary[:2], vocabulary[-2:], vocabulary[::2][:2]]
    top_k = min(10, len(vocabulary) - 2)
    expected = [local.recommend(context, top_k=top_k) for context in contexts]
    hashes = {
        name: {"bytes": (args.folder / name).stat().st_size,
               "sha256": digest(args.folder / name)}
        for name in sorted(FILES)
    }
    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=args.folder / name)
        for name in sorted(FILES)
    ]
    if "evaluation.json" in previous_files:
        operations.append(CommitOperationDelete(path_in_repo="evaluation.json"))
    commit = api.create_commit(
        repo, operations, parent_commit=before.sha,
        commit_message=f"Publish {tag}: all canonical recipe records, no held-out score")
    for name, expected_file in hashes.items():
        downloaded = Path(hf_hub_download(repo, name, revision=commit.oid))
        if (downloaded.stat().st_size != expected_file["bytes"]
                or digest(downloaded) != expected_file["sha256"]):
            raise ValueError(f"{name}: downloaded bytes differ from the published package")
    if set(api.list_repo_files(repo, revision=commit.oid)) - {".gitattributes"} != FILES:
        raise ValueError("the remote package inventory differs from the allowed files")
    child = """
import json, sys
from ingredient_model.config import PATHS
from ingredient_model.hub import IngredientPredictor
assert not PATHS.data.exists()
model = IngredientPredictor.from_pretrained(
    sys.argv[1], revision=sys.argv[2], token=True, local_files_only=True)
contexts = json.loads(sys.argv[3])
print(json.dumps([model.recommend(context, top_k=int(sys.argv[4])) for context in contexts]))
"""
    with tempfile.TemporaryDirectory(prefix="llmmm-inference-") as temporary:
        process = subprocess.run(
            [sys.executable, "-c", child, repo, commit.oid, json.dumps(contexts), str(top_k)],
            env={**os.environ, "IM_DATA": str(Path(temporary) / "absent-data")},
            check=True, capture_output=True, text=True)
    if json.loads(process.stdout) != expected:
        raise ValueError("the remote model's corpus-free recommendations differ from the local export")
    api.create_tag(repo, tag=tag, revision=commit.oid, exist_ok=False)
    after = api.model_info(repo)
    current_tags = {ref.name: ref.target_commit for ref in api.list_repo_refs(repo).tags}
    if (not after.private or after.sha != commit.oid
            or api.model_info(repo, revision=tag).sha != commit.oid
            or any(current_tags.get(name) != revision for name, revision in tags.items())):
        raise ValueError("repository visibility, publication revision or version tags changed unexpectedly")
    receipt = {
        "repo_id": repo, "url": f"https://huggingface.co/{repo}",
        "revision": commit.oid, "tag": tag, "private": after.private,
        "status": "verified_private_all_record_model",
        "source_code_revision": package["source_code_revision"],
        "recipes_per_epoch": package["training"]["recipes_per_epoch"],
        "epochs": package["training"]["epochs"],
        "example_presentations": package["training"]["example_presentations"],
        "evaluation_status": "not_run",
        "files": hashes, "preserved_tags": tag_revisions, "preserved_tag_refs": tags,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "verification": {
            "file_hashes_match": True, "version_tag_matches_commit": True,
            "remote_model_reloads": True, "inference_without_private_corpus": True,
            "recommendation_contexts_matched": len(contexts),
            "stale_evaluation_file_absent": True,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Verified private model: {receipt['url']}/tree/{tag}")
    print(f"Revision: {commit.oid}; publication receipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
