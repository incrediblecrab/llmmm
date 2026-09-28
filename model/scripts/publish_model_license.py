"""Publish the model's MIT license: card, policy files and LICENSE only, with every weight file preserved.

Without --publish this is a dry run: it verifies the live model, renders the licensed files into --stage
and prints the card diff, uploading nothing.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import httpx
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download, set_client_factory

from ingredient_model.config import PATHS
from ingredient_model.hub import IngredientPredictor
from ingredient_model.recipe_demo import MODEL_REPOSITORY
from export_native_model import WEIGHTS_LICENSE, WEIGHTS_LICENSE_HUB_ID, WEIGHTS_LICENSE_NOTE
from export_recipe_search import render_card

ROOT = Path(__file__).resolve().parents[2]
TAG = "v0.4.1-mit"
RECEIPT = PATHS.results / "huggingface_model_license_update.json"
LATEST_CARD_RECEIPT = PATHS.results / "huggingface_recipe_link_release.json"
SEARCH_RECEIPT = PATHS.results / "huggingface_recipe_search_release.json"
NATIVE_EVIDENCE = PATHS.results / "public_model_export.json"
CHANGED = ("README.md", "LICENSE", "release_policy.json", "recipe_release_policy.json")
SOURCE_FILES = (
    "LICENSE", "model/scripts/export_native_model.py", "model/scripts/export_recipe_search.py",
    "model/ingredient_model/ingredient_demo.py", "model/scripts/publish_model_license.py",
)
# The card's only permitted differences: the metadata license, the licensing section and one demo sentence.
CARD_EDITS = (
    ("---\nlibrary_name: pytorch\n", f"---\nlicense: {WEIGHTS_LICENSE_HUB_ID}\nlibrary_name: pytorch\n"),
    ("Model weights and their terms are unchanged.", "Model weights are unchanged."),
    ("## Intended use and limits\n\nFor noncommercial research and education. No permissive weights license is granted. ",
     f"## License and limits\n\nReleased under the [MIT License](LICENSE). {WEIGHTS_LICENSE_NOTE} "),
    (" and applicable terms before redistribution or commercial use.", " before reusing any source data."),
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def record(data: bytes) -> dict:
    return {"bytes": len(data), "sha256": sha256(data)}


def json_bytes(value: dict, **options) -> bytes:
    return (json.dumps(value, indent=2, **options) + "\n").encode("utf-8")


def verify_source_revision(revision: str) -> None:
    for name in SOURCE_FILES:
        stored = subprocess.run(["git", "show", f"{revision}:{name}"], cwd=ROOT, capture_output=True, check=False)
        if stored.returncode or stored.stdout != (ROOT / name).read_bytes():
            raise ValueError(f"{name}: commit and push the tested source before publishing")
    remote = subprocess.run(["git", "branch", "-r", "--contains", revision], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    if "origin/" not in remote.stdout:
        raise ValueError("the source revision must be pushed so the card's links resolve")


def licensed_files(live: dict[str, bytes], source_revision: str) -> dict[str, bytes]:
    load = lambda name: json.loads(live[name])
    native = json.loads(NATIVE_EVIDENCE.read_text())
    rendered = render_card(native, load("recipe_search_config.json"), load("recipe_training.json"),
                           load("recipe_evaluation.json"), load("recipe_catalog.json")).encode("utf-8")
    expected = live["README.md"].decode("utf-8")
    for old, new in CARD_EDITS:
        if expected.count(old) != 1:
            raise ValueError(f"the live card no longer contains exactly one {old[:40]!r}")
        expected = expected.replace(old, new)
    if rendered != expected.encode("utf-8"):
        raise ValueError("the generators' card differs from the live card beyond the license edits")

    policies = {}
    for name, fields, options in (
            ("release_policy.json", {"weights_license": WEIGHTS_LICENSE,
                                     "intended_use": "Ingredient completion; predictions are not validated for taste, allergies or food safety.",
                                     "rights_note": WEIGHTS_LICENSE_NOTE}, {}),
            ("recipe_release_policy.json", {"license": WEIGHTS_LICENSE, "license_note": WEIGHTS_LICENSE_NOTE},
             {"allow_nan": False})):
        document = load(name)
        if json_bytes(document, **options) != live[name]:
            raise ValueError(f"{name}: live bytes are not in the exporter's JSON format")
        policies[name] = json_bytes({**document, **fields}, **options)
    license_text = subprocess.run(["git", "show", f"{source_revision}:LICENSE"], cwd=ROOT,
                                  capture_output=True, check=True).stdout
    if not license_text.startswith(b"MIT License\n"):
        raise ValueError("the repository LICENSE is not the MIT License")
    return {"README.md": rendered, "LICENSE": license_text, **policies}


def recommendations(revision: str) -> list:
    model = IngredientPredictor.from_pretrained(MODEL_REPOSITORY, revision=revision, token=False)
    contexts = [["tomato", "basil"], list(model.vocabulary[:2]), list(model.vocabulary[-2:])]
    return [model.recommend(context, top_k=10) for context in contexts]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--stage", type=Path, required=True, help="empty directory for the rendered files")
    parser.add_argument("--publish", action="store_true", help="upload, tag and write the receipt")
    args = parser.parse_args()
    if RECEIPT.exists():
        raise FileExistsError(f"{RECEIPT}: refusing to overwrite a publication receipt")
    if args.stage.exists() and any(args.stage.iterdir()):
        raise FileExistsError(f"{args.stage}: the stage directory must be empty")
    if args.publish:
        verify_source_revision(args.source_revision)
    set_client_factory(lambda: httpx.Client(
        transport=httpx.HTTPTransport(local_address="0.0.0.0"), follow_redirects=True, timeout=60))
    api = HfApi()

    search = json.loads(SEARCH_RECEIPT.read_text())
    card_update = json.loads(LATEST_CARD_RECEIPT.read_text())["model_card_update"]
    original = {**search["files"], **search["preserved_native_files"]}
    before = api.model_info(MODEL_REPOSITORY, token=False)
    if before.private or before.gated or before.sha != card_update["revision"]:
        raise ValueError("the public model's main branch is not the last recorded card revision")
    if {file.rfilename for file in before.siblings} - {".gitattributes"} != set(original):
        raise ValueError("the model's file inventory changed since the recorded release")
    tags_before = {ref.name: ref.target_commit for ref in api.list_repo_refs(MODEL_REPOSITORY).tags}
    if TAG in tags_before:
        raise ValueError("version tags are immutable")
    live = {}
    for name in original:
        live[name] = Path(hf_hub_download(MODEL_REPOSITORY, name, revision=before.sha, token=False,
                                          cache_dir=ROOT / "model/data/hf_cache")).read_bytes()
        expected = card_update["readme_sha256"] if name == "README.md" else card_update["non_readme_artifact_sha256"][name]
        if sha256(live[name]) != expected:
            raise ValueError(f"{name}: live bytes differ from the recorded release")

    changed = licensed_files(live, args.source_revision)
    args.stage.mkdir(parents=True, exist_ok=True)
    for name, data in changed.items():
        (args.stage / name).write_bytes(data)
    for name in ("README.md", "release_policy.json", "recipe_release_policy.json"):
        print("".join(difflib.unified_diff(
            live[name].decode().splitlines(True), changed[name].decode().splitlines(True),
            f"live/{name}", f"licensed/{name}", n=0)))
    if not args.publish:
        print(f"Dry run: staged {sorted(changed)} in {args.stage}; nothing uploaded")
        return 0

    expected_inference = recommendations(before.sha)
    commit = api.create_commit(
        MODEL_REPOSITORY, parent_commit=before.sha,
        operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=changed[name]) for name in CHANGED],
        commit_message="Release the model under the MIT License; weights unchanged")
    after_files = {**live, **changed}
    from publish_recipe_demo import public_files
    public_files(api, MODEL_REPOSITORY, "model", commit.oid, after_files)
    if recommendations(commit.oid) != expected_inference:
        raise ValueError("anonymous inference changed with a license-only update")
    api.create_tag(MODEL_REPOSITORY, tag=TAG, revision=commit.oid, exist_ok=False)
    after = HfApi(token=False).model_info(MODEL_REPOSITORY)
    tags_after = {ref.name: ref.target_commit for ref in api.list_repo_refs(MODEL_REPOSITORY).tags}
    card_license = (after.card_data or {}).get("license")
    if (after.private or after.gated or after.sha != commit.oid or card_license != WEIGHTS_LICENSE_HUB_ID
            or tags_after != {**tags_before, TAG: commit.oid}):
        raise ValueError("visibility, main revision, card license or preserved tags are not as published")

    receipt = {
        "schema_version": 1, "status": "verified_public_license_update",
        "repo_id": MODEL_REPOSITORY, "url": f"https://huggingface.co/{MODEL_REPOSITORY}",
        "tag": TAG, "revision": commit.oid, "previous_revision": before.sha,
        "source_code_revision": args.source_revision, "license": WEIGHTS_LICENSE,
        "hub_card_license": card_license, "license_note": WEIGHTS_LICENSE_NOTE,
        "changed_files": {name: {"before": record(live[name]) if name in live else None,
                                 "after": record(changed[name])} for name in CHANGED},
        "files": {name: record(data) for name, data in sorted(after_files.items())},
        "unchanged_files_verified": len(original) - len(set(changed) & set(original)),
        "card_generators_reproduce_live_card": True,
        "card_edits": [{"before": old, "after": new} for old, new in CARD_EDITS],
        "anonymous_inference_identical_before_and_after": True,
        "inference_contexts": 3, "preserved_tags": tags_before,
        "weights_changed": False, "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Published {TAG} at {commit.oid}; receipt {RECEIPT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
