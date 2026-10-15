"""Relicense the published model, dataset and Space under CC BY-NC 4.0: cards, LICENSE files, the model's policy files
and the dataset and Space manifests' records of those files only. No weight, data, index or application file changes.

Without --publish this is a dry run: it verifies each live repository against its last receipt, renders the relicensed
files from the generators into --stage and prints the card diffs, uploading nothing.
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
from ingredient_model.ingredient_catalog import _json_bytes as dataset_json_bytes
from ingredient_model.ingredient_dataset import _dataset_card
from ingredient_model.ingredient_demo import INGREDIENT_DATASET_REPOSITORY, ingredient_space_card
from ingredient_model.licensing import (
    DATASET_LICENSE_NOTE, LICENSE_HUB_ID, LICENSE_NAME, LICENSE_TITLE, SPACE_LICENSE_NOTE, WEIGHTS_LICENSE_NOTE,
    license_bytes,
)
from ingredient_model.recipe_demo import MODEL_INVENTORY_RECEIPT, MODEL_REPOSITORY, SPACE_REPOSITORY
from ingredient_model.recipe_demo import _json_bytes as space_json_bytes
from export_recipe_search import render_card
from publish_model_license import recommendations
from publish_recipe_demo import wait_for_space

ROOT = Path(__file__).resolve().parents[2]
TAGS = {"model": "v0.4.2-cc-by-nc", "dataset": "v0.3.1-cc-by-nc", "space": "v0.4.1-cc-by-nc"}
REPOSITORIES = {"model": MODEL_REPOSITORY, "dataset": INGREDIENT_DATASET_REPOSITORY, "space": SPACE_REPOSITORY}
PREVIOUS_LICENSES = {"model": "MIT", "dataset": "other", "space": None}
# Files each renderer reads, downloaded and hash-checked even where the Hub stores them in LFS.
RENDER_INPUTS = {
    "model": ("README.md", "release_policy.json", "recipe_release_policy.json", "recipe_search_config.json",
              "recipe_training.json", "recipe_evaluation.json", "recipe_catalog.json"),
    "dataset": ("README.md", "dataset-manifest.json", "index/ingredient-index.json"),
    "space": ("README.md", "manifest.json", "catalog.json"),
}
MODEL_RECEIPT = PATHS.results / MODEL_INVENTORY_RECEIPT
RELEASE_RECEIPT = PATHS.results / "huggingface_dataset_space_cc_by_nc_update.json"
PREVIOUS_MODEL_RECEIPT = PATHS.results / "huggingface_model_license_update.json"
PREVIOUS_RELEASE_RECEIPT = PATHS.results / "huggingface_recipe_link_release.json"
NATIVE_EVIDENCE = PATHS.results / "public_model_export.json"
SPACE_URL = "https://incrediblecrab-llmmm-recipes-demo.static.hf.space"
SOURCE_FILES = (
    "model/ingredient_model/licensing.py", "model/ingredient_model/licenses/CC-BY-NC-4.0.txt",
    "model/ingredient_model/ingredient_dataset.py", "model/ingredient_model/ingredient_demo.py",
    "model/ingredient_model/recipe_demo.py", "model/scripts/export_native_model.py",
    "model/scripts/export_recipe_search.py", "model/scripts/publish_license_update.py",
)
# Each card's only permitted differences from the live card; the generators must reproduce the result exactly.
CARD_EDITS = {
    "model": (
        ("license: mit\n", f"license: {LICENSE_HUB_ID}\n"),
        ("Released under the [MIT License](LICENSE). The MIT License covers the weights and configuration files in this "
         "repository. It grants no rights in the recipes or source datasets used for training, which keep their own terms. ",
         f"Released under the [{LICENSE_TITLE}](LICENSE) ({LICENSE_NAME}). {WEIGHTS_LICENSE_NOTE} "),
    ),
    "dataset": (
        ("license: other\n", f"license: {LICENSE_HUB_ID}\n"),
        ("[Model and its separate terms](", "[Model]("),
        ("## Use\n\nThe maintainer confirmed permission to publish these ingredient facts, titles and ingredient lines. "
         "`license: other` is not a grant of rights to source-page instructions, prose or photos, or to the separate model "
         "weights. Source-page material and model weights retain their separate terms.\n",
         f"## License\n\nReleased under the [{LICENSE_TITLE}](LICENSE) ({LICENSE_NAME}). {DATASET_LICENSE_NOTE} "
         "The maintainer confirmed permission to publish these ingredient facts, titles and ingredient lines.\n"),
    ),
    "space": (
        ("app_file: index.html\n", f"app_file: index.html\nlicense: {LICENSE_HUB_ID}\n"),
        ("The maintainer confirmed permission to publish this extract, including the recorded titles and ingredient "
         "lines. Cooking instructions and other recipe-page content stay on the source sites under their own terms, and "
         "the model weights keep their separate terms. ",
         f"Released under the [{LICENSE_TITLE}](LICENSE) ({LICENSE_NAME}). {SPACE_LICENSE_NOTE} The maintainer "
         "confirmed permission to publish this extract, including the recorded titles and ingredient lines. Cooking "
         "instructions and other recipe-page content stay on the source sites. "),
    ),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def record(data: bytes) -> dict:
    return {"bytes": len(data), "sha256": sha256(data)}


def policy_bytes(value: dict, **options) -> bytes:
    return (json.dumps(value, indent=2, **options) + "\n").encode("utf-8")


def edited(kind: str, card: bytes) -> bytes:
    text = card.decode("utf-8")
    for old, new in CARD_EDITS[kind]:
        if text.count(old) != 1:
            raise ValueError(f"{kind}: the live card no longer contains exactly one {old[:40]!r}")
        text = text.replace(old, new)
    return text.encode("utf-8")


def verify_source_revision(revision: str) -> None:
    for name in SOURCE_FILES:
        stored = subprocess.run(["git", "show", f"{revision}:{name}"], cwd=ROOT, capture_output=True, check=False)
        if stored.returncode or stored.stdout != (ROOT / name).read_bytes():
            raise ValueError(f"{name}: commit and push the tested source before publishing")
    remote = subprocess.run(["git", "branch", "-r", "--contains", revision], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    if "origin/" not in remote.stdout:
        raise ValueError("the source revision must be pushed before publishing")


def recorded_releases() -> dict[str, dict]:
    """The last receipt's revision and per-file records for each repository."""
    model = json.loads(PREVIOUS_MODEL_RECEIPT.read_text())
    release = json.loads(PREVIOUS_RELEASE_RECEIPT.read_text())
    flatten = lambda files: {name: {"bytes": value["bytes"], "sha256": value["sha256"]} if isinstance(value, dict)
                             else value for name, value in files.items()}
    return {
        "model": {"revision": model["revision"], "files": model["files"]},
        "dataset": {"revision": release["dataset"]["revision"], "files": flatten(release["dataset"]["files"])},
        "space": {"revision": release["space"]["revision"], "files": flatten(release["space"]["files"])},
    }


def live_files(api: HfApi, kind: str, expected: dict) -> tuple[str, dict[str, bytes]]:
    """Check head, visibility and every file against the receipt; return the bytes of every regular file and render input."""
    repository, revision = REPOSITORIES[kind], expected["revision"]
    info = api.repo_info(repository, repo_type=kind, files_metadata=True, token=False)
    if info.private or getattr(info, "gated", False) or info.sha != revision:
        raise ValueError(f"{repository}: main is not the last recorded public revision")
    siblings = {entry.rfilename: entry for entry in info.siblings if entry.rfilename != ".gitattributes"}
    if set(siblings) != set(expected["files"]):
        raise ValueError(f"{repository}: the file inventory changed since the recorded release")
    small = {}
    for name, wanted in expected["files"].items():
        entry = siblings[name]
        if entry.lfs is not None and name not in RENDER_INPUTS[kind]:
            if entry.lfs.sha256 != wanted["sha256"] or entry.lfs.size != wanted["bytes"]:
                raise ValueError(f"{repository}/{name}: live bytes differ from the recorded release")
            continue
        data = Path(hf_hub_download(repository, name, repo_type=kind, revision=revision, token=False,
                                    cache_dir=ROOT / "model/data/hf_cache")).read_bytes()
        if record(data) != {"bytes": wanted["bytes"], "sha256": wanted["sha256"]}:
            raise ValueError(f"{repository}/{name}: live bytes differ from the recorded release")
        small[name] = data
    return revision, small


def model_files(live: dict[str, bytes]) -> dict[str, bytes]:
    load = lambda name: json.loads(live[name])
    rendered = render_card(json.loads(NATIVE_EVIDENCE.read_text()), load("recipe_search_config.json"),
                           load("recipe_training.json"), load("recipe_evaluation.json"),
                           load("recipe_catalog.json")).encode("utf-8")
    if rendered != edited("model", live["README.md"]):
        raise ValueError("model: the generators' card differs from the live card beyond the license edits")
    policies = {}
    for name, fields, options in (
            ("release_policy.json", {"weights_license": LICENSE_NAME, "rights_note": WEIGHTS_LICENSE_NOTE}, {}),
            ("recipe_release_policy.json", {"license": LICENSE_NAME, "license_note": WEIGHTS_LICENSE_NOTE},
             {"allow_nan": False})):
        document = load(name)
        if policy_bytes(document, **options) != live[name]:
            raise ValueError(f"{name}: live bytes are not in the exporter's JSON format")
        policies[name] = policy_bytes({**document, **fields}, **options)
    return {"README.md": rendered, "LICENSE": license_bytes(), **policies}


def dataset_files(live: dict[str, bytes]) -> dict[str, bytes]:
    manifest = json.loads(live["dataset-manifest.json"])
    if dataset_json_bytes(manifest) != live["dataset-manifest.json"]:
        raise ValueError("dataset-manifest.json: live bytes are not in the builder's JSON format")
    metadata = json.loads(live["index/ingredient-index.json"])
    rendered = _dataset_card(metadata, manifest["source_revision"]).encode("utf-8")
    if rendered != edited("dataset", live["README.md"]):
        raise ValueError("dataset: the generator's card differs from the live card beyond the license edits")
    changed = {"README.md": rendered, "LICENSE": license_bytes()}
    files = {**manifest["files"], **{name: record(data) for name, data in changed.items()}}
    manifest["files"] = {name: files[name] for name in sorted(files)}
    return {**changed, "dataset-manifest.json": dataset_json_bytes(manifest)}


def space_files(live: dict[str, bytes]) -> dict[str, bytes]:
    manifest = json.loads(live["manifest.json"])
    if space_json_bytes(manifest) != live["manifest.json"]:
        raise ValueError("manifest.json: live bytes are not in the builder's JSON format")
    bundle = json.loads(live["catalog.json"])
    rendered = ingredient_space_card(bundle["catalog"], bundle["provenance"]).encode("utf-8")
    if rendered != edited("space", live["README.md"]):
        raise ValueError("space: the generator's card differs from the live card beyond the license edits")
    changed = {"README.md": rendered, "LICENSE": license_bytes()}
    manifest["documentation_files"] = {name: record(changed[name]) for name in ("README.md", "LICENSE")}
    return {**changed, "manifest.json": space_json_bytes(manifest)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--stage", type=Path, required=True, help="empty directory for the rendered files")
    parser.add_argument("--publish", action="store_true", help="upload, tag and write the receipts")
    args = parser.parse_args()
    for receipt in (MODEL_RECEIPT, RELEASE_RECEIPT):
        if receipt.exists():
            raise FileExistsError(f"{receipt}: refusing to overwrite a publication receipt")
    if args.stage.exists() and any(args.stage.iterdir()):
        raise FileExistsError(f"{args.stage}: the stage directory must be empty")
    if args.publish:
        verify_source_revision(args.source_revision)
    set_client_factory(lambda: httpx.Client(
        transport=httpx.HTTPTransport(local_address="0.0.0.0"), follow_redirects=True, timeout=60))
    api = HfApi()

    recorded = recorded_releases()
    tags_before, before, live, changed = {}, {}, {}, {}
    for kind, render in (("model", model_files), ("dataset", dataset_files), ("space", space_files)):
        tags_before[kind] = {ref.name: ref.target_commit
                             for ref in api.list_repo_refs(REPOSITORIES[kind], repo_type=kind).tags}
        if TAGS[kind] in tags_before[kind]:
            raise ValueError(f"{kind}: {TAGS[kind]} exists; version tags are immutable")
        before[kind], live[kind] = live_files(api, kind, recorded[kind])
        changed[kind] = render(live[kind])
        for name, data in changed[kind].items():
            path = args.stage / kind / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        for name in sorted(changed[kind]):
            if name in live[kind] and (name == "README.md" or name.endswith(".json")):
                print("".join(difflib.unified_diff(
                    live[kind][name].decode().splitlines(True), changed[kind][name].decode().splitlines(True),
                    f"live/{kind}/{name}", f"licensed/{kind}/{name}", n=0)))
    if not args.publish:
        print(f"Dry run: staged {', '.join(f'{kind}: {sorted(files)}' for kind, files in changed.items())} "
              f"in {args.stage}; nothing uploaded")
        return 0

    expected_inference = recommendations(before["model"])
    after = {}
    for kind in ("model", "dataset", "space"):
        repository = REPOSITORIES[kind]
        commit = api.create_commit(
            repository, repo_type=kind, parent_commit=before[kind],
            operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=data)
                        for name, data in changed[kind].items()],
            commit_message=f"Release under the {LICENSE_NAME} license; no weight, data or application file changes")
        after[kind] = commit.oid
        api.create_tag(repository, repo_type=kind, tag=TAGS[kind], revision=commit.oid, exist_ok=False)

    files_after, verification = {}, {}
    for kind in ("model", "dataset", "space"):
        repository = REPOSITORIES[kind]
        info = HfApi(token=False).repo_info(repository, repo_type=kind, files_metadata=True)
        tags = {ref.name: ref.target_commit for ref in api.list_repo_refs(repository, repo_type=kind).tags}
        card_license = (info.card_data or {}).get("license")
        if (info.private or getattr(info, "gated", False) or info.sha != after[kind] or card_license != LICENSE_HUB_ID
                or tags != {**tags_before[kind], TAGS[kind]: after[kind]}):
            raise ValueError(f"{kind}: visibility, main revision, card license or preserved tags are not as published")
        files_after[kind] = {**recorded[kind]["files"],
                             **{name: record(data) for name, data in changed[kind].items()}}
        siblings = {entry.rfilename: entry for entry in info.siblings if entry.rfilename != ".gitattributes"}
        if set(siblings) != set(files_after[kind]):
            raise ValueError(f"{kind}: the public file inventory is not the intended one")
        for name, entry in siblings.items():
            if entry.lfs is not None and (entry.lfs.sha256, entry.lfs.size) != (
                    files_after[kind][name]["sha256"], files_after[kind][name]["bytes"]):
                raise ValueError(f"{kind}/{name}: an LFS file changed in a license-only update")
        for name, data in {**live[kind], **changed[kind]}.items():
            downloaded = Path(hf_hub_download(repository, name, repo_type=kind, revision=after[kind], token=False,
                                              cache_dir=ROOT / "model/data/hf_cache")).read_bytes()
            if downloaded != data:
                raise ValueError(f"{repository}/{name}: anonymous download differs from the intended release")
        verification[kind] = {"anonymous_inventory_matches": True, "lfs_files_unchanged": sum(
            entry.lfs is not None for entry in siblings.values()), "small_files_downloaded_and_compared": len(
            {**live[kind], **changed[kind]})}
    if recommendations(after["model"]) != expected_inference:
        raise ValueError("anonymous inference changed with a license-only update")
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        wait_for_space(client, SPACE_URL, changed["space"]["manifest.json"])

    verified_at = datetime.now(timezone.utc).isoformat()
    common = {
        "license": LICENSE_NAME, "hub_card_license": LICENSE_HUB_ID, "license_title": LICENSE_TITLE,
        "license_text_sha256": sha256(license_bytes()), "source_code_revision": args.source_revision,
        "earlier_tags_keep_their_published_terms": True, "verified_at": verified_at,
    }

    def section(kind: str) -> dict:
        return {
            "repository": REPOSITORIES[kind], "tag": TAGS[kind], "revision": after[kind],
            "previous_revision": before[kind], "previous_license": PREVIOUS_LICENSES[kind],
            "changed_files": {name: {"before": record(live[kind][name]) if name in live[kind] else None,
                                     "after": record(data)} for name, data in sorted(changed[kind].items())},
            "card_edits": [{"before": old, "after": new} for old, new in CARD_EDITS[kind]],
            "card_generators_reproduce_live_card": True, "files": files_after[kind],
            "preserved_tags": tags_before[kind], "public_verification": verification[kind],
        }

    model = section("model")
    MODEL_RECEIPT.write_text(json.dumps({
        "schema_version": 1, "status": "verified_public_license_update",
        "repo_id": MODEL_REPOSITORY, "url": f"https://huggingface.co/{MODEL_REPOSITORY}",
        **{key: value for key, value in model.items() if key != "repository"}, **common,
        "license_note": WEIGHTS_LICENSE_NOTE, "weights_changed": False,
        "anonymous_inference_identical_before_and_after": True, "inference_contexts": 3,
    }, indent=2) + "\n")
    RELEASE_RECEIPT.write_text(json.dumps({
        "schema_version": 1, "status": "verified_public_license_update", **common,
        "dataset": {**section("dataset"), "license_note": DATASET_LICENSE_NOTE, "data_files_changed": False},
        "space": {**section("space"), "license_note": SPACE_LICENSE_NOTE, "application_files_changed": False,
                  "public_space_serves_manifest": True, "app_url": SPACE_URL},
    }, indent=2) + "\n")
    print(f"Published {TAGS}; receipts {MODEL_RECEIPT.relative_to(ROOT)} and {RELEASE_RECEIPT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
