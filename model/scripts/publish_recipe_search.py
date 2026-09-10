"""Publish the allowlisted recipe-search extension while preserving native weights."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download, set_client_factory
from huggingface_hub.utils import disable_progress_bars

from ingredient_model.config import PATHS
from ingredient_model.hub import IngredientPredictor
from ingredient_model.recipe_ranker import FEATURE_NAMES, RecipeRankingPolicy
from ingredient_model.recipe_search import RecipeFinder, RecipeQuery
from export_recipe_search import (
    FILES, POLICY_FILES, _metadata_only, committed_training_evidence, digest, validate_live_evaluation,
    verify_evaluated_catalog, verify_source_revision)

NATIVE_FILES = {
    "config.json", "model.safetensors", "training_manifest.json",
    "training_verification.json", "release_policy.json",
}

SMOKE_SCRIPT = """
import json,sys
from pathlib import Path
installation=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(installation))
import httpx,numpy as np
from huggingface_hub import hf_hub_download,set_client_factory
from ingredient_model.config import PATHS
from ingredient_model.hub import IngredientPredictor
from ingredient_model.recipe_ranker import FEATURE_NAMES,RecipeRankingPolicy
from ingredient_model.recipe_search import RecipeFinder,RecipeQuery
assert not PATHS.data.exists()
repo,revision,cache,catalog,corpus,contexts,bundle=sys.argv[2:9]
set_client_factory(lambda:httpx.Client(
    transport=httpx.HTTPTransport(local_address='0.0.0.0'),follow_redirects=True,timeout=60))
model=IngredientPredictor.from_pretrained(repo,revision=revision,cache_dir=cache,token=False)
native=[model.recommend(context,top_k=5) for context in json.loads(contexts)]
features=np.random.default_rng(301).uniform(size=(64,len(FEATURE_NAMES))).astype(np.float32)
policies={}
for name in ('supervised','reinforce'):
    if bundle:
        policy_dir=Path(bundle)/'recipe_policies'/name
    else:
        for filename in ('recipe_ranker_config.json','recipe_ranker.safetensors'):
            path=hf_hub_download(repo,f'recipe_policies/{name}/{filename}',
                                 revision=revision,cache_dir=cache,token=False)
        policy_dir=Path(path).parent
    policies[name]=RecipeRankingPolicy.load(policy_dir).score(features).tolist()
finder=RecipeFinder.from_pretrained(bundle or repo,revision=None if bundle else revision,
                                    cache_dir=cache,token=False,
                                    catalog_path=catalog,corpus_path=corpus)
result=finder.search(RecipeQuery(['chicken','rice','broccoli'],must_use=['chicken'],
                                max_total_minutes=30,max_missing=2,top_k=5))
for name,module in tuple(sys.modules.items()):
    if name in ('ingredient_model','models') or name.startswith(('ingredient_model.','models.')):
        location=getattr(module,'__file__',None)
        if location is None or not Path(location).resolve().is_relative_to(installation):
            raise RuntimeError(f'{name} was not loaded from the isolated public source package')
print(json.dumps({'native':native,'policies':policies,
                  'search':[[recipe.recipe_id,recipe.score] for recipe in result.recipes]}))
"""


def isolated_environment(root: Path) -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith("GIT_")
        and key not in ("GH_TOKEN", "GITHUB_TOKEN", "PYTHONPATH", "PYTHONSTARTUP")
    }
    environment.update({
        "IM_DATA": str(root / "absent-data"), "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0", "PIP_CONFIG_FILE": os.devnull,
        "PIP_NO_INPUT": "1", "PYTHONNOUSERSITE": "1",
    })
    return environment


def install_public_source(revision: str, destination: Path, environment: dict[str, str]) -> None:
    destination.mkdir()
    probe_code = """
import sys
from importlib.machinery import PathFinder
if PathFinder.find_spec('ingredient_model', [sys.argv[1]]) is None:
    raise ModuleNotFoundError('isolated ingredient_model package is missing')
"""
    probe = subprocess.run(
        [sys.executable, "-I", "-c", probe_code, str(destination)],
        cwd=destination.parent, env=environment, capture_output=True, text=True)
    if probe.returncode != 1 or "ModuleNotFoundError" not in probe.stderr:
        raise ValueError("the isolated package directory was not empty or its import probe failed unexpectedly")
    requirement = (
        "ingredient-model @ git+https://github.com/incrediblecrab/llmmm.git@"
        f"{revision}#subdirectory=model")
    installed = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
         "--no-index", "--no-deps", "--no-build-isolation", "--target", str(destination), requirement],
        cwd=destination.parent, env=environment, capture_output=True, text=True, timeout=300)
    if installed.returncode != 0:
        raise RuntimeError(
            "anonymous installation of the pinned public source failed:\n" + installed.stderr[-16000:])


def isolated_smoke(installation: Path, environment: dict[str, str], *, repo: str,
                   revision: str, cache: Path, catalog: Path, corpus: Path,
                   contexts: list[list[str]], bundle: Path | None = None) -> dict:
    process = subprocess.run(
        [sys.executable, "-I", "-c", SMOKE_SCRIPT, str(installation), repo, revision,
         str(cache), str(catalog.resolve()), str(corpus.resolve()), json.dumps(contexts),
         "" if bundle is None else str(bundle.resolve())],
        cwd=installation.parent, env=environment, capture_output=True, text=True, timeout=300)
    if process.returncode != 0:
        raise RuntimeError("isolated public-source inference failed:\n" + process.stderr[-16000:])
    return json.loads(process.stdout)


def read_package(folder: Path) -> tuple[dict, dict]:
    paths = [path for path in folder.rglob("*") if path.is_file()]
    if ({str(path.relative_to(folder)) for path in paths} != FILES
            or any(path.is_symlink() for path in paths)):
        raise ValueError("recipe export must contain only the allowed regular package files")
    for path in paths:
        if path.stat().st_size > 2_000_000:
            raise ValueError(f"{path.name}: recipe package file exceeds its bounded format")
    config = json.loads((folder / "recipe_search_config.json").read_text())
    if not isinstance(config.get("source_code_revision"), str) or not re.fullmatch(
            r"[a-f0-9]{40}", config["source_code_revision"]):
        raise ValueError("recipe package must pin a full source commit SHA")
    training = json.loads((folder / "recipe_training.json").read_text())
    evaluation = json.loads((folder / "recipe_evaluation.json").read_text())
    coverage = json.loads((folder / "recipe_catalog.json").read_text())
    release_policy = json.loads((folder / "recipe_release_policy.json").read_text())
    validate_live_evaluation(evaluation)
    for report in (config, training, evaluation, coverage, release_policy):
        _metadata_only(report)
    if (config.get("schema_version") != 1 or config.get("requires_metadata_index") is not True
            or config.get("requires_corpus_index") is not True
            or training.get("is_full_corpus_optimizer_coverage") is not True
            or training.get("status") != "completed" or training.get("mode") != "full"
            or evaluation.get("operational_gate_passed") is not True
            or evaluation.get("release_evaluation") is not True
            or config.get("selected_policy") != evaluation.get("selected_on_validation")
            or release_policy.get("visibility") != "public"
            or release_policy.get("source_recipe_data_included") is not False):
        raise ValueError("recipe package lacks verified training, deployment or publication evidence")
    selected = config["selected_policy"]
    policy_name = "reinforce" if selected == "heuristic" else selected
    if (selected not in ("heuristic", "supervised", "reinforce")
            or config["deployed_ranker"] != ("heuristic" if selected == "heuristic" else "learned")
            or config["policy_directory"] != f"recipe_policies/{policy_name}"
            or config["policy_files"] != config["available_policy_files"][policy_name]
            or config["search_defaults"] != evaluation["retrieval"]
            or training["model"]["time_features_enabled"] is not True
            or evaluation.get("training_query_partition_overlap") != 0):
        raise ValueError("recipe package deployment differs from its evaluated policy")
    if len({config["corpus_sha256"], coverage["corpus_sha256"],
            evaluation["corpus_sha256"], training["corpus"]["sha256"]}) != 1:
        raise ValueError("recipe package components use different corpus identities")
    verify_evaluated_catalog(evaluation, coverage, training, config)
    for stage in ("supervised", "reinforce"):
        folder_hashes = {name: digest(folder / "recipe_policies" / stage / name)
                         for name in POLICY_FILES}
        if (folder_hashes != config["available_policy_files"][stage]
                or folder_hashes != evaluation["policy_files_sha256"][stage]
                or folder_hashes != training["checkpoint_binding"]["checkpoint_files_sha256"][stage]):
            raise ValueError(f"{stage}: policy bytes differ from the evaluated package")
        RecipeRankingPolicy.load(folder / "recipe_policies" / stage)
    return config, evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=PATHS.recipes / "recipe_search.sqlite")
    parser.add_argument("--corpus", type=Path, default=PATHS.recipes / "recipe_ids.npz")
    parser.add_argument("--native-receipt", type=Path,
                        default=PATHS.results / "huggingface_public_release.json")
    parser.add_argument("--out", type=Path,
                        default=PATHS.results / "huggingface_recipe_search_release.json")
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()
    if not args.public:
        parser.error("--public is required for this public recipe-search release")
    if args.out.exists():
        raise FileExistsError(f"{args.out}: publication receipts are immutable")
    config, evaluation = read_package(args.folder)
    verify_source_revision(config["source_code_revision"])
    training = json.loads((args.folder / "recipe_training.json").read_text())
    committed = training["committed_evidence"]
    evidence, evidence_hash = committed_training_evidence(
        config["source_code_revision"], committed["path"])
    if (evidence_hash != committed["sha256"]
            or evidence["checkpoint_binding"] != training["checkpoint_binding"]):
        raise ValueError("published checkpoint provenance differs from the committed verification record")
    if digest(args.catalog) != config["evaluated_catalog_sha256"]:
        raise ValueError("publication catalog differs from the evaluated source catalog")
    native_receipt = json.loads(args.native_receipt.read_text())
    repo, tag = config["repo_id"], config["tag"]
    if native_receipt["repo_id"] != repo:
        raise ValueError("native preservation receipt refers to a different repository")
    disable_progress_bars()
    set_client_factory(lambda: httpx.Client(
        transport=httpx.HTTPTransport(local_address="0.0.0.0"),
        follow_redirects=True, timeout=60))
    api = HfApi()
    before = api.model_info(repo)
    if before.private or before.gated:
        raise ValueError("recipe extension requires the existing public, ungated model")
    tags = {ref.name: ref.target_commit for ref in api.list_repo_refs(repo).tags}
    if tag in tags:
        raise ValueError("version tags are immutable; choose a new version")
    resolved_tags = {name: api.model_info(repo, revision=name).sha for name in tags}
    previous_files = set(api.list_repo_files(repo, revision=before.sha))
    if (previous_files - NATIVE_FILES - FILES - {".gitattributes"}
            or not NATIVE_FILES <= previous_files):
        raise ValueError("inspect unexpected or missing files before extending the remote model")
    for name in sorted(NATIVE_FILES):
        downloaded = Path(hf_hub_download(repo, name, revision=before.sha, token=False))
        if digest(downloaded) != native_receipt["files"][name]["sha256"]:
            raise ValueError(f"{name}: existing native artifact differs from its preservation receipt")
    native = IngredientPredictor.from_pretrained(repo, revision=before.sha, token=False)
    contexts = [list(native.vocabulary[:2]), list(native.vocabulary[-2:])]
    expected_native = [native.recommend(context, top_k=5) for context in contexts]
    query = RecipeQuery(
        ["chicken", "rice", "broccoli"], must_use=["chicken"],
        max_total_minutes=30, max_missing=2, top_k=5)
    local = RecipeFinder.from_pretrained(
        str(args.folder), catalog_path=args.catalog, corpus_path=args.corpus)
    result = local.search(query)
    if not result.recipes:
        raise ValueError("the local package did not return the required real-catalog smoke query")
    expected_search = [[recipe.recipe_id, recipe.score] for recipe in result.recipes]
    feature_cases = np.random.default_rng(301).uniform(size=(64, len(FEATURE_NAMES))).astype(np.float32)
    expected_policies = {
        name: RecipeRankingPolicy.load(args.folder / "recipe_policies" / name).score(
            feature_cases).tolist() for name in ("supervised", "reinforce")}
    hashes = {name: {"bytes": (args.folder / name).stat().st_size,
                     "sha256": digest(args.folder / name)} for name in sorted(FILES)}
    expected_inference = {
        "native": expected_native, "policies": expected_policies, "search": expected_search}
    with tempfile.TemporaryDirectory(prefix="llmmm-recipe-public-") as temporary:
        temporary = Path(temporary)
        environment = isolated_environment(temporary)
        installation = temporary / "installed-source"
        install_public_source(config["source_code_revision"], installation, environment)
        before_upload = isolated_smoke(
            installation, environment, repo=repo, revision=before.sha,
            cache=temporary / "before-upload-cache", catalog=args.catalog, corpus=args.corpus,
            contexts=contexts, bundle=args.folder)
        if before_upload != expected_inference:
            raise ValueError("the publicly installable source differs from the tested local implementation")
        commit = api.create_commit(
            repo, parent_commit=before.sha,
            operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=args.folder / name)
                        for name in sorted(FILES)],
            commit_message=f"Publish {tag}: constrained recipe search and evaluated ranking policies")
        if set(api.list_repo_files(repo, revision=commit.oid)) - {".gitattributes"} != FILES | NATIVE_FILES:
            raise ValueError("remote inventory differs from the native-plus-search allowlist")
        for name in sorted(FILES | NATIVE_FILES):
            downloaded = Path(hf_hub_download(repo, name, revision=commit.oid, token=False))
            expected = hashes[name] if name in FILES else native_receipt["files"][name]
            if downloaded.stat().st_size != expected["bytes"] or digest(downloaded) != expected["sha256"]:
                raise ValueError(f"{name}: remote bytes changed during publication")
        actual = isolated_smoke(
            installation, environment, repo=repo, revision=commit.oid,
            cache=temporary / "fresh-after-upload-cache", catalog=args.catalog, corpus=args.corpus,
            contexts=contexts)
        if actual != expected_inference:
            raise ValueError("fresh anonymous inference differs from the verified local package")
    api.create_tag(repo, tag=tag, revision=commit.oid, exist_ok=False)
    after = HfApi(token=False).model_info(repo, revision=tag)
    current_tags = {ref.name: ref.target_commit for ref in api.list_repo_refs(repo).tags}
    if (after.private or after.gated or after.sha != commit.oid
            or any(current_tags.get(name) != reference for name, reference in tags.items())
            or api.model_info(repo).sha != commit.oid):
        raise ValueError("public visibility, current revision or preserved tags changed unexpectedly")
    receipt = {
        "schema_version": 1, "repo_id": repo, "url": f"https://huggingface.co/{repo}",
        "tag": tag, "revision": commit.oid, "previous_revision": before.sha,
        "source_code_revision": config["source_code_revision"],
        "status": "verified_public_recipe_search_release", "private": False, "gated": False,
        "selected_policy": config["selected_policy"],
        "files": hashes, "preserved_native_files": {
            name: native_receipt["files"][name] for name in sorted(NATIVE_FILES)},
        "preserved_tags": resolved_tags, "preserved_tag_refs": tags,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "verification": {
            "native_artifact_bytes_unchanged": True, "source_recipes_uploaded": False,
            "anonymous_native_inference_without_corpus": True,
            "anonymous_policy_reload_without_corpus": True,
            "anonymous_model_download_and_search_with_authorized_local_catalog": True,
            "anonymous_pinned_source_installation": True,
            "inference_from_isolated_public_source_before_and_after_upload": True,
            "native_contexts_matched": len(contexts), "policy_feature_vectors_matched": 64,
            "real_recipe_results_matched": len(expected_search),
            "operational_gate_passed": evaluation["operational_gate_passed"],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Verified public release: {receipt['url']}/tree/{tag}")
    print(f"Revision: {commit.oid}; receipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
