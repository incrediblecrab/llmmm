"""Export full-corpus ranking weights and measured search evidence, never source recipes."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import sqlite3
import subprocess
import tempfile
from contextlib import closing
from datetime import datetime
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from ingredient_model.config import PATHS
from ingredient_model._hashing import file_sha256 as digest
from ingredient_model.data.recipe_search_metadata import (
    RecipeSearchMetadata, default_search_metadata_path)
from ingredient_model.recipe_ranker import FEATURE_NAMES, RecipeRankingPolicy
from ingredient_model.recipe_ingredients import CanonicalIngredientIndex
from evaluate_recipe_search import _operational_pass
from export_native_model import render_production_card
from train_recipe_ranker import array_digest

POLICY_FILES = ("recipe_ranker_config.json", "recipe_ranker.safetensors")
FILES = {
    "README.md", "recipe_search_config.json", "recipe_training.json",
    "recipe_evaluation.json", "recipe_catalog.json", "recipe_release_policy.json",
    *(f"recipe_policies/{stage}/{name}"
      for stage in ("supervised", "reinforce") for name in POLICY_FILES),
}
PARAMETERS = {
    "actions_per_query", "batch_size", "bootstrap_repetitions", "candidates",
    "entropy_coefficient", "hidden_dim", "kl_coefficient", "max_missing", "max_noise",
    "max_train_rows", "mode", "pool_size", "reinforce_epochs", "reinforce_lr", "seed",
    "supervised_epochs", "supervised_lr", "temperature", "test_queries", "threads",
    "validation_queries",
}


def validate_live_evaluation(report: dict) -> str:
    count = report.get("queries_per_partition")
    selected = report.get("selected_on_validation")
    policies = {"heuristic", "supervised", "reinforce"}
    if (report.get("schema_version") != 1 or type(count) is not int or count < 200
            or selected not in policies or report.get("release_evaluation") is not True
            or report.get("test_scored") is not True
            or report.get("selection_frozen_before_test_scoring") is not True
            or report.get("operational_gate_passed") is not True
            or report.get("training_query_partition_overlap") != 0):
        raise ValueError("release requires a completed held-out live evaluation")
    for phase in ("validation", "test"):
        summaries = report.get(phase)
        if (not isinstance(summaries, dict) or set(summaries) != policies
                or any(summary.get("queries") != count for summary in summaries.values())
                or not _operational_pass(summaries[selected], report["retrieval"]["timeout_seconds"])):
            raise ValueError(f"{phase}: live evaluation failed its population or operational gate")
    return selected


def verify_evaluated_catalog(evaluation: dict, coverage: dict, training: dict,
                             configuration: dict | None = None) -> None:
    expected = evaluation.get("catalog_sha256")
    if (not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected)
            or coverage.get("catalog_sha256") != expected
            or (configuration is not None
                and configuration.get("evaluated_catalog_sha256") != expected)):
        raise ValueError("evaluation and release use different source catalogs")
    if training["time_metadata"]["numeric_time_array_sha256"] != coverage["numeric_time_array_sha256"]:
        raise ValueError("current source times differ from the ranking-training time array")


def verify_source_revision(revision: str) -> None:
    root = Path(__file__).resolve().parents[2]
    for name in (
            "model/ingredient_model/recipe_search.py",
            "model/ingredient_model/recipe_ranker.py",
            "model/ingredient_model/recipe_ingredients.py",
            "model/ingredient_model/data/recipe_search_metadata.py",
            "model/ingredient_model/data/recipe_catalog.py",
            "model/ingredient_model/_hashing.py",
            "model/ingredient_model/hub.py", "model/ingredient_model/config.py",
            "model/ingredient_model/cli.py", "model/pyproject.toml",
            "model/scripts/build_recipe_catalog.py",
            "model/scripts/train_recipe_ranker.py",
            "model/scripts/verify_recipe_ranker.py",
            "model/scripts/evaluate_recipe_search.py",
            "model/scripts/export_recipe_search.py",
            "model/scripts/publish_recipe_search.py"):
        stored = subprocess.run(
            ["git", "show", f"{revision}:{name}"], cwd=root, capture_output=True, check=False)
        if stored.returncode or hashlib.sha256(stored.stdout).hexdigest() != digest(root / name):
            raise ValueError(f"{name}: commit the tested source before exporting a pinned release")


def committed_training_evidence(revision: str, relative: str) -> tuple[dict, str]:
    path = Path(relative)
    if (path.is_absolute() or path.parts[:2] != ("model", "results")
            or ".." in path.parts or path.suffix != ".json"):
        raise ValueError("training evidence must be a tracked model/results JSON file")
    root = Path(__file__).resolve().parents[2]
    saved = subprocess.run(
        ["git", "show", f"{revision}:{path.as_posix()}"], cwd=root, capture_output=True, check=False)
    if saved.returncode:
        raise ValueError("the pinned source commit does not contain the training verification record")
    return json.loads(saved.stdout), hashlib.sha256(saved.stdout).hexdigest()


def _metadata_only(value: object) -> None:
    if isinstance(value, dict):
        forbidden = {"title", "steps", "raw_ingredients", "ingredient_ids", "recipe_ids",
                     "source_rows", "coverage_artifact", "catalog_path"}
        if forbidden & value.keys():
            raise ValueError("public evidence contains a private data field")
        for item in value.values():
            _metadata_only(item)
    elif isinstance(value, list):
        for item in value:
            _metadata_only(item)
    elif isinstance(value, str) and (value.startswith("/") or re.match(r"^[A-Za-z]:\\", value)):
        raise ValueError("public evidence contains an absolute local path")


def public_training_report(report: dict) -> dict:
    public = {name: copy.deepcopy(report[name]) for name in (
        "schema_version", "status", "mode", "completed_utc", "corpus", "population",
        "model", "protocol", "reward", "evaluation", "training",
        "is_full_corpus_optimizer_coverage",
        "process_peak_rss_bytes", "production_or_human_preference_quality_established",
    )}
    for stage in public["training"].values():
        for epoch in stage["epochs"]:
            epoch["coverage"].pop("coverage_artifact", None)
    public["training_totals"] = {
        "queries": sum(stage["n_queries"] for stage in public["training"].values()),
        "optimizer_steps": sum(stage["n_optimizer_steps"] for stage in public["training"].values()),
        "source_ingredient_slots": sum(
            epoch["coverage"]["n_source_ingredient_slots"]
            for stage in public["training"].values() for epoch in stage["epochs"]),
        "reinforce_sampled_actions": sum(
            epoch["bandit"]["sampled_actions"]
            for epoch in public["training"]["reinforce"]["epochs"]),
    }
    public["time_metadata"] = {name: report["time_metadata"][name] for name in (
        "enabled", "n_known", "n_unknown", "numeric_time_array_sha256", "semantics")}
    configuration = report["run_configuration"]
    utc_elapsed = (datetime.fromisoformat(configuration["completed_utc"])
                   - datetime.fromisoformat(configuration["started_utc"])).total_seconds()
    public["timing"] = {
        "perf_counter_seconds": report["total_wall_seconds"],
        "recorded_utc_elapsed_seconds": utc_elapsed,
        "clock_discrepancy_seconds": utc_elapsed - report["total_wall_seconds"],
        "caveat": (
            "The timer and UTC intervals disagree; the cause was not independently established. "
            "Do not describe the perf_counter value as end-to-end calendar wall time."
            if abs(utc_elapsed - report["total_wall_seconds"]) > 1 else
            "Timer and recorded UTC intervals agree within one second."),
    }
    public["run_configuration"] = {
        "parameters": {name: configuration["arguments"][name] for name in sorted(PARAMETERS)},
        "code_sha256": configuration["code_sha256"],
        "environment": configuration["environment"],
        "started_utc": configuration["started_utc"],
        "completed_utc": configuration["completed_utc"],
        "status": configuration["status"],
    }
    _metadata_only(public)
    return public


def verify_training(run: Path, metadata: dict) -> dict:
    report = json.loads((run / "report.json").read_text())
    population, slots = metadata["n_recipes"], metadata["n_slots"]
    if (report.get("status") != "completed" or report.get("mode") != "full"
            or report.get("is_full_corpus_optimizer_coverage") is not True
            or report["corpus"]["sha256"] != metadata["corpus_sha256"]
            or report["corpus"]["n_recipes"] != population
            or report["corpus"]["n_ingredient_slots"] != slots
            or report["population"]["training_source_rows_per_epoch"] != population):
        raise ValueError("release requires completed, corpus-matched all-record policy training")
    if (report["model"]["initialized_from_scratch"] is not True
            or report["model"]["pretrained_parameters_used"] is not False
            or report["model"]["time_features_enabled"] is not True
            or report["time_metadata"]["enabled"] is not True):
        raise ValueError("release requires independently initialized, time-enabled policies")
    if report["protocol"]["validation_test_pantry_contexts_never_used_for_training"] is not True:
        raise ValueError("policy training must exclude validation and test pantry hashes")
    arguments = report["run_configuration"]["arguments"]
    expected_steps = (population + arguments["batch_size"] - 1) // arguments["batch_size"]
    for name in ("supervised", "reinforce"):
        stage = report["training"][name]
        expected_epochs = arguments[f"{name}_epochs"]
        if (expected_epochs < 1 or len(stage["epochs"]) != expected_epochs
                or stage["epochs_completed"] != expected_epochs
                or stage["n_queries"] != population * expected_epochs
                or stage["n_optimizer_steps"] != expected_steps * expected_epochs):
            raise ValueError(f"{name}: incomplete stage counters")
        for number, epoch in enumerate(stage["epochs"], 1):
            coverage = epoch["coverage"]
            if (coverage["every_catalog_row_exactly_once"] is not True
                    or coverage["n_queries"] != population
                    or coverage["unique_source_rows"] != population
                    or coverage["n_source_ingredient_slots"] != slots
                    or coverage["n_optimizer_steps"] != expected_steps):
                raise ValueError(f"{name} epoch {number}: incomplete all-record coverage")
            proof = run / "private_coverage" / f"{name}-epoch-{number:03d}.npz"
            if digest(proof) != coverage["coverage_artifact_sha256"]:
                raise ValueError(f"{name} epoch {number}: changed coverage proof")
            with np.load(proof, allow_pickle=False) as arrays:
                counts = arrays["seen_counts"]
                if (counts.shape != (population,) or counts.dtype != np.uint8
                        or not np.all(counts == 1)
                        or int(arrays["n_corpus_rows"]) != population):
                    raise ValueError(f"{name} epoch {number}: not every row was seen exactly once")
            buckets = epoch["query_partition_bucket_counts"]
            if len(buckets) != 10 or sum(buckets) != population or any(buckets[8:]):
                raise ValueError(f"{name} epoch {number}: held-out query leakage")
            if name == "reinforce":
                if epoch["bandit"]["sampled_actions"] != population * arguments["actions_per_query"]:
                    raise ValueError("REINFORCE sampled-action count does not match its declared budget")
    checkpoints = report.get("checkpoint_files_sha256")
    binding = {"capture": "training_report_before_test_scoring"}
    if checkpoints is None:
        audit_path = run / "aggregate_evidence.json"
        audit = json.loads(audit_path.read_text())
        if (audit.get("canonical_corpus_sha256") != metadata["corpus_sha256"]
                or audit["artifact_sha256"].get("report.json") != digest(run / "report.json")):
            raise ValueError("post-training audit does not bind this completed training report")
        checkpoints = {
            stage: {filename: audit["artifact_sha256"][f"{stage}/{filename}"]
                    for filename in POLICY_FILES} for stage in ("supervised", "reinforce")}
        binding = {
            "capture": "independent_post_training_audit",
            "audited_utc": audit["audited_utc"], "audit_sha256": digest(audit_path),
            "limitation": "These hashes were captured in the independent post-run audit, not inside the original training process.",
        }
    if set(checkpoints) != {"supervised", "reinforce"}:
        raise ValueError("completed training must bind both learned checkpoints")
    for stage, files in checkpoints.items():
        if set(files) != set(POLICY_FILES):
            raise ValueError("unexpected completed checkpoint inventory")
        for filename, expected in files.items():
            if digest(run / stage / filename) != expected:
                raise ValueError(f"{stage}/{filename}: checkpoint differs from completed training evidence")
    public = public_training_report(report)
    public["checkpoint_binding"] = {
        **binding, "training_report_sha256": digest(run / "report.json"),
        "checkpoint_files_sha256": checkpoints,
    }
    return public


def training_code_references(hashes: dict[str, str]) -> dict:
    if set(hashes) != {"recipe_ranker.py", "train_recipe_ranker.py"}:
        raise ValueError("unexpected policy training source inventory")
    root = Path(__file__).resolve().parents[2]
    references = {}
    for name, expected in hashes.items():
        relative = f"model/{'scripts' if name.startswith('train_') else 'ingredient_model'}/{name}"
        if name == "recipe_ranker.py" and digest(root / relative) != expected:
            raise ValueError("current ranking implementation differs from the trained feature/model code")
        revisions = subprocess.check_output(
            ["git", "log", "-50", "--format=%H", "--", relative], cwd=root, text=True).splitlines()
        for revision in revisions:
            saved = subprocess.run(
                ["git", "show", f"{revision}:{relative}"], cwd=root, capture_output=True, check=False)
            if saved.returncode == 0 and hashlib.sha256(saved.stdout).hexdigest() == expected:
                references[name] = {"sha256": expected, "git_revision": revision, "path": relative}
                break
        else:
            raise ValueError(f"{name}: the exact recorded training source is absent from git history")
    return references


def catalog_evidence(catalog: Path, corpus: Path) -> tuple[dict, dict]:
    RecipeSearchMetadata.load(catalog)
    with closing(sqlite3.connect(catalog.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        metadata = {key: json.loads(value)
                    for key, value in connection.execute("SELECT key,value FROM metadata")}
        connection.create_function(
            "readable_recipe", 3,
            lambda title, steps, status: int(bool(title.strip()) and bool(steps.strip())
                                            and status != "unparsed_steps"),
            deterministic=True)
        counts = connection.execute(
            "SELECT count(*),sum(length(ingredient_ids)/2),"
            "sum(readable_recipe(title,steps,text_status)),sum(total_minutes IS NOT NULL),"
            "sum(servings IS NOT NULL),"
            "sum(total_minutes IS NOT NULL AND readable_recipe(title,steps,text_status)),"
            "sum(total_minutes IS NOT NULL AND (total_minutes<=0 OR time_status!='source_total')) "
            "FROM recipes").fetchone()
    if (metadata.get("partial") is not False or counts[:2] != (
            metadata["n_recipes"], metadata["n_slots"]) or counts[6] != 0):
        raise ValueError("catalog population or source-total provenance failed verification")
    CanonicalIngredientIndex.load(
        corpus, corpus_sha256=metadata["corpus_sha256"], n_recipes=metadata["n_recipes"],
        n_slots=metadata["n_slots"], n_vocab=len(metadata["vocabulary"]))
    evidence = {
        "schema_version": 1, "corpus_sha256": metadata["corpus_sha256"],
        "catalog_sha256": digest(catalog),
        "text_index_sha256": metadata["text_index_sha256"], "recipes": counts[0],
        "ingredient_slots": counts[1], "vocabulary_size": len(metadata["vocabulary"]),
        "readable_title_and_steps": counts[2], "known_source_total_minutes": counts[3],
        "unknown_source_total_minutes": counts[0] - counts[3],
        "known_servings": counts[4], "timed_readable_recipes": counts[5],
        "readable_definition": "nonempty Python-stripped title and steps; unparsed_steps excluded",
        "time_definition": "source-provided positive totals only; no prep+cook substitution",
        "metadata_index_catalog_hash_verified": True,
        "canonical_ingredient_corpus_hash_verified": True,
        "numeric_time_array_sha256": array_digest(np.load(
            default_search_metadata_path(catalog) / "total_minutes.npy",
            allow_pickle=False, mmap_mode="r")),
        "source_recipes_included_in_release": False,
    }
    return metadata, evidence


def render_card(native: dict, config: dict, training: dict, evaluation: dict,
                coverage: dict) -> str:
    native = {**native, "source_code_revision": config["source_code_revision"],
              "tag": config["tag"]}
    card = render_production_card(native, public=True)
    selected = evaluation["selected_on_validation"]
    score = evaluation["test"][selected]
    epochs = training["training"]
    presentations = sum(stage["n_queries"] for stage in epochs.values())
    actions = sum(epoch["bandit"]["sampled_actions"] for epoch in epochs["reinforce"]["epochs"])
    choice = (
        "The heuristic remains the default: learned ranking did not establish a validation gain. "
        "Both trained policies are included for reproducible experiments."
        if selected == "heuristic" else
        f"The `{selected}` policy is the default, selected on validation before test scoring."
    )
    if selected == "supervised":
        choice += " REINFORCE did not establish an additional paired validation gain."
        failures = evaluation["validation"]["reinforce"]["timeouts"]
        if failures:
            noun = "request" if failures == 1 else "requests"
            choice += (
                f" {failures} RL validation {noun} timed out; "
                "the cause was not isolated.")
    comparison = evaluation["paired_test_against_heuristic"].get(selected)
    paired = ""
    if comparison is not None:
        low, high = comparison["ci95"]
        paired = (
            f"Compared with the heuristic, top-five source-set recovery changed by "
            f"**{100 * comparison['difference_in_recall_at_5']:+.1f} percentage points** "
            f"(paired query-bootstrap CI95: {100 * low:.1f} to {100 * high:.1f} points).\n\n")
    section = f"""## Finding recipes

This release also includes a constrained recipe finder and our own
{training['model']['learned_parameter_count']:,}-parameter ranking policies. It retrieves existing
recipes; it does not write new ones. Supply canonical ingredient names, a maximum
source-reported total time, required or excluded ingredients, missing-item limits,
servings, and an optional source-language code.

**Recipe search requires an authorized local catalog, its verified metadata
index, and the canonical `recipe_ids.npz` ingredient corpus. These data files
are not included in this public model.** The catalog contains
{coverage['recipes']:,} records, but only {coverage['known_source_total_minutes']:,}
have a known source total time. Unknown times do not pass a time limit.
The repository's catalog builder reconstructs these files from authorized local
sources; [recipe_catalog.json](recipe_catalog.json) records the measured coverage.

```python
from ingredient_model.recipe_search import RecipeFinder, RecipeQuery

finder = RecipeFinder.from_pretrained(
    "{config['repo_id']}",
    revision="{config['tag']}",
    catalog_path="/private/recipe_search.sqlite",
    corpus_path="/private/recipe_ids.npz",
    token=False,
)
result = finder.search(RecipeQuery(
    ["chicken", "rice", "broccoli"],
    must_use=["chicken"],
    max_total_minutes=30,
    max_missing=2,
))
for recipe in result.recipes:
    print(recipe.title, recipe.total_minutes, recipe.missing_ingredients, recipe.source_url)
```

Returned records include source ingredients, separate quantity values, instructions,
and warnings. Normalized ingredient matches are not a complete shopping list or an
allergen-safety check. Missing units are not invented, inconsistent quantity arrays
are not paired, and serving counts do not scale quantities or cooking time.

### Ranking training and measured limits

The policies were initialized from scratch. Listwise supervised learning was
followed by sampled-action REINFORCE with entropy regularization and a KL penalty
to the supervised policy. Each stage used all {coverage['recipes']:,} canonical records.
Together, the stages produced **{presentations:,} training queries** and
**{actions:,} sampled reinforcement actions**.
The reward is recovery of the source's canonical ingredient set, not human feedback.
The source is deliberately inserted into sampled training/evaluation candidate sets;
those sampled-ranking scores are not full-catalog search accuracy.
[recipe_training.json](recipe_training.json) records coverage and the complete protocol.

{choice}

The separate live search evaluation did not insert the source recipe into retrieval.
On {score['queries']:,} held-out test pantry queries, the selected policy recovered
the source ingredient set in its top five **{score['source_set_recall_at_5']:.1%}**
of the time. The source set reached the shortlist on
**{score['source_set_in_shortlist']:.1%}** of queries. Median request time was
**{score['latency_ms']['median'] / 1000:.2f}s**, with p95
**{score['latency_ms']['p95'] / 1000:.2f}s**;
{score['timeouts']} timeouts and {score['constraint_violations']} constraint violations
were observed. Full counts, failures, truncation and paired comparisons are in
[recipe_evaluation.json](recipe_evaluation.json).

{paired}{score['truncated_retrievals']} of {score['queries']} test queries reached a retrieval
budget. Request timings exclude initialization; this is a measured local run,
not a service-level guarantee.

Validation and test pantry hashes are excluded from ranking-policy training.
Recipes and duplicate families are not held out. These synthetic recovery
measurements do not establish taste, cooking quality, unseen-recipe generalization
or a service-level guarantee. Retrieval is bounded and reports when its shortlist
or scan budget is reached. There is no silent fallback between learned and heuristic ranking.

"""
    card = card.replace("- ingredient-completion\n", "- ingredient-completion\n- recipe-retrieval\n")
    card = card.replace("## Training evidence\n", section + "## Ingredient-model training evidence\n", 1)
    card = card.replace("## Evaluation status\n", "## Ingredient-model evaluation status\n", 1)
    card = card.replace("## Usage\n", "## Ingredient-predictor usage\n", 1)
    card = card.replace("Inference does not require the training corpus.",
                        "Ingredient prediction does not require the training corpus.", 1)
    card = card.replace("## Acknowledgements\n", """## Acknowledgements

The ranking experiment uses [REINFORCE](https://doi.org/10.1007/BF00992696)
(Ronald J. Williams), with a supervised warm-start and KL regularization.
[SQLite FTS5](https://sqlite.org/fts5.html) supplies the ingredient-token retrieval
index. These are method and implementation credits, not imported learned weights.
""", 1)
    card = card.replace("## Ideas for using this model\n", """## Ideas for using this model

- Build a pantry search over an authorized recipe collection, showing source links,
  reported cooking times, missing canonical ingredients and data-quality warnings.
- Compare supervised, reinforcement-trained and heuristic rankings on held-out
  queries. Keep the simpler policy when measured gains are not established.
""", 1)
    return card


def export(args) -> dict:
    if args.out.exists():
        raise FileExistsError(f"{args.out}: exports are immutable")
    verify_source_revision(args.source_revision)
    metadata, coverage = catalog_evidence(args.catalog, args.corpus)
    training = verify_training(args.run, metadata)
    root = Path(__file__).resolve().parents[2]
    evidence_relative = args.training_evidence.resolve().relative_to(root).as_posix()
    evidence, evidence_hash = committed_training_evidence(args.source_revision, evidence_relative)
    if (digest(args.training_evidence) != evidence_hash
            or evidence.get("checkpoint_binding") != training["checkpoint_binding"]):
        raise ValueError("checkpoint binding differs from the committed full-training verification record")
    training["committed_evidence"] = {"path": evidence_relative, "sha256": evidence_hash}
    evaluation = json.loads(args.evaluation.read_text())
    selected = validate_live_evaluation(evaluation)
    if evaluation.get("corpus_sha256") != metadata["corpus_sha256"]:
        raise ValueError("release requires a successful, corpus-matched held-out live evaluation")
    verify_evaluated_catalog(evaluation, coverage, training)
    source = Path(__file__).resolve().parents[1]
    for field, path in (
            ("code_sha256", Path(__file__).with_name("evaluate_recipe_search.py")),
            ("serving_code_sha256", source / "ingredient_model" / "recipe_search.py"),
            ("metadata_index_code_sha256",
             source / "ingredient_model" / "data" / "recipe_search_metadata.py"),
            ("ingredient_index_code_sha256", source / "ingredient_model" / "recipe_ingredients.py")):
        if evaluation[field] != digest(path):
            raise ValueError("live evaluation does not cover the current serving/evaluation code")
    training["training_code_references"] = training_code_references(
        training["run_configuration"]["code_sha256"])
    inventory = {}
    for name in ("supervised", "reinforce"):
        inventory[name] = {file: digest(args.run / name / file) for file in POLICY_FILES}
        if (evaluation["policy_files_sha256"].get(name) != inventory[name]
                or training["checkpoint_binding"]["checkpoint_files_sha256"][name] != inventory[name]):
            raise ValueError(f"{name}: live evaluation used different policy files")
    policy_name = selected if selected != "heuristic" else "reinforce"
    config = {
        "schema_version": 1, "repo_id": args.repo_id, "tag": args.tag,
        "source_code_revision": args.source_revision,
        "corpus_sha256": metadata["corpus_sha256"],
        "evaluated_catalog_sha256": coverage["catalog_sha256"],
        "vocabulary_sha256": hashlib.sha256(json.dumps(
            metadata["vocabulary"], ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
        "deployed_ranker": "heuristic" if selected == "heuristic" else "learned",
        "selected_policy": selected, "requires_metadata_index": True,
        "requires_corpus_index": True,
        "search_defaults": evaluation["retrieval"],
        "policy_directory": f"recipe_policies/{policy_name}",
        "policy_files": inventory[policy_name], "available_policy_files": inventory,
        "feature_names": list(FEATURE_NAMES),
        "catalog_required": True, "recipe_generation_supported": False,
    }
    release_policy = {
        "schema_version": 1, "visibility": "public", "source_recipe_data_included": False,
        "private_query_and_coverage_arrays_included": False,
        "external_pretrained_parameters_used": False,
        "license": "No permissive weights license is granted; review source terms for downstream use.",
        "native_ingredient_checkpoint": "unchanged",
    }
    _metadata_only(evaluation)
    native = json.loads(args.native_evidence.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".recipe-export-", dir=args.out.parent) as temporary:
        prepared = Path(temporary) / "package"
        prepared.mkdir()
        reload_count = 64
        features = np.random.default_rng(41).uniform(
            size=(reload_count, len(FEATURE_NAMES))).astype(np.float32)
        fresh_dir = Path(temporary) / "fresh-initialization"
        RecipeRankingPolicy(
            hidden_dim=training["model"]["hidden_dim"],
            seed=training["run_configuration"]["parameters"]["seed"],
            time_features_enabled=True).save(fresh_dir)
        fresh_tensors = load_file(str(fresh_dir / POLICY_FILES[1]))
        for name in ("supervised", "reinforce"):
            destination = prepared / "recipe_policies" / name
            destination.mkdir(parents=True)
            for file in POLICY_FILES:
                shutil.copyfile(args.run / name / file, destination / file)
            original = RecipeRankingPolicy.load(args.run / name)
            restored = RecipeRankingPolicy.load(destination)
            np.testing.assert_array_equal(original.score(features), restored.score(features))
            tensors = load_file(str(destination / POLICY_FILES[1]))
            if not any(not np.array_equal(tensors[key], fresh_tensors[key]) for key in tensors):
                raise ValueError(f"{name}: checkpoint did not change from fresh initialization")
        config["export_verification"] = {
            "reload_feature_vectors_per_policy": reload_count,
            "exact_scores_after_reload": True, "both_policies_changed_from_initialization": True,
        }
        for name, value in (
                ("recipe_search_config.json", config), ("recipe_training.json", training),
                ("recipe_evaluation.json", evaluation), ("recipe_catalog.json", coverage),
                ("recipe_release_policy.json", release_policy)):
            (prepared / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        (prepared / "README.md").write_text(render_card(native, config, training, evaluation, coverage))
        found = {str(path.relative_to(prepared)) for path in prepared.rglob("*") if path.is_file()}
        if found != FILES:
            raise ValueError("export inventory differs from the allowed recipe package")
        prepared.rename(args.out)
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=PATHS.recipes / "recipe_search.sqlite")
    parser.add_argument("--corpus", type=Path, default=PATHS.recipes / "recipe_ids.npz")
    parser.add_argument("--native-evidence", type=Path, default=PATHS.results / "public_model_export.json")
    parser.add_argument("--training-evidence", type=Path,
                        default=PATHS.results / "recipe_ranker_training.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=PATHS.results / "recipe_search_export.json")
    parser.add_argument("--repo-id", default="incrediblecrab/llmmm-recipes")
    parser.add_argument("--tag", default="v0.4.0-recipe-search")
    parser.add_argument("--source-revision")
    args = parser.parse_args()
    args.source_revision = args.source_revision or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()
    if not re.fullmatch(r"[a-f0-9]{40}", args.source_revision):
        parser.error("--source-revision must be a full git commit SHA")
    if args.report.exists():
        raise FileExistsError(f"{args.report}: export receipts are immutable")
    if args.out.resolve() in args.report.resolve().parents:
        parser.error("--report must be outside the export package")
    config = export(args)
    receipt = {
        "schema_version": 1, "status": "verified_local_recipe_search_export_not_published",
        "repo_id": config["repo_id"], "tag": config["tag"],
        "source_code_revision": config["source_code_revision"],
        "selected_policy": config["selected_policy"],
        "files": {name: {"sha256": digest(args.out / name),
                         "bytes": (args.out / name).stat().st_size} for name in sorted(FILES)},
        "catalog": json.loads((args.out / "recipe_catalog.json").read_text()),
        "training": json.loads((args.out / "recipe_training.json").read_text()),
        "evaluation": json.loads((args.out / "recipe_evaluation.json").read_text()),
        "reload_verification": config["export_verification"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"export": str(args.out), "selected_policy": config["selected_policy"],
                      "tag": config["tag"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
