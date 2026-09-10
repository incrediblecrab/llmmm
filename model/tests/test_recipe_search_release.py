from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def exporter(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("export_recipe_search")


@pytest.fixture
def complete_run(tmp_path, exporter):
    run = tmp_path / "run"
    proof_dir = run / "private_coverage"
    proof_dir.mkdir(parents=True)
    arguments = {name: 1 for name in exporter.PARAMETERS}
    arguments.update(batch_size=2, actions_per_query=2, mode="full", max_train_rows=0,
                     corpus="/private/corpus", catalog="/private/catalog",
                     output="/private/output", index_cache="/private/index")
    stages = {}
    checkpoints = {}
    for name in ("supervised", "reinforce"):
        checkpoint_dir = run / name
        checkpoint_dir.mkdir()
        checkpoints[name] = {}
        for filename in exporter.POLICY_FILES:
            path = checkpoint_dir / filename
            path.write_bytes(f"unit-test-{name}-{filename}".encode())
            checkpoints[name][filename] = exporter.digest(path)
        proof = proof_dir / f"{name}-epoch-001.npz"
        np.savez(proof, seen_counts=np.ones(4, dtype=np.uint8),
                 eligible_bitmap=np.asarray([15], dtype=np.uint8),
                 n_corpus_rows=np.int64(4))
        stages[name] = {
            "epochs_completed": 1, "n_queries": 4, "n_optimizer_steps": 2,
            "epochs": [{
                "coverage": {
                    "every_catalog_row_exactly_once": True, "n_queries": 4,
                    "unique_source_rows": 4, "n_source_ingredient_slots": 12,
                    "n_optimizer_steps": 2, "coverage_artifact": str(proof),
                    "coverage_artifact_sha256": exporter.digest(proof),
                },
                "query_partition_bucket_counts": [4] + [0] * 9,
                "bandit": {"sampled_actions": 8} if name == "reinforce" else None,
            }],
        }
    report = {
        "schema_version": 1, "status": "completed", "mode": "full",
        "checkpoint_files_sha256": checkpoints,
        "completed_utc": "2026-09-10T00:00:00+00:00",
        "corpus": {"sha256": "a" * 64, "n_recipes": 4, "n_ingredient_slots": 12},
        "population": {"training_source_rows_per_epoch": 4},
        "model": {"initialized_from_scratch": True, "pretrained_parameters_used": False,
                  "time_features_enabled": True},
        "protocol": {"validation_test_pantry_contexts_never_used_for_training": True},
        "reward": {"human_feedback": False}, "evaluation": {}, "training": stages,
        "is_full_corpus_optimizer_coverage": True, "total_wall_seconds": 1,
        "process_peak_rss_bytes": 1000, "production_or_human_preference_quality_established": False,
        "time_metadata": {"enabled": True, "n_known": 2, "n_unknown": 2,
                          "numeric_time_array_sha256": "b" * 64, "semantics": "source totals only"},
        "run_configuration": {
            "arguments": arguments, "code_sha256": {}, "environment": {},
            "started_utc": "2026-09-10T00:00:00+00:00",
            "completed_utc": "2026-09-10T00:00:01+00:00", "status": "completed",
        },
    }
    (run / "report.json").write_text(json.dumps(report))
    return run, report, {"n_recipes": 4, "n_slots": 12, "corpus_sha256": "a" * 64}


def test_public_training_projection_excludes_private_paths(exporter, complete_run):
    run, _, metadata = complete_run
    public = exporter.verify_training(run, metadata)
    encoded = json.dumps(public)
    assert "/private/" not in encoded
    assert "coverage_artifact\"" not in encoded
    assert public["training"]["reinforce"]["epochs"][0]["bandit"]["sampled_actions"] == 8
    assert public["checkpoint_binding"]["checkpoint_files_sha256"]


def test_replaced_weights_cannot_inherit_a_full_training_coverage_report(exporter, complete_run):
    run, _, metadata = complete_run
    (run / "supervised" / "recipe_ranker.safetensors").write_bytes(b"replacement checkpoint")
    with pytest.raises(ValueError, match="differs from completed training evidence"):
        exporter.verify_training(run, metadata)


def test_legacy_run_requires_a_report_bound_post_training_audit(exporter, complete_run):
    run, report, metadata = complete_run
    checkpoints = report.pop("checkpoint_files_sha256")
    (run / "report.json").write_text(json.dumps(report))
    audit = {
        "canonical_corpus_sha256": metadata["corpus_sha256"],
        "audited_utc": "2026-09-10T00:00:02+00:00",
        "artifact_sha256": {
            "report.json": exporter.digest(run / "report.json"),
            **{f"{stage}/{filename}": digest for stage, files in checkpoints.items()
               for filename, digest in files.items()},
        },
    }
    (run / "aggregate_evidence.json").write_text(json.dumps(audit))
    assert exporter.verify_training(run, metadata)["checkpoint_binding"][
        "capture"] == "independent_post_training_audit"
    audit["artifact_sha256"]["report.json"] = "0" * 64
    (run / "aggregate_evidence.json").write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="does not bind"):
        exporter.verify_training(run, metadata)


def test_public_evidence_rejects_recipe_text_or_absolute_paths(exporter):
    for value in ({"steps": ["Source instructions"]}, {"source": "/Users/private/file"}):
        with pytest.raises(ValueError, match="public evidence"):
            exporter._metadata_only(value)


def test_publisher_refuses_source_data_files_before_reading_a_package(exporter, tmp_path):
    pytest.importorskip("huggingface_hub")
    publisher = importlib.import_module("publish_recipe_search")
    (tmp_path / "recipe_search.sqlite").write_bytes(b"not for publication")
    with pytest.raises(ValueError, match="allowed regular package files"):
        publisher.read_package(tmp_path)


def test_publication_requires_explicit_authorization(exporter, tmp_path):
    pytest.importorskip("huggingface_hub")
    script = Path(__file__).resolve().parents[1] / "scripts" / "publish_recipe_search.py"
    process = subprocess.run(
        [sys.executable, str(script), "--folder", str(tmp_path)],
        capture_output=True, text=True)
    assert process.returncode == 2
    assert "--public is required" in process.stderr


def test_anonymous_source_environment_disables_checkout_and_git_credential_inheritance(
        exporter, monkeypatch, tmp_path):
    pytest.importorskip("huggingface_hub")
    publisher = importlib.import_module("publish_recipe_search")
    monkeypatch.setenv("PYTHONPATH", "/unrelated/checkout")
    monkeypatch.setenv("GIT_CONFIG_PARAMETERS", "unrelated-configuration")
    monkeypatch.setenv("GH_TOKEN", "unit-test-placeholder")
    environment = publisher.isolated_environment(tmp_path)
    assert "PYTHONPATH" not in environment
    assert "GIT_CONFIG_PARAMETERS" not in environment
    assert "GH_TOKEN" not in environment
    assert environment["GIT_TERMINAL_PROMPT"] == "0"
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"


def test_release_recomputes_live_operational_gates_instead_of_trusting_a_boolean(exporter):
    summary = {
        "queries": 200, "constraint_violations": 0, "timeouts": 0,
        "empty_results": 0, "latency_ms": {"maximum": 100},
    }
    report = {
        "schema_version": 1, "queries_per_partition": 200,
        "selected_on_validation": "supervised", "release_evaluation": True,
        "test_scored": True, "selection_frozen_before_test_scoring": True,
        "operational_gate_passed": True, "training_query_partition_overlap": 0,
        "retrieval": {"timeout_seconds": 5},
        **{phase: {name: dict(summary) for name in ("heuristic", "supervised", "reinforce")}
           for phase in ("validation", "test")},
    }
    assert exporter.validate_live_evaluation(report) == "supervised"
    report["test"]["supervised"]["empty_results"] = 11
    with pytest.raises(ValueError, match="operational gate"):
        exporter.validate_live_evaluation(report)


def test_same_ingredient_corpus_does_not_authorize_changed_recipe_metadata(exporter):
    evaluation = {"corpus_sha256": "a" * 64, "catalog_sha256": "b" * 64}
    coverage = {"corpus_sha256": "a" * 64, "catalog_sha256": "b" * 64,
                "numeric_time_array_sha256": "c" * 64}
    training = {"time_metadata": {"numeric_time_array_sha256": "c" * 64}}
    exporter.verify_evaluated_catalog(evaluation, coverage, training)
    coverage["catalog_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="different source catalogs"):
        exporter.verify_evaluated_catalog(evaluation, coverage, training)
    coverage["catalog_sha256"] = "b" * 64
    coverage["numeric_time_array_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="source times differ"):
        exporter.verify_evaluated_catalog(evaluation, coverage, training)


def test_all_record_gate_checks_saved_counts_not_just_manifest_claims(exporter, complete_run):
    run, report, metadata = complete_run
    proof = run / "private_coverage" / "supervised-epoch-001.npz"
    np.savez(proof, seen_counts=np.asarray([1, 1, 0, 1], dtype=np.uint8),
             eligible_bitmap=np.asarray([15], dtype=np.uint8), n_corpus_rows=np.int64(4))
    report["training"]["supervised"]["epochs"][0]["coverage"][
        "coverage_artifact_sha256"] = exporter.digest(proof)
    (run / "report.json").write_text(json.dumps(report))
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    process = subprocess.run(
        [sys.executable, "-c",
         "import json,sys; from pathlib import Path; "
         "from export_recipe_search import verify_training; "
         "verify_training(Path(sys.argv[1]),json.loads(sys.argv[2]))",
         str(run), json.dumps(metadata)],
        cwd=scripts, capture_output=True, text=True)
    assert process.returncode != 0
    assert "not every row was seen exactly once" in process.stderr


@pytest.mark.parametrize("defect", ["pilot", "leakage", "missing_actions"])
def test_release_rejects_pilots_leaked_pantries_and_false_rl_counts(
        exporter, complete_run, defect):
    run, report, metadata = complete_run
    if defect == "pilot":
        report["mode"] = "pilot"
    elif defect == "leakage":
        report["training"]["supervised"]["epochs"][0][
            "query_partition_bucket_counts"] = [3] + [0] * 8 + [1]
    else:
        report["training"]["reinforce"]["epochs"][0]["bandit"]["sampled_actions"] = 0
    (run / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        exporter.verify_training(run, metadata)
