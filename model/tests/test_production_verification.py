from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from ingredient_model import artifacts
from ingredient_model.data.recipes import RecipeCorpus
from ingredient_model.production import verify_full_training
from ingredient_model.spec import TrainContext


@pytest.fixture
def production(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from models.set_transformer import train

    data = tmp_path / "data"
    (data / "recipes").mkdir(parents=True)
    corpus = RecipeCorpus(
        flat=np.array([0, 0, 1, 0, 1, 2]), offsets=np.array([0, 1, 3, 6]),
        lang=np.array(["en"] * 3), source=np.array(["fixture"] * 3),
        itos=["a", "b", "c", "d"])
    corpus_path = data / "recipes" / "recipe_ids.npz"
    np.savez(corpus_path, flat=corpus.flat, offsets=corpus.offsets)
    digest = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    (data / "GENERATION.json").write_text(json.dumps({
        "generation": "v2", "corpus": "recipe_ids.npz",
        "sha256": digest, "recipes": 3, "slots": 6, "vocab": 4,
    }))
    monkeypatch.setattr(train, "load_recipes", lambda _: corpus)
    monkeypatch.setattr(artifacts, "_environment", lambda: {
        "corpus_generation": "v2", "corpus_sha256": digest,
    })
    params = {**train.DEFAULTS, "d_model": 8, "n_heads": 2, "n_layers": 1,
              "epochs": 2, "batch_size": 2, "min_len": 1, "max_len": None,
              "max_recipes": 0, "expected_recipes": 3, "dropout": 0.0, "warmup": 0}
    run = tmp_path / "run"
    result = train.train_masked_set(TrainContext(
        graph="ii_graph.npz", seed=1, out_dir=run, params=params, split="full"))
    from ingredient_model.registry import get
    artifacts.save_run(
        "fixture", get("masked-set"), result, graph="ii_graph.npz",
        seed=1, params={**params, "split": "full", "no_eval": True},
        duration_s=0, out_dir=run)
    artifacts.save_metrics(run, artifacts.unevaluated_metrics("full"))
    return run, data


def test_completed_coverage_requires_counts_code_and_real_restorable_weights(production):
    run, data = production
    result = verify_full_training(run, data, expected_recipes=3)
    assert result["recipes_per_epoch"] == 3
    assert result["example_presentations"] == 6
    assert result["ingredient_slot_presentations"] == 12
    assert result["complete_predictor_restored"] is True
    assert result["evaluation_status"] == "not_run"
    assert len(result["artifact_sha256"]["state__tok__weight.npy"]) == 64
    assert not any("recall" in key for key in result)


def test_training_learns_weights_and_biases_from_a_fresh_network(production):
    import torch
    from models.set_transformer.train import _build

    run, _ = production
    manifest = artifacts.Manifest.load(run)
    torch.manual_seed(manifest.seed)
    initial = _build(manifest.shape[0], manifest.params, "cpu")
    for name in ("tok.weight", "bias"):
        trained = np.load(run / f"state__{name.replace('.', '__')}.npy", allow_pickle=False)
        assert not np.array_equal(
            trained, initial.state_dict()[name].detach().numpy())


@pytest.mark.parametrize("defect", [
    "examples", "unique_rows", "ingredient_slots", "missing_epoch",
    "missing_weights", "false_score", "different_corpus",
])
def test_incomplete_or_inconsistent_training_cannot_pass_verification(production, defect):
    run, data = production
    manifest = json.loads((run / "manifest.json").read_text())
    if defect == "examples":
        manifest["metadata"]["n_examples_seen"] -= 1
    elif defect == "unique_rows":
        manifest["metadata"]["epoch_coverage"][0]["unique_recipes"] -= 1
    elif defect == "ingredient_slots":
        manifest["metadata"]["n_ingredient_slots_seen"] -= 1
    elif defect == "missing_epoch":
        manifest["metadata"]["epoch_coverage"].pop()
    elif defect == "missing_weights":
        (run / "state__bias.npy").unlink()
    elif defect == "false_score":
        artifacts.save_metrics(run, {"split": "full", "M6_native_recall_at_10": 1.0})
    elif defect == "different_corpus":
        manifest["environment"]["corpus_sha256"] = "0" * 64
    (run / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_full_training(run, data, expected_recipes=3)


@pytest.mark.parametrize("public", [False, True])
def test_production_export_has_coverage_and_reload_evidence_not_a_borrowed_score(
        production, tmp_path, monkeypatch, public):
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("safetensors")
    from ingredient_model.config import REPO
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    import export_native_model as exporter

    run, data = production
    monkeypatch.setattr(exporter, "PATHS", SimpleNamespace(data=data))
    monkeypatch.setattr(exporter, "load_recipes", lambda: SimpleNamespace(itos=["a", "b", "c", "d"]))
    generation = json.loads((data / "GENERATION.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    args = SimpleNamespace(
        out=tmp_path / "export", report=tmp_path / "release.json", candidate="fixture",
        repo_id="incrediblecrab/llmmm-recipes", tag="v0.3.0-all-recipes", public=public)
    assert exporter.export_production(args, generation, run, manifest) == 0
    report = json.loads(args.report.read_text())
    assert report["training"]["example_presentations"] == 6
    assert report["evaluation_status"] == "not_run"
    assert report["reload_parity"]["all_state_tensors_identical"] is True
    assert report["reload_parity"]["max_absolute_logit_error"] == 0
    assert "recall_at_10" not in report
    assert not (args.out / "evaluation.json").exists()
    card = (args.out / "README.md").read_text()
    assert "all 3 canonical recipe records" in card
    assert "no held-out quality score" in card
    assert ("hf auth login" not in card) is public
    assert f"token={not public}" in card
    assert "its own learned\nweights and biases" in card
    assert "No pretrained checkpoint was used for initialization" in card
    assert card.rsplit("\n## ", 1)[1].startswith("Ideas for using this model\n")
    assert card.index("## Acknowledgements") < card.index("## Ideas for using this model")

    def forbidden(*args, **kwargs):
        pytest.fail("rendering documentation must not train, load data or export weights")

    monkeypatch.setattr(exporter, "load_recipes", forbidden)
    monkeypatch.setattr(exporter, "verify_full_training", forbidden)
    monkeypatch.setattr(exporter, "export_predictor", forbidden)
    assert exporter.render_production_card(report, public=public) == card
    policy = json.loads((args.out / "release_policy.json").read_text())
    assert policy["visibility"] == ("public" if public else "private")
    assert policy["weights_license"] is None
    from publish_native_model import read_package
    assert read_package(args.out)["training"]["recipes_per_epoch"] == 3
    (args.out / "evaluation.json").write_text('{"recall_at_10": 1.0}')
    with pytest.raises(ValueError, match="six allowed"):
        read_package(args.out)
