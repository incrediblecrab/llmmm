"""Saved predictors must survive re-evaluation, even with optional dependencies."""
from __future__ import annotations

import builtins
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ingredient_model import artifacts, cli
from ingredient_model.data.recipes import RecipeCorpus
from ingredient_model.data.splits import DEFAULT_SPLIT, get_split
from ingredient_model.eval import EvalContext
from ingredient_model.eval.report import collect
from ingredient_model.spec import ModelSpec, TrainResult

N_VOCAB = 32
NATIVE_METRICS = {
    "M6_native_recall_at_10": 1.0,
    "M6_native_recall_at_50": 1.0,
    "M6_native_mrr": 1.0,
    "M6_native_median_rank": 1.0,
    "M6_native_lift_over_popularity": 1.0,
}


@pytest.fixture
def save_run(monkeypatch):
    monkeypatch.setattr(artifacts, "_environment", lambda: {})

    def save(path, W, *, model="ease", arrays=None, metadata=None):
        def no_training(ctx):
            pytest.fail("restoring a saved run must not train")

        spec = ModelSpec(
            name=model, family={"ease": "recipe_basket",
                                "masked-set": "set_transformer"}.get(model, "matrix"),
            train=no_training)
        result = TrainResult(W, extra_arrays=arrays or {}, metadata=metadata or {})
        return artifacts.save_run(
            path.name, spec, result, graph=get_split("recipe-holdout").graph,
            seed=0, params={"split": "recipe-holdout"}, duration_s=1,
            out_dir=path)

    return save


@pytest.fixture
def mini_eval(monkeypatch):
    """Use the real harness/ranker with a tiny, entirely in-memory context."""
    W = np.random.default_rng(3).normal(size=(N_VOCAB, 6)).astype(np.float32)
    corpus = RecipeCorpus(
        flat=np.tile(np.arange(3, dtype=np.uint16), 12),
        offsets=np.arange(0, 37, 3, dtype=np.int64),
        lang=np.full(12, "en"), source=np.full(12, "test"),
        itos=[f"item-{i}" for i in range(N_VOCAB)])
    unigram = np.arange(1, N_VOCAB + 1, dtype=np.float64)
    calls, metrics = [], []

    def build_context(split):
        calls.append(("context", split))
        ctx = EvalContext(
            split=get_split(split), n=N_VOCAB, itos=corpus.itos,
            unigram=unigram, degree=np.ones(N_VOCAB), edge_set=set(),
            subs=SimpleNamespace(
                tier=lambda tier: [(0, 1), (0, 2), (1, 2)],
                anchors=lambda tier, min_subs: {0: {1, 2, 3}}),
            held=(np.array([0, 1, 2]), np.array([1, 2, 3])))
        ctx.link_negatives = np.array([10, 11, 12])
        return ctx

    def held_out_recipes(split, limit):
        calls.append(("corpus", split, limit))
        return corpus

    evaluate = cli.evaluate

    def record_evaluation(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        metrics.append(result)
        return result

    monkeypatch.setattr(cli, "build_context", build_context)
    monkeypatch.setattr(cli, "held_out_recipes", held_out_recipes)
    monkeypatch.setattr(cli, "evaluate", record_evaluation)
    return SimpleNamespace(W=W, corpus=corpus, unigram=unigram,
                           calls=calls, metrics=metrics)


@pytest.fixture
def no_torch(monkeypatch):
    original = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError("torch intentionally unavailable", name="torch")
        return original(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)


def _ease_scores():
    B = np.zeros((N_VOCAB, N_VOCAB), np.float32)
    B[:3, :3] = 1
    np.fill_diagonal(B, 0)
    return B


def _eval_args(target, *, whiten=False, split=None):
    args = ["eval", str(target), "--n-completion", "9"]
    if whiten:
        args.append("--whiten")
    if split is not None:
        args += ["--split", split]
    return cli.build_parser().parse_args(args)


def _old_metrics(run):
    path = run / artifacts.METRICS
    path.write_text('{"M6_native_recall_at_10": -1, "old_only": true}\n')
    return path.read_bytes()


def test_ease_restores_asymmetric_item_scores(tmp_path, save_run, no_torch):
    B = np.arange(16, dtype=np.float32).reshape(4, 4)
    np.fill_diagonal(B, 0)
    run = save_run(tmp_path / "ease", np.ones((4, 3), np.float32),
                   arrays={"item_scores": B})
    scorer = artifacts.load_native_scorer(run, 4)
    assert scorer is not None
    got = scorer(np.array([[0, 2], [1, 3]], dtype=np.int64))
    np.testing.assert_array_equal(got, [[8, 10, 2, 14], [16, 13, 20, 7]])


@pytest.mark.parametrize("whiten", [False, True])
def test_cmd_eval_keeps_native_metrics(tmp_path, save_run, mini_eval, no_torch,
                                      capsys, whiten):
    run = save_run(tmp_path / "ease", mini_eval.W,
                   arrays={"item_scores": _ease_scores()})
    previous = _old_metrics(run)

    assert cli.cmd_eval(_eval_args(run, whiten=whiten)) == 0

    metrics = mini_eval.metrics[0]
    native = {k: v for k, v in metrics.items() if k.startswith("M6_native_")}
    assert native == NATIVE_METRICS
    assert metrics["M6_n"] == 9
    assert metrics["whitened"] is whiten
    assert metrics["M6_native_recall_at_10"] > metrics["M6_recall_at_10"]
    assert mini_eval.calls == [
        ("context", "recipe-holdout"), ("corpus", "recipe-holdout", 36)]
    output = capsys.readouterr().out
    assert "native scorer" in output
    assert ("[whitened]" in output) is whiten
    if whiten:
        assert (run / artifacts.METRICS).read_bytes() == previous
    else:
        assert artifacts.load_metrics(run) == metrics
        assert "old_only" not in metrics


@pytest.mark.parametrize("model, asset", [
    ("ease", "item_scores.npy"),
    ("masked-set", "state__tok__weight.npy"),
])
def test_missing_native_assets_preserve_metrics(
        tmp_path, save_run, mini_eval, no_torch, model, asset):
    run = save_run(tmp_path / model, mini_eval.W, model=model)
    previous = _old_metrics(run)
    with pytest.raises(FileNotFoundError, match=asset):
        cli.cmd_eval(_eval_args(run))
    assert (run / artifacts.METRICS).read_bytes() == previous
    assert not mini_eval.calls
    assert not mini_eval.metrics


@pytest.mark.parametrize("damage", ["bytes", "shape", "nan", "dtype", "archive"])
def test_corrupt_ease_assets_preserve_metrics(
        tmp_path, save_run, mini_eval, damage):
    run = save_run(tmp_path / "ease", mini_eval.W)
    path = run / "item_scores.npy"
    if damage == "bytes":
        path.write_bytes(b"not a numpy array")
    elif damage == "shape":
        np.save(path, np.zeros((N_VOCAB, N_VOCAB - 1), np.float32))
    elif damage == "nan":
        np.save(path, np.full((N_VOCAB, N_VOCAB), np.nan))
    elif damage == "dtype":
        np.save(path, np.full((N_VOCAB, N_VOCAB), "invalid"))
    else:
        with path.open("wb") as f:
            np.savez(f, scores=_ease_scores())
    previous = _old_metrics(run)
    with pytest.raises(ValueError, match="item_scores.npy"):
        cli.cmd_eval(_eval_args(run))
    assert (run / artifacts.METRICS).read_bytes() == previous
    assert not mini_eval.calls
    assert not mini_eval.metrics


@pytest.mark.parametrize("shape", [(N_VOCAB - 1, 6), (N_VOCAB, 5)])
def test_manifest_embedding_mismatch_preserves_metrics(
        tmp_path, save_run, mini_eval, shape):
    run = save_run(tmp_path / "ease", mini_eval.W,
                   arrays={"item_scores": _ease_scores()})
    np.save(run / artifacts.EMBEDDING, np.zeros(shape, np.float32))
    previous = _old_metrics(run)
    with pytest.raises(ValueError, match="manifest records"):
        cli.cmd_eval(_eval_args(run))
    assert (run / artifacts.METRICS).read_bytes() == previous
    assert not mini_eval.calls


def test_native_scorer_rejects_a_different_vocabulary(tmp_path, save_run):
    run = save_run(tmp_path / "ease", np.ones((4, 3), np.float32),
                   arrays={"item_scores": np.eye(4, dtype=np.float32)})
    with pytest.raises(ValueError, match="4 vocabulary rows, expected 5"):
        artifacts.load_native_scorer(run, 5)


@pytest.fixture
def masked_run(tmp_path, save_run):
    torch = pytest.importorskip("torch")
    from models.set_transformer.train import DEFAULTS, _build, _make_scorer

    params = {**DEFAULTS, "d_model": 8, "n_heads": 2, "n_layers": 1,
              "ff_mult": 1, "dropout": 0.0, "tie_output": False}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = _build(N_VOCAB, params, "cpu")
    W = model.tok.weight.detach().numpy()[:N_VOCAB]
    arrays = {f"state__{key.replace('.', '__')}": value.numpy()
              for key, value in model.state_dict().items()}
    run = save_run(tmp_path / "masked-set", W, model="masked-set",
                   arrays=arrays, metadata=params)
    return run, _make_scorer(model, N_VOCAB, "cpu")


def test_masked_set_restores_metadata_and_actual_scores(masked_run):
    run, original = masked_run
    contexts = np.array([[0, 2], [1, 3], [5, 6]], dtype=np.int64)
    expected = original(contexts)
    scorer = artifacts.load_native_scorer(run, N_VOCAB)
    assert scorer is not None
    actual = scorer(contexts)
    assert actual.shape == (3, N_VOCAB)
    assert np.ptp(actual) > 0
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("damage", ["missing", "shape", "nan", "bytes", "tokens"])
def test_corrupt_masked_set_assets_preserve_metrics(masked_run, mini_eval, damage):
    run, _ = masked_run
    path = run / "state__bias.npy"
    if damage == "missing":
        path.unlink()
    elif damage == "shape":
        np.save(path, np.zeros(N_VOCAB - 1, np.float32))
    elif damage == "nan":
        np.save(path, np.full(N_VOCAB, np.nan, np.float32))
    elif damage == "bytes":
        path.write_bytes(b"corrupt tensor")
    else:
        np.save(run / "state__tok__weight.npy",
                np.zeros((N_VOCAB, 8), np.float32))
    previous = _old_metrics(run)
    with pytest.raises(ValueError, match="native scorer"):
        cli.cmd_eval(_eval_args(run))
    assert (run / artifacts.METRICS).read_bytes() == previous
    assert not mini_eval.calls
    assert not mini_eval.metrics


@pytest.mark.parametrize("kind", ["run", "npy"])
@pytest.mark.parametrize("whiten", [False, True])
def test_embedding_only_eval_needs_no_torch(
        tmp_path, save_run, mini_eval, no_torch, capsys, kind, whiten):
    if kind == "run":
        target = save_run(tmp_path / "svd", mini_eval.W, model="svd-ppmi",
                          arrays={"item_scores": _ease_scores()})
        directory = target
        split = "recipe-holdout"
    else:
        target = tmp_path / "loose.npy"
        np.save(target, mini_eval.W)
        directory = tmp_path
        split = DEFAULT_SPLIT
    previous = _old_metrics(directory)

    assert cli.cmd_eval(_eval_args(target, whiten=whiten)) == 0

    metrics = mini_eval.metrics[0]
    assert not any(k.startswith("M6_native_") for k in metrics)
    assert metrics["M6_n"] == 9
    assert metrics["whitened"] is whiten
    assert mini_eval.calls == [("context", split), ("corpus", split, 36)]
    output = capsys.readouterr().out
    assert "native scorer" not in output
    assert ("[whitened]" in output) is whiten
    if kind == "run" and not whiten:
        assert artifacts.load_metrics(directory) == metrics
    else:
        assert (directory / artifacts.METRICS).read_bytes() == previous


def test_eval_keeps_explicit_split_override(tmp_path, save_run, mini_eval):
    run = save_run(tmp_path / "ease", mini_eval.W,
                   arrays={"item_scores": _ease_scores()})
    assert cli.cmd_eval(_eval_args(run, split="edge-holdout")) == 0
    assert mini_eval.calls == [
        ("context", "edge-holdout"), ("corpus", "edge-holdout", 36)]
    assert artifacts.load_metrics(run)["split"] == "edge-holdout"
    assert artifacts.load_metrics(run)["M6_native_recall_at_10"] == 1.0


def test_missing_training_data_points_to_verified_restore(monkeypatch):
    monkeypatch.setattr(
        cli, "get", lambda name: SimpleNamespace(name=name, requires=("recipes",)))
    monkeypatch.setattr(cli, "check_available", lambda requires: ["recipes"])
    args = cli.build_parser().parse_args(["train", "ease"])

    with pytest.raises(SystemExit) as refused:
        cli.cmd_train(args)

    assert str(refused.value) == (
        "ease needs missing datasets: recipes\n"
        "  make restore BUNDLE=/private/path/archive.tar.gz (from model/)")


def test_cold_imports_do_not_require_torch(tmp_path, save_run):
    W = np.ones((4, 3), np.float32)
    native = save_run(tmp_path / "ease", W,
                      arrays={"item_scores": np.eye(4, dtype=np.float32)})
    embedding = save_run(tmp_path / "svd", W, model="svd-ppmi")
    code = """
import builtins
import sys
from pathlib import Path
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError("torch intentionally unavailable", name="torch")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import numpy as np
from ingredient_model import cli
from ingredient_model.artifacts import load_native_scorer
from scripts import bootstrap_m6, m6_intervals
scorer = load_native_scorer(Path(sys.argv[1]), 4)
np.testing.assert_array_equal(scorer(np.array([[0, 2]])), [[1, 0, 1, 0]])
assert load_native_scorer(Path(sys.argv[2]), 4) is None
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(native), str(embedding)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_metadata_only_runs_are_visible_but_not_weight_candidates(
        tmp_path, save_run, monkeypatch, no_torch, capsys):
    root = tmp_path / "runs"
    W = np.ones((4, 3), np.float32)
    metadata = save_run(root / "sweep" / "metadata-ease", W)
    unscored = save_run(root / "sweep" / "unscored", W, model="svd-ppmi")
    ready = save_run(root / "ready", W, model="svd-ppmi")
    for run in (metadata, unscored):
        (run / artifacts.EMBEDDING).unlink()
    artifacts.save_metrics(metadata, {
        "M4_link_auc": 0.6, "M6_recall_at_10": 0.2,
        "M6_native_recall_at_10": 0.8, "M6_popularity_recall_at_10": 0.3})
    artifacts.save_metrics(ready, {"M4_link_auc": 0.7, "M6_recall_at_10": 0.4})
    partial = root / "partial"
    partial.mkdir()
    np.save(partial / artifacts.EMBEDDING, W)
    artifacts.save_metrics(partial, {"M4_link_auc": 0.9})

    assert list(artifacts.iter_runs(root)) == [ready]
    assert list(artifacts.iter_runs(root, require_embedding=False)) == [
        ready, metadata, unscored]
    assert list(artifacts.iter_runs(root / "absent", require_embedding=False)) == []
    for run in (ready, partial):
        (run / artifacts.EMBEDDING).unlink()
    assert not list(root.rglob(artifacts.EMBEDDING))
    assert list(artifacts.iter_runs(root)) == []
    rows = collect(root)
    assert {row["run_id"] for row in rows} == {"ready", "metadata-ease"}
    native_row = next(row for row in rows if row["run_id"] == "metadata-ease")
    assert native_row["_M6_best"] == 0.8
    assert native_row["_M6_best_lift"] == 0.5

    monkeypatch.setattr(artifacts, "PATHS", SimpleNamespace(runs=root))
    assert cli.cmd_runs(cli.build_parser().parse_args(["runs"])) == 0
    output = capsys.readouterr().out
    assert all(name in output for name in ("metadata-ease", "unscored", "ready"))
    assert "partial" not in output
    assert cli.cmd_report(cli.build_parser().parse_args(["report"])) == 0
    output = capsys.readouterr().out
    assert "2 scored runs" in output
    assert "0.8000" in output


def _script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["bootstrap_m6", "m6_intervals"])
@pytest.mark.parametrize("missing", [False, True])
def test_interval_scripts_restore_native_scores(
        tmp_path, save_run, mini_eval, monkeypatch, no_torch, name, missing):
    script = _script(name)
    run = save_run(tmp_path / "ease", mini_eval.W,
                   arrays={"item_scores": _ease_scores()})
    runs = {"ease": {"dir": run, "metrics": {},
                     "manifest": {"run_id": run.name}}}
    if missing:
        (run / "item_scores.npy").unlink()
    monkeypatch.setattr(script, "held_out_recipes", lambda split: mini_eval.corpus)

    if name == "bootstrap_m6":
        monkeypatch.setattr(script, "load_recipes", lambda name: mini_eval.corpus)
        if missing:
            with pytest.raises(FileNotFoundError, match="item_scores.npy"):
                script.compute_ranks(runs)
        else:
            store = script.compute_ranks(runs)
            np.testing.assert_array_equal(store["ease::native"], np.ones(12))
        return

    output = tmp_path / "m6_replication.json"
    previous = b'{"old": true}\n'
    output.write_bytes(previous)
    monkeypatch.setattr(script, "PATHS", SimpleNamespace(results=tmp_path))
    monkeypatch.setattr(script, "OUT_JSON", output)
    monkeypatch.setattr(script, "load_runs", lambda root: runs)
    monkeypatch.setattr(script, "load_ii_graph",
                        lambda name: SimpleNamespace(unigram=mini_eval.unigram))
    monkeypatch.setattr(script, "compare_to_canonical", lambda out: 0)
    monkeypatch.setattr(sys, "argv", [name, "--n-boot", "8"])
    if missing:
        with pytest.raises(FileNotFoundError, match="item_scores.npy"):
            script.main()
        assert output.read_bytes() == previous
    else:
        with pytest.raises(SystemExit) as exited:
            script.main()
        assert exited.value.code == 0
        row = json.loads(output.read_text())["models"][0]
        assert row["served"] == "native"
        assert row["native"]["recall_at_10"] == 1.0
        assert row["native"]["mrr"] == 1.0
