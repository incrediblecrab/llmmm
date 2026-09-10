from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

from ingredient_model.recipe_ranker import (
    CONFIG_FILENAME,
    FEATURE_NAMES,
    TIME_FEATURE_START,
    WEIGHTS_FILENAME,
    RecipeRankingPolicy,
    candidate_features,
    deterministic_baseline_score,
    heuristic_scores,
)

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_recipe_ranker.py"
_SPEC = importlib.util.spec_from_file_location("recipe_ranker_training_test_module", _SCRIPT)
training = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = training
_SPEC.loader.exec_module(training)


def features(available=(0, 1), candidates=((0, 1), (0, 2), (2, 3)), **kwargs):
    return candidate_features(available, candidates, ingredient_frequency=np.array([10, 8, 4, 2]),
                              n_recipes=10, **kwargs)


@pytest.fixture
def corpus():
    rng = np.random.default_rng(117)
    rows = [np.array(ids) for ids in ([0, 1, 2], [0, 1, 2], [0, 1], [1, 2], [0], [1], [2])]
    rows.extend(np.sort(rng.choice(24, size=int(rng.integers(1, 7)), replace=False))
                for _ in range(89))
    flat = np.concatenate(rows).astype(np.uint16)
    offsets = np.r_[np.int64(0), np.cumsum([len(row) for row in rows])]
    result = training.NumericCorpus(
        flat, offsets, tuple(f"ingredient_{index}" for index in range(24)),
        training.array_digest(flat, offsets))
    result.build_postings()
    return result


def test_exact_feature_definitions_and_equivalent_sets():
    values = features()
    by_name = {name: values[:, index] for index, name in enumerate(FEATURE_NAMES)}
    assert values.dtype == np.float32
    assert values.shape == (3, 20)
    assert np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()
    np.testing.assert_allclose(by_name["recipe_coverage"], [1, 0.5, 0])
    np.testing.assert_allclose(by_name["pantry_coverage"], [1, 0.5, 0])
    np.testing.assert_allclose(by_name["jaccard"], [1, 1 / 3, 0])
    np.testing.assert_array_equal(by_name["exact_set_match"], [1, 0, 0])
    np.testing.assert_array_equal(by_name["fully_available"], [1, 0, 0])
    np.testing.assert_allclose(by_name["missing_fraction"], [0, 0.5, 1])
    idf = 1 + np.log(11 / (1 + np.array([10, 8, 4, 2])))
    assert by_name["idf_recipe_coverage"][1] == pytest.approx(idf[0] / (idf[0] + idf[2]))
    duplicate = features([1, 0, 1], [[1, 0, 0], [2, 0], [3, 2]])
    np.testing.assert_array_equal(values, duplicate)
    np.testing.assert_array_equal(values, features({0, 1}, [{0, 1}, {0, 2}, {2, 3}]))


def test_heuristic_is_fixed_deterministic_and_prefers_exact_matching():
    scores = deterministic_baseline_score(features())
    assert scores.dtype == np.float64
    assert scores[0] > scores[1] > scores[2]
    np.testing.assert_array_equal(scores, heuristic_scores(features()))
    permutation = np.array([2, 0, 1])
    np.testing.assert_array_equal(scores[permutation], deterministic_baseline_score(features()[permutation]))


def test_unknown_time_is_distinct_from_a_known_zero_and_never_imputes_feasibility():
    values = features(total_minutes=[None, 0, 45], max_total_minutes=30)
    index = {name: position for position, name in enumerate(FEATURE_NAMES)}
    np.testing.assert_array_equal(values[:, index["time_known"]], [0, 1, 1])
    np.testing.assert_array_equal(values[:, index["budget_known"]], [1, 1, 1])
    np.testing.assert_array_equal(values[:, index["within_time_budget"]], [0, 1, 0])
    np.testing.assert_array_equal(values[:, index["time_budget_slack"]], [0, 1, 0])
    np.testing.assert_array_equal(values[:, index["time_budget_fraction"]], [0, 0, 1])
    assert values.shape[0] == 3  # Features cannot filter or silently relax constraints.
    extreme = features(total_minutes=[1e308, None, 1e-300], max_total_minutes=1e-300)
    assert np.isfinite(extreme).all()


@pytest.mark.parametrize("available,candidates", [
    ([], [[0]]), ([0.0], [[0]]), ([True], [[0]]), ([0, True], [[0]]),
    ([-1], [[0]]), ([4], [[0]]), ([[0]], [[0]]), ("0", [[0]]),
    ([0], [[]]), ([0], [[4]]), ([0], [[False]]), ([0], [[1.0]]),
    ([0], ["0"]), ([0], (value for value in [[0]])), ([0], np.array(1)),
    ([0], [np.array(1)]),
])
def test_invalid_ingredient_inputs_raise_instead_of_silent_zero_scores(available, candidates):
    with pytest.raises(ValueError):
        features(available, candidates)


@pytest.mark.parametrize("frequency", [
    np.array([-1, 2]), np.array([11, 2]), np.array([1.5, 2]),
    np.array([np.nan, 2]), np.array([np.inf, 2]), np.array([[1, 2]]),
    np.array([True, False]), np.array([]),
])
def test_invalid_document_frequencies_raise(frequency):
    with pytest.raises(ValueError, match="ingredient_frequency"):
        candidate_features([0], [[0]], ingredient_frequency=frequency, n_recipes=10)


@pytest.mark.parametrize("kwargs", [
    {"max_total_minutes": 0}, {"max_total_minutes": -1}, {"max_total_minutes": True},
    {"max_total_minutes": float("nan")}, {"max_total_minutes": float("inf")},
    {"total_minutes": [1, 2]}, {"total_minutes": [1, True, 2]},
    {"total_minutes": [1, -1, 2]}, {"total_minutes": [1, float("nan"), 2]},
    {"total_minutes": [1, float("inf"), 2]}, {"total_minutes": "123"},
    {"total_minutes": np.array(1)},
])
def test_invalid_time_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        features(**kwargs)


def test_empty_shortlist_and_bounded_candidate_count():
    empty = features(candidates=[])
    assert empty.shape == (0, len(FEATURE_NAMES))
    assert not len(RecipeRankingPolicy().score(empty))
    assert not len(deterministic_baseline_score(empty))
    with pytest.raises(ValueError, match="at most"):
        features(candidates=[[0]] * 10_001)


def test_model_is_own_parameter_seeded_and_does_not_perturb_global_rng():
    before = torch.random.get_rng_state().clone()
    first, second = RecipeRankingPolicy(seed=7), RecipeRankingPolicy(seed=7)
    assert torch.equal(before, torch.random.get_rng_state())
    assert sum(value.numel() for value in first.parameters()) == 705
    np.testing.assert_array_equal(first.score(features()), second.score(features()))
    np.testing.assert_array_equal(first.score(features())[::-1], first.score(features()[::-1]))
    assert not np.array_equal(first.score(features()), RecipeRankingPolicy(seed=8).score(features()))
    changing_time = features(total_minutes=[1, 5, 20], max_total_minutes=30)
    np.testing.assert_array_equal(first.score(features()), first.score(changing_time))
    enabled = RecipeRankingPolicy(seed=7, time_features_enabled=True)
    assert not np.array_equal(enabled.score(features()), enabled.score(changing_time))


@pytest.mark.parametrize("bad", [
    np.zeros((1, 1)), np.zeros(20), np.zeros((1, 20), dtype=bool),
    np.full((1, 20), np.nan), np.full((1, 20), np.inf),
    np.full((1, 20), -0.1), np.full((1, 20), 1.1), np.zeros((10_001, 20)),
])
def test_score_and_baseline_reject_invalid_feature_arrays(bad):
    for scorer in (RecipeRankingPolicy().score, deterministic_baseline_score):
        with pytest.raises(ValueError):
            scorer(bad)


def test_nonfinite_policy_parameters_cannot_produce_success_shaped_scores():
    policy = RecipeRankingPolicy()
    with torch.no_grad():
        next(policy.parameters()).fill_(torch.nan)
    with pytest.raises(ValueError, match="non-finite scores"):
        policy.score(features())


def test_strict_safetensors_roundtrip_and_no_overwrite(tmp_path):
    policy = RecipeRankingPolicy(8, seed=21, time_features_enabled=True)
    path = tmp_path / "policy"
    policy.save(path)
    loaded = RecipeRankingPolicy.load(path)
    np.testing.assert_array_equal(policy.score(features()), loaded.score(features()))
    assert {value.name for value in path.iterdir()} == {CONFIG_FILENAME, WEIGHTS_FILENAME}
    with pytest.raises(FileExistsError, match="overwrite"):
        policy.save(path)
    with pytest.raises(FileNotFoundError, match="safetensors"):
        RecipeRankingPolicy.load(tmp_path / "missing")
    pickle_only = tmp_path / "pickle_only"
    pickle_only.mkdir()
    (pickle_only / "pytorch_model.bin").write_bytes(b"not a model; never read")
    with pytest.raises(FileNotFoundError, match="safetensors"):
        RecipeRankingPolicy.load(pickle_only)


@pytest.mark.parametrize("mutation", [
    lambda config: config.update(extra="unrecognized"),
    lambda config: config.update(schema_version=True),
    lambda config: config.update(hidden_dim=True),
    lambda config: config.update(hidden_dim=100_000),
    lambda config: config.update(feature_names=list(reversed(FEATURE_NAMES))),
    lambda config: config.update(weights_file="../another.safetensors"),
    lambda config: config.update(time_features_enabled=1),
])
def test_configuration_schema_is_strict(tmp_path, mutation):
    RecipeRankingPolicy().save(tmp_path)
    path = tmp_path / CONFIG_FILENAME
    config = json.loads(path.read_text())
    mutation(config)
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        RecipeRankingPolicy.load(tmp_path)


@pytest.mark.parametrize("text", [
    '{"schema_version": 1, "schema_version": 1}',
    '{"schema_version": NaN}',
    "[]",
])
def test_json_duplicate_keys_nonfinite_constants_and_nonobjects_are_rejected(tmp_path, text):
    RecipeRankingPolicy().save(tmp_path)
    (tmp_path / CONFIG_FILENAME).write_text(text)
    with pytest.raises(ValueError):
        RecipeRankingPolicy.load(tmp_path)


def test_checksum_is_checked_before_weights_are_loaded(tmp_path):
    RecipeRankingPolicy().save(tmp_path)
    (tmp_path / WEIGHTS_FILENAME).write_bytes(b"corrupted tensor file")
    with pytest.raises(ValueError, match="SHA256"):
        RecipeRankingPolicy.load(tmp_path)


@pytest.mark.parametrize("corruption", ["missing", "extra", "shape", "dtype", "nonfinite", "mask"])
def test_tensor_inventory_shapes_dtypes_and_values_are_strict(tmp_path, corruption):
    RecipeRankingPolicy().save(tmp_path)
    weights_path = tmp_path / WEIGHTS_FILENAME
    state = load_tensors(weights_path.read_bytes())
    name = "network.0.weight"
    if corruption == "missing":
        state.pop(name)
    elif corruption == "extra":
        state["extra.weight"] = torch.ones(1)
    elif corruption == "shape":
        state[name] = state[name][:-1]
    elif corruption == "dtype":
        state[name] = state[name].double()
    elif corruption == "nonfinite":
        state[name][0, 0] = torch.nan
    else:
        state["feature_mask"][TIME_FEATURE_START] = 1
    data = save_tensors({key: value.contiguous() for key, value in state.items()})
    weights_path.write_bytes(data)
    config_path = tmp_path / CONFIG_FILENAME
    config = json.loads(config_path.read_text())
    config["weights_sha256"] = hashlib.sha256(data).hexdigest()
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        RecipeRankingPolicy.load(tmp_path)


def test_numeric_postings_are_full_document_frequency_aligned_and_cache_verified(corpus, tmp_path):
    metadata = corpus.build_postings(tmp_path / "index")
    assert metadata["n_slots_indexed"] == len(corpus.flat)
    for ingredient in range(corpus.n_vocab):
        lower, upper = corpus.posting_offsets[ingredient:ingredient + 2]
        rows = corpus.posting_rows[lower:upper]
        expected = [row for row in range(corpus.n_recipes) if ingredient in corpus.recipe(row)]
        np.testing.assert_array_equal(rows, expected)
    assert corpus.build_postings(tmp_path / "index")["cache_reused"] is True
    (tmp_path / "index/posting_rows.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        corpus.build_postings(tmp_path / "index")


@pytest.mark.parametrize("flat,offsets", [
    ([0, 0], [0, 2]), ([1, 0], [0, 2]), ([0], [0, 0, 1]), ([2], [0, 1]),
])
def test_invalid_or_empty_numeric_recipe_sets_are_not_silently_excluded(flat, offsets):
    with pytest.raises(ValueError):
        training.NumericCorpus(np.asarray(flat), np.asarray(offsets), ("a", "b"), "fixture")


def test_hash_partitions_are_source_independent_set_disjoint_and_reproducible(corpus):
    sampler = training.QuerySampler(corpus, candidates=8, pool_size=32)
    all_signatures = []
    for split in ("train", "validation", "test"):
        first = sampler.batch(np.arange(8), np.random.default_rng(24), split=split)
        second = sampler.batch(np.arange(8), np.random.default_rng(24), split=split)
        assert first.digest() == second.digest()
        signatures = [value.tobytes() for value in first.signatures]
        assert all(training.query_partition(signature) == split for signature in signatures)
        all_signatures.append(set(signatures))
    assert not (all_signatures[0] & all_signatures[1])
    assert not (all_signatures[0] & all_signatures[2])
    assert not (all_signatures[1] & all_signatures[2])
    assert training.pantry_signature(np.array([2, 1, 2])) == training.pantry_signature(np.array([1, 2]))


def test_noise_never_draws_zero_frequency_terms_even_at_cdf_boundary(monkeypatch):
    corpus = training.NumericCorpus(
        np.array([1, 2], dtype=np.uint16), np.array([0, 1, 2]),
        ("unobserved_first", "observed_first", "observed_second", "unobserved_last"), "fixture")
    corpus.build_postings()
    sampler = training.QuerySampler(corpus, candidates=2, pool_size=2, max_noise=1)

    class BoundaryRng:
        def integers(self, high):
            return min(1, high - 1)

        def random(self, size):
            return np.zeros(size)

    monkeypatch.setattr(training, "query_partition", lambda signature: "train")
    pantries, *_ = sampler._pantries(np.array([1]), BoundaryRng(), "train", None)
    np.testing.assert_array_equal(pantries[0], [1, 2])
    assert sampler.observed_vocabulary_size == 2


def test_vectorized_training_features_match_public_interface_and_hard_constraints(corpus):
    times = np.arange(corpus.n_recipes, dtype=np.float64) + 1
    times[1::3] = np.nan
    sampler = training.QuerySampler(corpus, candidates=8, pool_size=64, times=times)
    batch = sampler.batch(np.arange(24), np.random.default_rng(49), split="train")
    for index, source in enumerate(batch.source_rows):
        pantry = batch.pantry_flat[batch.pantry_offsets[index]:batch.pantry_offsets[index + 1]]
        valid = batch.valid[index]
        rows = batch.candidate_rows[index, valid]
        recipes = [corpus.recipe(int(row)) for row in rows]
        budget = batch.budgets[index]
        expected = candidate_features(
            pantry, recipes, ingredient_frequency=corpus.frequency, n_recipes=corpus.n_recipes,
            total_minutes=[None if np.isnan(times[row]) else float(times[row]) for row in rows],
            max_total_minutes=None if np.isnan(budget) else float(budget))
        np.testing.assert_allclose(batch.features[index, valid], expected, atol=1e-7, rtol=0)
        assert len(rows) == len(set(rows.tolist()))
        assert source in rows
        for position, row in enumerate(rows):
            ingredients = set(corpus.recipe(int(row)).tolist())
            assert ingredients & set(pantry.tolist())
            assert len(ingredients - set(pantry.tolist())) <= batch.max_missing[index]
            if np.isfinite(budget):
                assert np.isfinite(times[row]) and times[row] <= budget
            assert bool(batch.positives[index, position]) == np.array_equal(
                corpus.recipe(int(row)), corpus.recipe(int(source)))
    assert batch.statistics["negative_actions"] > 0
    assert batch.statistics["negative_actions_sharing_two_or_more_ingredients"] > 0
    assert batch.statistics["time_constrained_queries"] > 0


def test_identical_full_ingredient_sets_are_equivalent_positive_actions(corpus):
    sampler = training.QuerySampler(corpus, candidates=3, pool_size=3)
    _, _, positives = sampler._pool_features(
        [np.array([0, 1])], np.array([[0, 1, 2]]), np.array([0]), np.array([np.nan]))
    np.testing.assert_array_equal(positives, [[True, True, False]])


def test_training_feature_kernel_has_an_explicit_ingredient_memory_bound(corpus, monkeypatch):
    sampler = training.QuerySampler(corpus, candidates=3, pool_size=3)
    monkeypatch.setattr(corpus, "sizes", np.full(corpus.n_recipes, 2_000_000))
    with pytest.raises(ValueError, match="reduce batch_size"):
        sampler._pool_features(
            [np.array([0, 1])], np.array([[0, 1, 2]]), np.array([0]), np.array([np.nan]))


@pytest.fixture
def time_catalog(corpus, tmp_path):
    path = tmp_path / "times.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            "CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT);"
            "CREATE TABLE recipes(id INTEGER PRIMARY KEY, total_minutes REAL, time_status TEXT,"
            "prep_minutes REAL, cook_minutes REAL, derived_total_minutes REAL);")
        connection.executemany("INSERT INTO metadata VALUES (?, ?)", [
            (key, json.dumps(value)) for key, value in {
                "schema_version": 1, "partial": False, "n_recipes": corpus.n_recipes,
                "corpus_sha256": corpus.corpus_sha256,
            }.items()])
        connection.executemany("INSERT INTO recipes VALUES (?, ?, ?, ?, ?, ?)", [
            (row, None if row % 2 else float(row + 1),
             "unknown" if row % 2 else "source_total", 10, 15, 25)
            for row in range(corpus.n_recipes)])
    return path


def test_optional_times_read_only_matching_complete_catalog(corpus, time_catalog):
    path = time_catalog
    before = training.sha256_file(path)
    times, metadata = training.load_catalog_times(path, corpus)
    assert training.sha256_file(path) == before
    assert times.shape == (corpus.n_recipes,) and times[0] == 1
    assert np.isnan(times[1])
    assert metadata["n_known"] + metadata["n_unknown"] == corpus.n_recipes
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM recipes WHERE id = 1")
    with pytest.raises(ValueError, match="exactly once"):
        training.load_catalog_times(path, corpus)


@pytest.mark.parametrize("minutes,status", [
    (0, "source_total"), (-1, "source_total"), (float("inf"), "source_total"),
    (25, "derived_total"), (25, "unknown"),
])
def test_training_rejects_invalid_or_derived_known_totals(corpus, time_catalog, minutes, status):
    with sqlite3.connect(time_catalog) as connection:
        connection.execute("UPDATE recipes SET total_minutes=?, time_status=? WHERE id=0", (minutes, status))
    with pytest.raises(ValueError, match="finite positive source_total"):
        training.load_catalog_times(time_catalog, corpus)


def test_listwise_loss_sums_equivalent_positives_and_excludes_padding():
    logits = torch.tensor([[0.0, 0.0, 0.0, 100.0]], requires_grad=True)
    valid = torch.tensor([[True, True, True, False]])
    positives = torch.tensor([[True, True, False, False]])
    loss = training.listwise_loss(logits, valid, positives)
    assert float(loss.detach()) == pytest.approx(np.log(3 / 2))
    loss.backward()
    assert logits.grad[0, 3] == 0
    assert logits.grad[0, 0] < 0 and logits.grad[0, 1] < 0 and logits.grad[0, 2] > 0


def test_reinforce_samples_actions_and_has_reward_directed_policy_gradients():
    logits = torch.zeros((64, 4), requires_grad=True)
    valid = torch.tensor([[True, True, True, False]]).expand(64, 4)
    positives = torch.tensor([[True, False, False, False]]).expand(64, 4)
    loss, details = training.reinforce_loss(
        logits, logits.detach().clone(), valid, positives,
        generator=torch.Generator().manual_seed(61), actions_per_query=64,
        entropy_coefficient=0, kl_coefficient=0)
    assert details["sampled_actions"] == 4096
    assert 0 < details["sampled_reward_sum"] < 4096
    assert details["expected_reward_sum"] == pytest.approx(64 / 3, abs=1e-5)
    assert details["nonzero_sampled_advantages"] == 4096
    assert details["kl_to_supervised"] == pytest.approx(0)
    loss.backward()
    summed = logits.grad.sum(dim=0)
    assert torch.isfinite(summed).all()
    assert summed[0] < 0 and summed[1] > 0 and summed[2] > 0
    assert summed[3] == 0


def test_constant_reward_has_zero_score_function_gradient_without_regularization():
    logits = torch.tensor([[0.2, -0.1, 1.0]], requires_grad=True)
    valid = torch.tensor([[True, True, False]])
    loss, details = training.reinforce_loss(
        logits, logits.detach().clone(), valid, valid,
        generator=torch.Generator().manual_seed(1), entropy_coefficient=0, kl_coefficient=0)
    assert details["nonzero_sampled_advantages"] == 0
    assert details["sampled_reward_sum"] == details["sampled_actions"]
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


@pytest.mark.parametrize("valid,positives", [
    ([[False, False]], [[False, False]]),
    ([[True, True]], [[False, False]]),
    ([[True, False]], [[True, True]]),
])
def test_losses_reject_invalid_action_groups(valid, positives):
    with pytest.raises(ValueError, match="valid"):
        training.listwise_loss(torch.zeros((1, 2)), torch.tensor(valid), torch.tensor(positives))


def test_actual_coverage_catches_duplicates_missing_rows_and_wrong_population(corpus, tmp_path):
    ledger = training.CoverageLedger(corpus, np.arange(corpus.n_recipes))
    ledger.record(np.array([0]))
    with pytest.raises(RuntimeError, match="exactly once"):
        ledger.record(np.array([0]))
    with pytest.raises(RuntimeError, match="every"):
        ledger.finish()
    with pytest.raises(RuntimeError, match="exactly once"):
        ledger.record(np.array([1, 1]))
    ledger.record(np.arange(1, corpus.n_recipes))
    result = ledger.finish(tmp_path / "coverage.npz")
    assert result["every_catalog_row_exactly_once"]
    assert result["n_queries"] == corpus.n_recipes
    assert result["n_source_ingredient_slots"] == len(corpus.flat)
    assert result["n_optimizer_steps"] == 2
    with np.load(tmp_path / "coverage.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["seen_counts"], np.ones(corpus.n_recipes))
    partial = training.CoverageLedger(corpus, np.array([0, 1]))
    with pytest.raises(RuntimeError, match="exactly once"):
        partial.record(np.array([2]))
    partial.record(np.array([0, 1]))
    assert not partial.finish()["every_catalog_row_exactly_once"]


def test_coverage_keeps_singletons_pairs_and_all_98_ingredients():
    lengths = [1, 2, 3, 98]
    flat = np.concatenate([np.arange(length) for length in lengths]).astype(np.uint16)
    corpus = training.NumericCorpus(flat, np.r_[0, np.cumsum(lengths)],
                                    tuple(str(value) for value in range(98)), "fixture")
    ledger = training.CoverageLedger(corpus, np.arange(4))
    ledger.record(np.array([3, 0]))
    ledger.record(np.array([1, 2]))
    assert ledger.finish()["n_source_ingredient_slots"] == 104


def test_held_out_query_is_rejected_before_optimizer_can_change_weights(corpus, tmp_path, monkeypatch):
    sampler = training.QuerySampler(corpus, candidates=8, pool_size=32)
    original = sampler.batch

    def wrong_partition(rows, rng, *, split):
        return original(rows, rng, split="validation")

    monkeypatch.setattr(sampler, "batch", wrong_partition)
    policy = RecipeRankingPolicy(seed=19)
    before = [parameter.detach().clone() for parameter in policy.parameters()]
    with pytest.raises(RuntimeError, match="cannot enter an optimizer"):
        training.train_stage(
            policy, sampler, np.arange(16), stage="supervised", epochs=1,
            batch_size=16, seed=31, learning_rate=0.001, output=tmp_path)
    for expected, actual in zip(before, policy.parameters()):
        torch.testing.assert_close(expected, actual)


def test_tie_aware_recovery_does_not_award_source_position_an_advantage():
    cases = training.QueryBatch(
        np.zeros((2, 3, len(FEATURE_NAMES)), dtype=np.float32),
        np.array([[True, True, True], [True, True, False]]),
        np.array([[True, True, False], [True, False, False]]),
        np.array([[5, 4, 1], [2, 1, -1]]), np.array([5, 2]),
        np.array([0, 1]), np.array([0, 1, 2]), np.zeros(2, dtype="V16"),
        np.zeros(2, dtype=np.int16), np.full(2, np.nan), {})
    metrics, values = training.recovery_metrics(
        np.zeros((2, 3)), cases, seed=1, bootstrap_repetitions=100)
    np.testing.assert_allclose(values, [2 / 3, 1 / 2])
    assert metrics["all_queries"]["top_score_tie_fraction"] == 1
    assert metrics["all_queries"]["recipe_id_tiebroken_top1_recovery"] == 0
    assert metrics["all_queries"]["optimistic_top1_recovery"] == 1
    assert metrics["all_queries"]["pessimistic_top1_recovery"] == 0


def test_validation_selection_preserves_stronger_baseline_and_reports_negative_rl():
    selection = training.select_on_validation({
        "heuristic": np.ones(64) * 0.5, "supervised": np.ones(64),
        "reinforce": np.zeros(64),
    }, seed=7, bootstrap_repetitions=100)
    assert selection["selected"] == "supervised"
    assert selection["comparisons"][1]["mean_paired_recovery_difference"] == -1
    tied = training.select_on_validation(
        {name: np.ones(64) for name in ("heuristic", "supervised", "reinforce")},
        seed=7, bootstrap_repetitions=100)
    assert tied["selected"] == "heuristic"


def test_end_to_end_tiny_supervised_and_reinforce_pipeline(corpus, tmp_path):
    torch.set_num_threads(2)
    args = training.parser().parse_args([
        "--output", str(tmp_path), "--max-train-rows", str(corpus.n_recipes),
        "--batch-size", "16", "--candidates", "8", "--pool-size", "32",
        "--validation-queries", "12", "--test-queries", "12",
        "--bootstrap-repetitions", "100", "--hidden-dim", "8",
    ])
    sampler = training.QuerySampler(corpus, candidates=8, pool_size=32)
    report = training.experiment(
        corpus, sampler, output=tmp_path, population=np.arange(corpus.n_recipes),
        validation_rows=np.arange(12), test_rows=np.arange(12, 24), args=args)
    assert report["status"] == "completed"
    assert not report["production_or_human_preference_quality_established"]
    assert not report["model"]["pretrained_parameters_used"]
    assert report["evaluation"]["test_did_not_select_deployment"]
    assert report["training"]["supervised"]["n_queries"] == corpus.n_recipes
    reinforce = report["training"]["reinforce"]
    assert reinforce["n_queries"] == corpus.n_recipes
    assert reinforce["epochs"][0]["bandit"]["sampled_actions"] == corpus.n_recipes * 4
    assert reinforce["epochs"][0]["bandit"]["nonzero_sampled_advantages"] > 0
    assert reinforce["epochs"][0]["query_partition_bucket_counts"][8:] == [0, 0]
    assert reinforce["n_optimizer_steps"] == 6
    assert report["evaluation"]["validation"]["case_sha256"] != report["evaluation"]["test"]["case_sha256"]
    supervised = RecipeRankingPolicy.load(tmp_path / "supervised")
    learned = RecipeRankingPolicy.load(tmp_path / "reinforce")
    assert any(not torch.equal(first, second)
               for first, second in zip(supervised.parameters(), learned.parameters()))
    with np.load(tmp_path / "private_test_cases.npz", allow_pickle=False) as archive:
        assert all(archive[name].dtype.kind != "O" for name in archive.files)
    assert json.loads((tmp_path / "deployment.json").read_text())["selection"]["split"] == "validation"


def test_full_mode_cannot_relabel_a_sampling_cap_as_complete_training(tmp_path):
    args = training.parser().parse_args([
        "--output", str(tmp_path / "never_started"), "--mode", "full", "--max-train-rows", "100"])
    with pytest.raises(ValueError, match="sampling cap"):
        training.validate_args(args)
    args.max_train_rows = 0
    training.validate_args(args)
    args.threads = 5
    with pytest.raises(ValueError, match="4 CPU threads"):
        training.validate_args(args)


def test_canonical_loader_rejects_unpinned_manifest_before_loading_arrays(tmp_path):
    manifest = tmp_path / "GENERATION.json"
    manifest.write_text(json.dumps({"sha256": "a" * 64, "recipes": 3}))
    with pytest.raises(ValueError, match="pinned complete canonical"):
        training.NumericCorpus.load_canonical(tmp_path / "missing.npz", manifest)
