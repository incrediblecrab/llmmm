"""Own-parameter ranking of an already feasible recipe shortlist.

This module does not retrieve recipes or enforce, weaken, or infer constraints.
Callers must exclude infeasible candidates before ranking. ``None`` is the only
public representation of unknown time; zero is a known time, not an imputation.
Ingredient inputs have set semantics, so repeated IDs do not increase overlap.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence, Set
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors
from torch import nn

FEATURE_VERSION = 1
FEATURE_DEFINITIONS = {
    "overlap_log_fraction": "log1p(|A intersect R|) / log1p(vocabulary size)",
    "missing_log_fraction": "log1p(|R minus A|) / log1p(vocabulary size)",
    "recipe_size_log_fraction": "log1p(|R|) / log1p(vocabulary size)",
    "pantry_size_log_fraction": "log1p(|A|) / log1p(vocabulary size)",
    "recipe_coverage": "|A intersect R| / |R|",
    "pantry_coverage": "|A intersect R| / |A|",
    "jaccard": "|A intersect R| / |A union R|",
    "idf_recipe_coverage": "sum IDF(A intersect R) / sum IDF(R)",
    "idf_pantry_coverage": "sum IDF(A intersect R) / sum IDF(A)",
    "idf_jaccard": "sum IDF(A intersect R) / sum IDF(A union R)",
    "missing_fraction": "|R minus A| / |R|",
    "unused_pantry_fraction": "|A minus R| / |A|",
    "exact_set_match": "1 if A equals R, else 0",
    "fully_available": "1 if R is a subset of A, else 0",
    "time_known": "1 if total_minutes is provided, else 0",
    "budget_known": "1 if max_total_minutes is provided, else 0",
    "time_log_fraction": "log1p(min(total_minutes, 10080)) / log1p(10080); 0 if unknown",
    "time_budget_fraction": "min(total_minutes, budget) / budget; 0 unless both known",
    "time_budget_slack": "1 - time_budget_fraction; 0 unless both known",
    "within_time_budget": "1 if both known and total_minutes <= budget, else 0",
}
FEATURE_NAMES = tuple(FEATURE_DEFINITIONS)
IDF_DEFINITION = "IDF(i) = 1 + log((1 + n_recipes) / (1 + document_frequency(i)))"
TIME_FEATURE_START = FEATURE_NAMES.index("time_known")
MAX_CANDIDATES = 10_000
MAX_VOCABULARY = 65_536
MAX_INPUT_SLOTS = 2_000_000
CONFIG_FILENAME = "recipe_ranker_config.json"
WEIGHTS_FILENAME = "recipe_ranker.safetensors"
_MAX_WEIGHTS_BYTES = 4 * 1024 * 1024


def _positive_integer(value: object, name: str, maximum: int) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not 1 <= value <= maximum):
        raise ValueError(f"{name} must be an integer between 1 and {maximum}")
    return int(value)


def _ingredient_ids(values: Sequence[int], n_vocab: int, name: str) -> np.ndarray:
    if (isinstance(values, (str, bytes))
            or not isinstance(values, (Sequence, Set, np.ndarray))):
        raise ValueError(f"{name} must be a bounded sequence of integer ingredient IDs")
    if isinstance(values, np.ndarray):
        if values.ndim != 1 or values.dtype.kind not in "iu":
            raise ValueError(f"{name} must contain one-dimensional integer IDs, not booleans")
        ids = values
    else:
        if any(isinstance(value, (bool, np.bool_))
               or not isinstance(value, (int, np.integer)) for value in values):
            raise ValueError(f"{name} must contain integer IDs, not booleans")
        ids = np.asarray(list(values))
    if not 1 <= len(ids) <= MAX_INPUT_SLOTS:
        raise ValueError(f"{name} must be nonempty and contain at most {MAX_INPUT_SLOTS} IDs")
    if (ids < 0).any() or (ids >= n_vocab).any():
        raise ValueError(f"{name} contains an unknown ingredient ID outside [0, {n_vocab})")
    return np.unique(ids.astype(np.int64, copy=False))


def _finite_minutes(value: object, name: str, *, positive: bool = False) -> float:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return number


def _features_from_statistics(
        overlap: np.ndarray, recipe_size: np.ndarray, pantry_size: np.ndarray,
        idf_overlap: np.ndarray, idf_recipe: np.ndarray, idf_pantry: np.ndarray, *,
        n_vocab: int, total_minutes: np.ndarray | None = None,
        max_total_minutes: float | np.ndarray | None = None) -> np.ndarray:
    """Shared vectorized kernel; only callers with validated statistics may use it."""
    overlap, recipe_size, pantry_size, idf_overlap, idf_recipe, idf_pantry = (
        np.broadcast_arrays(overlap, recipe_size, pantry_size,
                            idf_overlap, idf_recipe, idf_pantry))
    shape = overlap.shape
    missing = recipe_size - overlap
    normalizer = np.log1p(n_vocab)
    minutes = np.broadcast_to(
        np.nan if total_minutes is None else total_minutes, shape)
    budget = np.broadcast_to(
        np.nan if max_total_minutes is None else max_total_minutes, shape)
    known, budget_known = np.isfinite(minutes), np.isfinite(budget)
    both_known = known & budget_known
    safe_minutes = np.where(known, minutes, 0.0)
    safe_budget = np.where(budget_known, budget, 1.0)
    time_fraction = np.where(
        both_known, np.minimum(safe_minutes, safe_budget) / safe_budget, 0.0)
    values = (
        np.log1p(overlap) / normalizer,
        np.log1p(missing) / normalizer,
        np.log1p(recipe_size) / normalizer,
        np.log1p(pantry_size) / normalizer,
        overlap / recipe_size,
        overlap / pantry_size,
        overlap / (recipe_size + pantry_size - overlap),
        idf_overlap / idf_recipe,
        idf_overlap / idf_pantry,
        idf_overlap / (idf_recipe + idf_pantry - idf_overlap),
        missing / recipe_size,
        (pantry_size - overlap) / pantry_size,
        (overlap == recipe_size) & (overlap == pantry_size),
        missing == 0,
        known,
        budget_known,
        np.log1p(np.minimum(safe_minutes, 10_080.0)) / np.log1p(10_080.0),
        time_fraction,
        np.where(both_known, 1.0 - time_fraction, 0.0),
        both_known & (minutes <= budget),
    )
    # Roundoff in weighted set sums can put an exact ratio just above one.
    return np.clip(np.stack(values, axis=-1), 0.0, 1.0).astype(np.float32)


def candidate_features(
        available_ids: Sequence[int],
        candidate_ingredient_ids: Sequence[Sequence[int]], *,
        ingredient_frequency: np.ndarray, n_recipes: int,
        total_minutes: Sequence[float | None] | None = None,
        max_total_minutes: float | None = None) -> np.ndarray:
    """Return finite float32 ``[n_candidates, len(FEATURE_NAMES)]`` in [0, 1].

    ``ingredient_frequency`` contains per-recipe document counts aligned to the
    vocabulary IDs, not probabilities or raw repeated-ingredient counts. Empty
    shortlists are supported, but an empty pantry or recipe is an input error.
    Work is bounded to 10,000 candidates and 2,000,000 ingredient input slots.
    Unknown times have explicit indicator features; they are never treated as
    evidence of satisfying a time constraint.
    """
    n_recipes = _positive_integer(n_recipes, "n_recipes", 2**53)
    if (not isinstance(ingredient_frequency, np.ndarray)
            or ingredient_frequency.ndim != 1
            or ingredient_frequency.dtype.kind not in "iuf"
            or not 1 <= len(ingredient_frequency) <= MAX_VOCABULARY):
        raise ValueError("ingredient_frequency must be a bounded one-dimensional numeric array")
    frequency = ingredient_frequency.astype(np.float64, copy=False)
    if (not np.isfinite(frequency).all() or (frequency < 0).any()
            or (frequency > n_recipes).any() or (frequency != np.floor(frequency)).any()):
        raise ValueError("ingredient_frequency must contain finite integer counts in [0, n_recipes]")
    n_vocab = len(frequency)
    available = _ingredient_ids(available_ids, n_vocab, "available_ids")
    if (isinstance(candidate_ingredient_ids, (str, bytes))
            or not isinstance(candidate_ingredient_ids, (Sequence, np.ndarray))
            or (isinstance(candidate_ingredient_ids, np.ndarray) and candidate_ingredient_ids.ndim == 0)
            or len(candidate_ingredient_ids) > MAX_CANDIDATES):
        raise ValueError(f"candidate_ingredient_ids must contain at most {MAX_CANDIDATES} recipes")
    n_candidates = len(candidate_ingredient_ids)
    budget = (None if max_total_minutes is None else
              _finite_minutes(max_total_minutes, "max_total_minutes", positive=True))
    minutes = np.full(n_candidates, np.nan, dtype=np.float64)
    if total_minutes is not None:
        if (isinstance(total_minutes, (str, bytes))
                or not isinstance(total_minutes, (Sequence, np.ndarray))
                or (isinstance(total_minutes, np.ndarray) and total_minutes.ndim != 1)
                or len(total_minutes) != n_candidates):
            raise ValueError("total_minutes must align one-to-one with the candidates")
        for index, value in enumerate(total_minutes):
            if value is not None:
                minutes[index] = _finite_minutes(value, f"total_minutes[{index}]")
    ids = []
    input_slots = len(available_ids)
    for index, values in enumerate(candidate_ingredient_ids):
        if (not hasattr(values, "__len__")
                or (isinstance(values, np.ndarray) and values.ndim != 1)):
            raise ValueError(f"candidate_ingredient_ids[{index}] must be a sequence")
        input_slots += len(values)
        if input_slots > MAX_INPUT_SLOTS:
            raise ValueError(f"ingredient inputs exceed the {MAX_INPUT_SLOTS}-slot bound")
        ids.append(_ingredient_ids(values, n_vocab, f"candidate_ingredient_ids[{index}]"))
    if not n_candidates:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    sizes = np.fromiter((len(values) for values in ids), dtype=np.int64)
    starts = np.r_[0, np.cumsum(sizes[:-1])]
    flat = np.concatenate(ids)
    pantry = np.zeros(n_vocab, dtype=bool)
    pantry[available] = True
    matched = pantry[flat]
    idf = 1.0 + np.log1p(n_recipes) - np.log1p(frequency)
    return _features_from_statistics(
        np.add.reduceat(matched.astype(np.int64), starts), sizes,
        np.full(n_candidates, len(available)),
        np.add.reduceat(matched * idf[flat], starts),
        np.add.reduceat(idf[flat], starts), np.full(n_candidates, idf[available].sum()),
        n_vocab=n_vocab, total_minutes=minutes, max_total_minutes=budget)


def _feature_array(features: np.ndarray) -> np.ndarray:
    if (not isinstance(features, np.ndarray) or features.ndim != 2
            or features.shape[1] != len(FEATURE_NAMES)
            or features.shape[0] > MAX_CANDIDATES or features.dtype.kind not in "fiu"):
        raise ValueError(
            f"features must be a numeric [N, {len(FEATURE_NAMES)}] array, N <= {MAX_CANDIDATES}")
    if (not np.isfinite(features).all() or (features < 0).any() or (features > 1).any()):
        raise ValueError("features must be finite values in [0, 1]")
    return features


def deterministic_baseline_score(features: np.ndarray) -> np.ndarray:
    """A fixed IDF/Jaccard baseline, not learned or described as reinforcement learning."""
    values = _feature_array(features).astype(np.float64, copy=False)
    index = {name: position for position, name in enumerate(FEATURE_NAMES)}
    return (
        3.0 * values[:, index["idf_jaccard"]]
        + 2.0 * values[:, index["jaccard"]]
        + 0.25 * values[:, index["recipe_coverage"]]
        + 0.25 * values[:, index["pantry_coverage"]]
        - 0.5 * values[:, index["missing_fraction"]]
        + 0.1 * values[:, index["fully_available"]]
        + 0.025 * values[:, index["time_budget_slack"]])


heuristic_scores = deterministic_baseline_score


def _json_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


class RecipeRankingPolicy(nn.Module):
    """A tiny tanh MLP, initialized from scratch, with no ingredient embeddings.

    Time inputs are disabled by default. Training without real time metadata
    must leave them disabled rather than deploying random, untrained time
    coefficients. Scores are uncalibrated ranking logits, not probabilities of
    human satisfaction. Serialization accepts only this exact JSON schema and
    float32 safetensors state, never pickle.
    """

    def __init__(self, hidden_dim: int = 32, *, seed: int = 0,
                 time_features_enabled: bool = False):
        super().__init__()
        self.hidden_dim = _positive_integer(hidden_dim, "hidden_dim", 256)
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if type(time_features_enabled) is not bool:
            raise ValueError("time_features_enabled must be boolean")
        self.time_features_enabled = time_features_enabled
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.network = nn.Sequential(
                nn.Linear(len(FEATURE_NAMES), self.hidden_dim, dtype=torch.float32, device="cpu"),
                nn.Tanh(), nn.Linear(self.hidden_dim, 1, dtype=torch.float32, device="cpu"))
        feature_mask = torch.ones(len(FEATURE_NAMES), dtype=torch.float32, device="cpu")
        if not time_features_enabled:
            feature_mask[TIME_FEATURE_START:] = 0
        self.register_buffer("feature_mask", feature_mask)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if (not isinstance(features, torch.Tensor) or features.ndim < 2
                or features.shape[-1] != len(FEATURE_NAMES)
                or not features.is_floating_point()):
            raise ValueError(f"features must be floating tensors with final dimension {len(FEATURE_NAMES)}")
        if not torch.isfinite(features).all() or (features < 0).any() or (features > 1).any():
            raise ValueError("features must be finite values in [0, 1]")
        return self.network(features * self.feature_mask).squeeze(-1)

    @torch.inference_mode()
    def score(self, features: np.ndarray) -> np.ndarray:
        features = _feature_array(features)
        if not len(features):
            return np.empty(0, dtype=np.float32)
        device = self.feature_mask.device
        results = [
            self(torch.tensor(np.ascontiguousarray(features[start:start + 1024]),
                              dtype=torch.float32, device=device))
            .detach().cpu().numpy()
            for start in range(0, len(features), 1024)
        ]
        scores = np.concatenate(results)
        if not np.isfinite(scores).all():
            raise ValueError("the recipe ranking policy produced non-finite scores")
        return scores

    def _configuration(self, weights_sha256: str) -> dict:
        return {
            "schema_version": 1,
            "model_type": "recipe-ranking-mlp",
            "feature_version": FEATURE_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "hidden_dim": self.hidden_dim,
            "activation": "tanh",
            "dtype": "float32",
            "time_features_enabled": self.time_features_enabled,
            "weights_file": WEIGHTS_FILENAME,
            "weights_sha256": weights_sha256,
        }

    def save(self, path: str | Path) -> None:
        """Write a new checkpoint; refuse to overwrite either existing policy file."""
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        weights_path, config_path = destination / WEIGHTS_FILENAME, destination / CONFIG_FILENAME
        if weights_path.exists() or config_path.exists():
            raise FileExistsError(f"{destination}: refusing to overwrite an existing ranking policy")
        tensors = {}
        for name, value in self.state_dict().items():
            tensor = value.detach().cpu().contiguous()
            if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                raise ValueError("recipe ranking weights must be finite float32 tensors")
            tensors[name] = tensor
        expected_mask = torch.ones(len(FEATURE_NAMES), dtype=torch.float32, device="cpu")
        if not self.time_features_enabled:
            expected_mask[TIME_FEATURE_START:] = 0
        if not torch.equal(tensors["feature_mask"], expected_mask):
            raise ValueError("feature mask does not match the policy configuration")
        data = save_tensors(tensors, metadata={"format": "llmmm-recipe-ranker", "schema_version": "1"})
        config = self._configuration(hashlib.sha256(data).hexdigest())
        with weights_path.open("xb") as stream:
            stream.write(data)
        with config_path.open("x", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2, allow_nan=False)
            stream.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> "RecipeRankingPolicy":
        source = Path(path)
        config_path, weights_path = source / CONFIG_FILENAME, source / WEIGHTS_FILENAME
        if not config_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(f"{source}: both strict JSON configuration and safetensors are required")
        if config_path.stat().st_size > 64 * 1024:
            raise ValueError("recipe ranking configuration is too large")

        def reject_constant(value: str):
            raise ValueError(f"non-finite JSON constant: {value}")

        config = json.loads(config_path.read_text(encoding="utf-8"),
                            object_pairs_hook=_json_object, parse_constant=reject_constant)
        if not isinstance(config, dict):
            raise ValueError("recipe ranking configuration must be a JSON object")
        keys = set(cls()._configuration(""))
        if set(config) != keys:
            raise ValueError("recipe ranking configuration has missing or unexpected schema fields")
        expected = {
            "schema_version": 1, "model_type": "recipe-ranking-mlp",
            "feature_version": FEATURE_VERSION, "feature_names": list(FEATURE_NAMES),
            "activation": "tanh", "dtype": "float32", "weights_file": WEIGHTS_FILENAME,
        }
        if any(config[key] != value or type(config[key]) is not type(value)
               for key, value in expected.items()):
            raise ValueError("unsupported recipe ranking schema, architecture, or feature definitions")
        digest = config["weights_sha256"]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise ValueError("weights_sha256 must be a lowercase SHA256 digest")
        model = cls(hidden_dim=config["hidden_dim"],
                    time_features_enabled=config["time_features_enabled"])
        if not 1 <= weights_path.stat().st_size <= _MAX_WEIGHTS_BYTES:
            raise ValueError("recipe ranking weights have an invalid file size")
        data = weights_path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("recipe ranking weights SHA256 does not match the configuration")
        state = load_tensors(data)
        expected_state = model.state_dict()
        if set(state) != set(expected_state):
            raise ValueError("recipe ranking weights contain missing or unexpected tensors")
        for name, expected_tensor in expected_state.items():
            tensor = state[name]
            if (tensor.shape != expected_tensor.shape or tensor.dtype != torch.float32
                    or not torch.isfinite(tensor).all()):
                raise ValueError(f"recipe ranking tensor {name!r} has invalid shape, dtype, or values")
        if not torch.equal(state["feature_mask"], expected_state["feature_mask"]):
            raise ValueError("serialized feature mask does not match the configuration")
        model.load_state_dict(state, strict=True)
        return model.eval()
