"""Train an own-parameter recipe ranker in an offline synthetic bandit simulator.

This is NOT training on logged human preferences. The reward is exactly one
when a selected candidate's complete canonical ingredient set equals the source
recipe's set, and zero otherwise. All identical sets are equivalent positives.
The source is inserted into a sampled feasible shortlist: evaluation here does
not measure full-catalog retrieval recall or production/user-preference quality.

Every selected source row generates one accepted training query per epoch.
Pantry sets have a source-independent, deterministic 80/10/10 hash partition:
train / validation / test. Thus all catalog rows can be training sources without
training on validation/test pantry contexts. This is held-out synthetic-query
evaluation over a KNOWN catalog, not unseen-recipe or ingredient-set-disjoint
generalization. A full run has no source-row sampling or length exclusions.

Warm-start maximizes listwise probability assigned to equivalent positives.
REINFORCE then samples actions from the current policy, observes simulator
rewards, and applies a score-function policy gradient with a detached per-query
expected-reward baseline, entropy bonus, and KL penalty to the frozen supervised
policy. This is on-policy contextual-bandit RL in an offline simulator, not
offline RL from human interaction logs. Validation alone selects deployment.
"""
from __future__ import annotations

import os

for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "4"

import argparse
import copy
import hashlib
import json
import math
import platform
import resource
import sqlite3
import sys
import time
from collections import Counter
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import safetensors
import torch

import ingredient_model.recipe_ranker as ranker_module
from ingredient_model.recipe_ranker import (
    FEATURE_DEFINITIONS,
    FEATURE_NAMES,
    IDF_DEFINITION,
    MAX_CANDIDATES,
    RecipeRankingPolicy,
    _features_from_statistics,
    deterministic_baseline_score,
)

CANONICAL_SHA256 = "414c4134035378e620e542633b18f71f086c192be2041fcbac9116f25d0758c9"
CANONICAL_RECIPES = 4_653_430
CANONICAL_SLOTS = 36_707_624
CANONICAL_VOCAB = 1_790
TIE_ATOL = 1e-6
_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict | list) -> None:
    pending = path.with_name(path.name + ".pending")
    with pending.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    pending.replace(path)


def array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(value.shape).encode())
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


@dataclass
class NumericCorpus:
    flat: np.ndarray
    offsets: np.ndarray
    vocabulary: tuple[str, ...]
    corpus_sha256: str
    generation_sha256: str | None = None
    sizes: np.ndarray = field(init=False)
    frequency: np.ndarray = field(init=False)
    idf: np.ndarray = field(init=False)
    idf_sums: np.ndarray = field(init=False)
    posting_rows: np.ndarray | None = field(default=None, init=False)
    posting_offsets: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self):
        if (not isinstance(self.flat, np.ndarray) or self.flat.ndim != 1
                or self.flat.dtype.kind not in "iu" or not len(self.flat)):
            raise ValueError("flat must be a nonempty one-dimensional integer array")
        if (not isinstance(self.offsets, np.ndarray) or self.offsets.ndim != 1
                or self.offsets.dtype.kind not in "iu" or len(self.offsets) < 2
                or self.offsets[0] != 0 or self.offsets[-1] != len(self.flat)):
            raise ValueError("offsets must delimit the complete flat array")
        if (not self.vocabulary or len(self.vocabulary) > 65_536
                or len(set(self.vocabulary)) != len(self.vocabulary)
                or not all(isinstance(value, str) and value for value in self.vocabulary)):
            raise ValueError("vocabulary must contain unique nonempty names")
        if self.n_recipes >= 2**31 or (self.flat < 0).any() or (self.flat >= self.n_vocab).any():
            raise ValueError("corpus rows or ingredient IDs exceed numeric bounds")
        self.offsets = self.offsets.astype(np.int64, copy=False)
        self.sizes = np.diff(self.offsets)
        if (self.sizes < 1).any():
            raise ValueError("every corpus row must be nonempty; no silent length exclusions")
        invalid = self.flat[1:] <= self.flat[:-1]
        invalid[self.offsets[1:-1] - 1] = False
        if invalid.any():
            raise ValueError("each canonical ingredient row must be a strictly sorted set")
        del invalid
        self.frequency = np.zeros(self.n_vocab, dtype=np.int64)
        for start in range(0, len(self.flat), 1_000_000):
            self.frequency += np.bincount(
                self.flat[start:start + 1_000_000], minlength=self.n_vocab)
        self.idf = 1.0 + np.log1p(self.n_recipes) - np.log1p(self.frequency)
        self.idf_sums = np.empty(self.n_recipes, dtype=np.float64)
        for start in range(0, self.n_recipes, 100_000):
            stop = min(start + 100_000, self.n_recipes)
            lower, upper = self.offsets[start], self.offsets[stop]
            self.idf_sums[start:stop] = np.add.reduceat(
                self.idf[self.flat[lower:upper]], self.offsets[start:stop] - lower)

    @property
    def n_recipes(self) -> int:
        return len(self.offsets) - 1

    @property
    def n_vocab(self) -> int:
        return len(self.vocabulary)

    def recipe(self, row: int) -> np.ndarray:
        return self.flat[self.offsets[row]:self.offsets[row + 1]]

    @classmethod
    def load_canonical(cls, path: Path, generation: Path) -> "NumericCorpus":
        manifest = json.loads(generation.read_text(encoding="utf-8"))
        expected = {"sha256": CANONICAL_SHA256, "recipes": CANONICAL_RECIPES,
                    "slots": CANONICAL_SLOTS, "vocab": CANONICAL_VOCAB}
        if not isinstance(manifest, dict) or any(
                manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("GENERATION.json does not identify the pinned complete canonical corpus")
        actual_hash = sha256_file(path)
        if actual_hash != CANONICAL_SHA256:
            raise ValueError("recipe_ids.npz SHA256 differs from the pinned GENERATION.json")
        with np.load(path, allow_pickle=False) as archive:
            # Deliberately never read the millions of lang/source strings.
            flat, offsets, vocabulary = archive["flat"], archive["offsets"], archive["itos"]
        if vocabulary.dtype.kind not in "US" or vocabulary.ndim != 1:
            raise ValueError("canonical vocabulary must be a non-object string array")
        corpus = cls(flat, offsets, tuple(str(value) for value in vocabulary),
                     actual_hash, sha256_file(generation))
        if (corpus.n_recipes != CANONICAL_RECIPES or len(flat) != CANONICAL_SLOTS
                or corpus.n_vocab != CANONICAL_VOCAB or flat.dtype != np.uint16
                or corpus.sizes.min() != 1 or corpus.sizes.max() != 98):
            raise ValueError("canonical numeric arrays do not match the pinned full-corpus dimensions")
        return corpus

    def build_postings(self, cache_path: Path | None = None) -> dict:
        """One numeric inverted index; no Python recipe objects or SQL per query."""
        started = time.perf_counter()
        if cache_path is not None and (cache_path / "index.json").exists():
            metadata = json.loads((cache_path / "index.json").read_text(encoding="utf-8"))
            if (metadata.get("schema_version") != 1
                    or metadata.get("corpus_sha256") != self.corpus_sha256
                    or metadata.get("n_slots") != len(self.flat)
                    or metadata.get("n_recipes") != self.n_recipes):
                raise ValueError("numeric postings cache belongs to a different corpus")
            for filename in ("posting_rows.npy", "posting_offsets.npy"):
                path = cache_path / filename
                if sha256_file(path) != metadata.get("sha256", {}).get(filename):
                    raise ValueError(f"{filename}: numeric postings cache checksum mismatch")
            rows = np.load(cache_path / "posting_rows.npy", mmap_mode="r", allow_pickle=False)
            offsets = np.load(cache_path / "posting_offsets.npy", mmap_mode="r", allow_pickle=False)
            if (rows.shape != (len(self.flat),) or rows.dtype != np.int32
                    or offsets.shape != (self.n_vocab + 1,) or offsets.dtype != np.int64
                    or not np.array_equal(np.diff(offsets), self.frequency)
                    or offsets[0] != 0 or offsets[-1] != len(self.flat)
                    or rows.min() < 0 or rows.max() >= self.n_recipes):
                raise ValueError("numeric postings cache arrays are invalid")
            self.posting_rows, self.posting_offsets = rows, offsets
            return {"cache_reused": True, "seconds": time.perf_counter() - started,
                    "n_slots_indexed": len(rows), "bytes": rows.nbytes + offsets.nbytes}
        if cache_path is not None:
            cache_path.mkdir(parents=True, exist_ok=True, mode=0o700)
            if any(cache_path.iterdir()):
                raise FileExistsError("refusing to overwrite an incomplete numeric postings cache")
        order = np.argsort(self.flat, kind="stable")
        slot_rows = np.repeat(np.arange(self.n_recipes, dtype=np.int32), self.sizes)
        rows = slot_rows[order]
        del order, slot_rows
        offsets = np.r_[np.int64(0), np.cumsum(self.frequency)]
        if cache_path is not None:
            np.save(cache_path / "posting_rows.npy", rows, allow_pickle=False)
            np.save(cache_path / "posting_offsets.npy", offsets, allow_pickle=False)
            write_json(cache_path / "index.json", {
                "schema_version": 1, "corpus_sha256": self.corpus_sha256,
                "n_slots": len(self.flat), "n_recipes": self.n_recipes,
                "sha256": {name: sha256_file(cache_path / name)
                           for name in ("posting_rows.npy", "posting_offsets.npy")},
            })
            del rows
            rows = np.load(cache_path / "posting_rows.npy", mmap_mode="r", allow_pickle=False)
        self.posting_rows, self.posting_offsets = rows, offsets
        return {"cache_reused": False, "seconds": time.perf_counter() - started,
                "n_slots_indexed": len(rows), "bytes": rows.nbytes + offsets.nbytes}


def load_catalog_times(path: Path, corpus: NumericCorpus) -> tuple[np.ndarray, dict]:
    """Read source-provided positive totals only; never impute from components."""
    started = time.perf_counter()
    times = np.full(corpus.n_recipes, np.nan, dtype=np.float64)
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        metadata = dict(connection.execute(
            "SELECT key, value FROM metadata WHERE key IN "
            "('schema_version', 'corpus_sha256', 'partial', 'n_recipes')"))
        metadata = {key: json.loads(value) for key, value in metadata.items()}
        if (metadata.get("schema_version") != 1 or metadata.get("partial") is not False
                or metadata.get("corpus_sha256") != corpus.corpus_sha256
                or metadata.get("n_recipes") != corpus.n_recipes):
            raise ValueError("time metadata must come from the matching complete canonical catalog")
        cursor = connection.execute("SELECT id, total_minutes, time_status FROM recipes ORDER BY id")
        position = 0
        while records := cursor.fetchmany(50_000):
            for row, minutes, time_status in records:
                if type(row) is not int or row != position or row >= corpus.n_recipes:
                    raise ValueError("catalog time rows must cover every canonical ID exactly once")
                if minutes is not None:
                    if (isinstance(minutes, bool) or not isinstance(minutes, (int, float))
                            or not math.isfinite(minutes) or minutes <= 0
                            or time_status != "source_total"):
                        raise ValueError("catalog known time requires a finite positive source_total")
                    times[row] = minutes
                position += 1
        if position != corpus.n_recipes:
            raise ValueError("catalog time metadata does not cover every canonical row")
    return times, {
        "enabled": bool(np.isfinite(times).any()), "n_known": int(np.isfinite(times).sum()),
        "n_unknown": int(np.isnan(times).sum()), "seconds": time.perf_counter() - started,
        "numeric_time_array_sha256": array_digest(times),
        "semantics": (
            "Finite positive source-provided total minutes with time_status=source_total only; "
            "prep/cook/derived totals are never substituted. Unknown remains NaN internally / None publicly."),
    }


def pantry_signature(available_ids: np.ndarray) -> bytes:
    ids = np.asarray(available_ids)
    if (ids.ndim != 1 or not len(ids) or ids.dtype.kind not in "iu"
            or (ids < 0).any() or (ids > 65_535).any()):
        raise ValueError("pantry signatures require nonempty bounded integer IDs")
    return hashlib.blake2b(np.unique(ids).astype("<u2").tobytes(),
                           digest_size=16, person=b"llmmm-query-v1").digest()


def query_partition(signature: bytes) -> str:
    if not isinstance(signature, bytes) or len(signature) != 16:
        raise ValueError("query partition requires a 16-byte pantry signature")
    bucket = int.from_bytes(signature[:8], "little") % 10
    return "train" if bucket < 8 else ("validation" if bucket == 8 else "test")


@dataclass
class QueryBatch:
    features: np.ndarray
    valid: np.ndarray
    positives: np.ndarray
    candidate_rows: np.ndarray
    source_rows: np.ndarray
    pantry_flat: np.ndarray
    pantry_offsets: np.ndarray
    signatures: np.ndarray
    max_missing: np.ndarray
    budgets: np.ndarray
    statistics: dict

    def digest(self) -> str:
        return array_digest(
            self.features, self.valid, self.positives, self.candidate_rows, self.source_rows,
            self.pantry_flat, self.pantry_offsets, self.signatures, self.max_missing, self.budgets)


class QuerySampler:
    def __init__(self, corpus: NumericCorpus, *, candidates: int = 16,
                 pool_size: int = 128, max_missing: int = 2, max_noise: int = 4,
                 times: np.ndarray | None = None):
        if (type(candidates) is not int or type(pool_size) is not int
                or not 2 <= candidates <= 64 or not candidates <= pool_size <= 1024):
            raise ValueError("require 2 <= candidates <= 64 and candidates <= pool_size <= 1024")
        if (type(max_missing) is not int or not 0 <= max_missing <= 8
                or type(max_noise) is not int or not 1 <= max_noise <= 16):
            raise ValueError("max_missing must be in [0, 8] and max_noise in [1, 16]")
        if corpus.posting_rows is None or corpus.posting_offsets is None:
            raise ValueError("build the numeric postings index before sampling")
        if times is not None and (
                times.shape != (corpus.n_recipes,) or times.dtype.kind != "f"
                or np.isinf(times).any() or (times[np.isfinite(times)] < 0).any()):
            raise ValueError("time metadata must align to canonical rows and use NaN only for unknowns")
        self.corpus, self.candidates, self.pool_size = corpus, candidates, pool_size
        self.max_missing, self.max_noise, self.times = max_missing, max_noise, times
        probability = np.sqrt(corpus.frequency.astype(np.float64))
        self.observed_vocabulary_size = int(np.count_nonzero(probability))
        self.noise_cdf = np.cumsum(probability / probability.sum())
        self.noise_cdf[-1] = 1.0

    def _pantries(self, rows: np.ndarray, rng: np.random.Generator, split: str,
                  used_signatures: set[bytes] | None) -> tuple:
        if split not in ("train", "validation", "test"):
            raise ValueError("unknown query split")
        pantries, signatures, limits, budgets = [], [], [], []
        attempts = 0
        for row in rows:
            source = self.corpus.recipe(int(row))
            source_set = set(source.tolist())
            for _ in range(1024):
                attempts += 1
                limit = int(rng.integers(self.max_missing + 1))
                missing = int(rng.integers(min(limit, len(source) - 1) + 1))
                retained = np.ones(len(source), dtype=bool)
                if missing:
                    retained[rng.choice(len(source), size=missing, replace=False)] = False
                noise_count = min(int(rng.integers(self.max_noise + 1)),
                                  self.observed_vocabulary_size - len(source))
                noise = set()
                for _ in range(32):
                    if len(noise) >= noise_count:
                        break
                    proposed = np.searchsorted(
                        self.noise_cdf, rng.random(2 * noise_count + 8), side="right")
                    for ingredient in proposed:
                        if int(ingredient) not in source_set:
                            noise.add(int(ingredient))
                            if len(noise) == noise_count:
                                break
                if len(noise) != noise_count:
                    remaining = np.ones(self.corpus.n_vocab, dtype=bool)
                    remaining[source] = False
                    if noise:
                        remaining[list(noise)] = False
                    candidates = np.flatnonzero(remaining & (self.corpus.frequency > 0))
                    if len(candidates) < noise_count - len(noise):
                        raise RuntimeError("not enough observed non-source ingredients for pantry noise")
                    weights = np.sqrt(self.corpus.frequency[candidates])
                    noise.update(rng.choice(candidates, size=noise_count - len(noise),
                                            replace=False, p=weights / weights.sum()).tolist())
                pantry = np.unique(np.r_[source[retained], np.asarray(sorted(noise), dtype=np.int64)])
                signature = pantry_signature(pantry)
                if query_partition(signature) != split:
                    continue
                if used_signatures is not None and signature in used_signatures:
                    continue
                budget = np.nan
                if (self.times is not None and np.isfinite(self.times[row])
                        and rng.random() < 0.5):
                    source_time = self.times[row]
                    budget = max(1.0, min(source_time, np.finfo(np.float64).max / 2)
                                 * float(rng.uniform(1.0, 1.5)))
                    if budget < source_time:
                        budget = source_time
                pantries.append(pantry)
                signatures.append(signature)
                limits.append(limit)
                budgets.append(budget)
                if used_signatures is not None:
                    used_signatures.add(signature)
                break
            else:
                raise RuntimeError(
                    f"source row {int(row)} cannot supply a unique {split} pantry after 1024 draws; "
                    "no source row has been silently skipped")
        return (pantries, np.asarray(signatures, dtype="V16"),
                np.asarray(limits, dtype=np.int16), np.asarray(budgets), attempts)

    def _pool_features(self, pantries: list[np.ndarray], rows: np.ndarray,
                       source_rows: np.ndarray, budgets: np.ndarray) -> tuple:
        corpus = self.corpus
        batch_size, width = rows.shape
        pantry_mask = np.zeros((batch_size, corpus.n_vocab), dtype=bool)
        source_mask = np.zeros_like(pantry_mask)
        for index, (pantry, source_row) in enumerate(zip(pantries, source_rows)):
            pantry_mask[index, pantry] = True
            source_mask[index, corpus.recipe(int(source_row))] = True
        flat_rows = rows.reshape(-1)
        lengths = corpus.sizes[flat_rows]
        n_slots = int(lengths.sum())
        if n_slots > 4_000_000:
            raise ValueError("ingredient pool batch exceeds 4000000 slots; reduce batch_size")
        starts = np.r_[np.int64(0), np.cumsum(lengths[:-1])]
        owner = np.repeat(np.arange(len(flat_rows), dtype=np.int32), lengths)
        positions = (np.repeat(corpus.offsets[flat_rows] - starts, lengths)
                     + np.arange(n_slots, dtype=np.int64))
        ingredients = corpus.flat[positions]
        query = owner // width
        matched = pantry_mask[query, ingredients]
        overlap = np.bincount(owner, weights=matched, minlength=len(flat_rows)).reshape(rows.shape)
        weighted_overlap = np.bincount(
            owner, weights=matched * corpus.idf[ingredients],
            minlength=len(flat_rows)).reshape(rows.shape)
        source_overlap = np.bincount(
            owner, weights=source_mask[query, ingredients],
            minlength=len(flat_rows)).reshape(rows.shape)
        positives = ((source_overlap == lengths.reshape(rows.shape))
                     & (lengths.reshape(rows.shape) == corpus.sizes[source_rows, None]))
        features = _features_from_statistics(
            overlap, lengths.reshape(rows.shape), pantry_mask.sum(axis=1)[:, None],
            weighted_overlap, corpus.idf_sums[rows],
            (pantry_mask * corpus.idf).sum(axis=1)[:, None], n_vocab=corpus.n_vocab,
            total_minutes=None if self.times is None else self.times[rows],
            max_total_minutes=budgets[:, None])
        return features, overlap, positives

    def batch(self, source_rows: np.ndarray, rng: np.random.Generator, *, split: str,
              used_signatures: set[bytes] | None = None) -> QueryBatch:
        source_rows = np.asarray(source_rows)
        if (source_rows.ndim != 1 or not len(source_rows) or source_rows.dtype.kind not in "iu"
                or (source_rows < 0).any() or (source_rows >= self.corpus.n_recipes).any()
                or len(source_rows) * (self.pool_size + 1) > 32_768):
            raise ValueError("source rows must be valid and the candidate pool batch must be bounded")
        pantries, signatures, limits, budgets, attempts = self._pantries(
            source_rows, rng, split, used_signatures)
        count = len(source_rows)
        hard_count = max(1, (self.pool_size * 7) // 8)
        anchors = np.empty((count, hard_count), dtype=np.int64)
        for index, pantry in enumerate(pantries):
            weights = self.corpus.idf[pantry]
            anchors[index] = rng.choice(pantry, size=hard_count, p=weights / weights.sum())
        positions = (self.corpus.posting_offsets[anchors]
                     + (rng.random(anchors.shape) * self.corpus.frequency[anchors]).astype(np.int64))
        hard = self.corpus.posting_rows[positions]
        random_rows = rng.integers(
            self.corpus.n_recipes, size=(count, self.pool_size - hard_count), dtype=np.int32)
        pool_rows = np.column_stack([hard, random_rows, source_rows])
        features, overlap, positives = self._pool_features(pantries, pool_rows, source_rows, budgets)
        feasible = ((self.corpus.sizes[pool_rows] - overlap <= limits[:, None]) & (overlap >= 1))
        if self.times is not None:
            feasible &= (np.isnan(budgets[:, None])
                         | (np.isfinite(self.times[pool_rows])
                            & (self.times[pool_rows] <= budgets[:, None])))
        if not feasible[:, -1].all() or not positives[:, -1].all():
            raise RuntimeError("the source recipe must be feasible and a positive action")
        baseline = score_features(deterministic_baseline_score, features)
        out_features = np.zeros((count, self.candidates, len(FEATURE_NAMES)), dtype=np.float32)
        valid = np.zeros((count, self.candidates), dtype=bool)
        out_positives = np.zeros_like(valid)
        candidate_rows = np.full(valid.shape, -1, dtype=np.int64)
        stats = Counter(queries=count, query_draw_attempts=attempts,
                        feasible_pool_slots=int(feasible.sum()),
                        time_constrained_queries=int(np.isfinite(budgets).sum()))
        jaccard_index = FEATURE_NAMES.index("jaccard")
        for index in range(count):
            order = np.lexsort((pool_rows[index], -baseline[index]))
            seen = {int(source_rows[index])}
            negative, equivalent = [], []
            for position in order:
                row = int(pool_rows[index, position])
                if not feasible[index, position] or row in seen:
                    continue
                seen.add(row)
                (equivalent if positives[index, position] else negative).append(int(position))
            # Retain hard non-equivalent actions even in duplicate-heavy catalogs.
            chosen = [self.pool_size] + negative[:max(1, self.candidates - 2)]
            if equivalent and len(chosen) < self.candidates:
                chosen.append(equivalent[0])
            chosen_set = set(chosen)
            for position in negative + equivalent:
                if len(chosen) == self.candidates:
                    break
                if position not in chosen_set:
                    chosen.append(position)
                    chosen_set.add(position)
            chosen = np.asarray(chosen)[rng.permutation(len(chosen))]
            size = len(chosen)
            out_features[index, :size] = features[index, chosen]
            valid[index, :size] = True
            out_positives[index, :size] = positives[index, chosen]
            candidate_rows[index, :size] = pool_rows[index, chosen]
            negatives = chosen[~positives[index, chosen]]
            stats["candidate_actions"] += size
            stats["positive_actions"] += int(positives[index, chosen].sum())
            stats["negative_actions"] += len(negatives)
            stats["negative_actions_sharing_two_or_more_ingredients"] += int(
                (overlap[index, negatives] >= 2).sum())
            stats["nontrivial_queries"] += int(len(negatives) > 0)
            stats["equivalent_positive_queries"] += int(positives[index, chosen].sum() > 1)
            stats["negative_jaccard_sum"] += float(features[index, negatives, jaccard_index].sum())
            stats["source_jaccard_sum"] += float(features[index, -1, jaccard_index])
        return QueryBatch(
            out_features, valid, out_positives, candidate_rows, source_rows.astype(np.int64),
            np.concatenate(pantries).astype(np.uint16),
            np.r_[np.int64(0), np.cumsum([len(pantry) for pantry in pantries])],
            signatures, limits, budgets, dict(stats))


def score_features(score_function, features: np.ndarray) -> np.ndarray:
    flat = features.reshape(-1, len(FEATURE_NAMES))
    output = np.concatenate([
        np.asarray(score_function(flat[start:start + MAX_CANDIDATES]), dtype=np.float64)
        for start in range(0, len(flat), MAX_CANDIDATES)
    ])
    if output.shape != (len(flat),) or not np.isfinite(output).all():
        raise ValueError("ranker must return one finite scalar per candidate")
    return output.reshape(features.shape[:-1])


def _validated_bandit_inputs(logits: torch.Tensor, valid: torch.Tensor,
                             positives: torch.Tensor) -> None:
    if (logits.ndim != 2 or not logits.is_floating_point()
            or logits.shape != valid.shape or logits.shape != positives.shape
            or valid.dtype != torch.bool or positives.dtype != torch.bool
            or not torch.isfinite(logits).all() or not valid.any(dim=1).all()
            or not (positives & valid).any(dim=1).all() or (positives & ~valid).any()):
        raise ValueError("each finite candidate group must have valid actions and at least one valid positive")


def listwise_loss(logits: torch.Tensor, valid: torch.Tensor,
                  positives: torch.Tensor) -> torch.Tensor:
    """Negative log probability of the union of all equivalent positive actions."""
    _validated_bandit_inputs(logits, valid, positives)
    all_logits = logits.masked_fill(~valid, -torch.inf)
    positive_logits = logits.masked_fill(~positives, -torch.inf)
    return (torch.logsumexp(all_logits, dim=1) - torch.logsumexp(positive_logits, dim=1)).mean()


def reinforce_loss(
        logits: torch.Tensor, reference_logits: torch.Tensor, valid: torch.Tensor,
        positives: torch.Tensor, *, generator: torch.Generator, actions_per_query: int = 4,
        temperature: float = 1.0, entropy_coefficient: float = 0.001,
        kl_coefficient: float = 0.02) -> tuple[torch.Tensor, dict]:
    """Sample actual actions; use detached, action-independent query baselines."""
    _validated_bandit_inputs(logits, valid, positives)
    _validated_bandit_inputs(reference_logits, valid, positives)
    if (type(actions_per_query) is not int or not 1 <= actions_per_query <= 64
            or not math.isfinite(temperature) or temperature <= 0
            or not math.isfinite(entropy_coefficient) or entropy_coefficient < 0
            or not math.isfinite(kl_coefficient) or kl_coefficient < 0):
        raise ValueError("invalid bandit temperature, regularization, or action count")
    log_probabilities = torch.log_softmax((logits / temperature).masked_fill(~valid, -torch.inf), dim=1)
    probabilities = log_probabilities.exp()
    reward_table = positives.to(logits.dtype)
    with torch.no_grad():
        actions = torch.multinomial(probabilities.detach(), actions_per_query,
                                    replacement=True, generator=generator)
        rewards = reward_table.gather(1, actions)
        baseline = ((probabilities.detach() * reward_table).sum(dim=1, keepdim=True)
                    / probabilities.detach().sum(dim=1, keepdim=True))
        advantages = rewards - baseline
        reference_log_probabilities = torch.log_softmax(
            (reference_logits.detach() / temperature).masked_fill(~valid, -torch.inf), dim=1)
    selected_log_probabilities = log_probabilities.gather(1, actions)
    policy_gradient = -(advantages.detach() * selected_log_probabilities).mean()
    safe_log_probabilities = log_probabilities.masked_fill(~valid, 0.0)
    entropy = -(probabilities * safe_log_probabilities).sum(dim=1).mean()
    kl = (probabilities * (
        safe_log_probabilities - reference_log_probabilities.masked_fill(~valid, 0.0))).sum(dim=1).mean()
    loss = policy_gradient - entropy_coefficient * entropy + kl_coefficient * kl
    return loss, {
        "sampled_actions": int(actions.numel()),
        "sampled_reward_sum": float(rewards.sum()),
        "expected_reward_sum": float(baseline.sum()),
        "policy_gradient_loss": float(policy_gradient.detach()),
        "entropy": float(entropy.detach()), "kl_to_supervised": float(kl.detach()),
        "nonzero_sampled_advantages": int(torch.count_nonzero(advantages)),
    }


class CoverageLedger:
    """Actual successful-optimizer-step source coverage, not a configured sampling cap."""

    def __init__(self, corpus: NumericCorpus, population: np.ndarray):
        population = np.asarray(population)
        if (population.ndim != 1 or not len(population) or population.dtype.kind not in "iu"
                or len(np.unique(population)) != len(population)
                or (population < 0).any() or (population >= corpus.n_recipes).any()):
            raise ValueError("training population must contain unique valid source rows")
        self.corpus, self.population = corpus, population
        self.allowed = np.zeros(corpus.n_recipes, dtype=bool)
        self.allowed[population] = True
        self.seen = np.zeros(corpus.n_recipes, dtype=np.uint8)
        self.examples = self.slots = self.steps = 0
        self.order_hash = hashlib.sha256()

    def record(self, rows: np.ndarray) -> None:
        rows = np.asarray(rows)
        if (rows.ndim != 1 or not len(rows) or rows.dtype.kind not in "iu"
                or (rows < 0).any() or (rows >= self.corpus.n_recipes).any()
                or len(np.unique(rows)) != len(rows) or not self.allowed[rows].all()
                or self.seen[rows].any()):
            raise RuntimeError("each population source row must be optimized exactly once per epoch")
        self.seen[rows] = 1
        self.examples += len(rows)
        self.slots += int(self.corpus.sizes[rows].sum())
        self.steps += 1
        self.order_hash.update(np.asarray(rows, dtype="<i8").tobytes())

    def finish(self, path: Path | None = None) -> dict:
        if not np.array_equal(self.seen.astype(bool), self.allowed):
            raise RuntimeError("epoch did not optimize every population source row exactly once")
        packed = np.packbits(self.seen, bitorder="little")
        if path is not None:
            np.savez_compressed(
                path, seen_counts=self.seen, eligible_bitmap=np.packbits(self.allowed, bitorder="little"),
                n_corpus_rows=np.asarray(self.corpus.n_recipes, dtype=np.int64))
        return {
            "n_queries": self.examples, "n_optimizer_steps": self.steps,
            "n_source_ingredient_slots": self.slots, "unique_source_rows": int(self.seen.sum()),
            "population_rows": len(self.population), "catalog_rows": self.corpus.n_recipes,
            "every_population_row_exactly_once": True,
            "every_catalog_row_exactly_once": bool(self.allowed.all()),
            "source_order_sha256": self.order_hash.hexdigest(),
            "seen_bitmap_sha256": hashlib.sha256(packed.tobytes()).hexdigest(),
            "coverage_artifact": None if path is None else str(path),
            "coverage_artifact_sha256": None if path is None else sha256_file(path),
        }


def train_stage(policy: RecipeRankingPolicy, sampler: QuerySampler, population: np.ndarray, *,
                stage: str, epochs: int, batch_size: int, seed: int, learning_rate: float,
                output: Path, reference: RecipeRankingPolicy | None = None,
                actions_per_query: int = 4, temperature: float = 1.0,
                entropy_coefficient: float = 0.001, kl_coefficient: float = 0.02,
                log_every: int = 200) -> dict:
    if stage not in ("supervised", "reinforce") or (stage == "reinforce" and reference is None):
        raise ValueError("REINFORCE requires a frozen supervised reference")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("training epochs, batch size, and learning rate must be positive")
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate, weight_decay=1e-4)
    action_rng = torch.Generator(device="cpu").manual_seed(seed + 91)
    coverage_path = output / "private_coverage"
    coverage_path.mkdir(exist_ok=True)
    started = time.perf_counter()
    histories = []
    policy.train()
    if reference is not None:
        reference.eval()
        reference.requires_grad_(False)
    for epoch in range(1, epochs + 1):
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 11 if stage == "supervised" else 29]))
        order = rng.permutation(population)
        ledger = CoverageLedger(sampler.corpus, population)
        sampling, totals = Counter(), Counter()
        query_hash = hashlib.sha256()
        partition_counts = np.zeros(10, dtype=np.int64)
        epoch_started = time.perf_counter()
        for start in range(0, len(order), batch_size):
            rows = order[start:start + batch_size]
            batch = sampler.batch(rows, rng, split="train")
            buckets = np.frombuffer(batch.signatures.tobytes(), dtype="<u8").reshape(-1, 2)[:, 0] % 10
            if (buckets >= 8).any():
                raise RuntimeError("a held-out pantry context cannot enter an optimizer step")
            features = torch.from_numpy(batch.features)
            valid, positives = torch.from_numpy(batch.valid), torch.from_numpy(batch.positives)
            optimizer.zero_grad(set_to_none=True)
            logits = policy(features)
            if stage == "supervised":
                loss = listwise_loss(logits, valid, positives)
                details = {}
            else:
                with torch.no_grad():
                    reference_logits = reference(features)
                loss, details = reinforce_loss(
                    logits, reference_logits, valid, positives, generator=action_rng,
                    actions_per_query=actions_per_query, temperature=temperature,
                    entropy_coefficient=entropy_coefficient, kl_coefficient=kl_coefficient)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{stage}: non-finite loss; no completed-epoch coverage claim is written")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in policy.parameters()):
                raise RuntimeError("optimizer produced non-finite ranking weights")
            ledger.record(rows)
            query_hash.update(batch.signatures.tobytes())
            partition_counts += np.bincount(buckets.astype(np.int64), minlength=10)
            sampling.update(batch.statistics)
            totals["loss_sum"] += float(loss.detach()) * len(rows)
            totals["gradient_norm_sum"] += float(gradient_norm) * len(rows)
            totals["steps_with_nonzero_gradient"] += int(gradient_norm > 0)
            for name, value in details.items():
                totals[name] += value * (len(rows) if name in (
                    "policy_gradient_loss", "entropy", "kl_to_supervised") else 1)
            if ledger.steps % log_every == 0 or start + len(rows) == len(order):
                progress = {
                    "status": "running", "stage": stage, "epoch": epoch,
                    "optimized_queries": ledger.examples, "population_rows": len(population),
                    "optimizer_steps": ledger.steps,
                    "queries_per_second": ledger.examples / (time.perf_counter() - epoch_started),
                }
                write_json(output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        coverage = ledger.finish(coverage_path / f"{stage}-epoch-{epoch:03d}.npz")
        seconds = time.perf_counter() - epoch_started
        histories.append({
            "epoch": epoch, "coverage": coverage, "seconds": seconds,
            "queries_per_second": ledger.examples / seconds,
            "mean_loss": totals["loss_sum"] / ledger.examples,
            "mean_gradient_norm_before_clip": totals["gradient_norm_sum"] / ledger.examples,
            "steps_with_nonzero_gradient": int(totals["steps_with_nonzero_gradient"]),
            "training_query_signatures_sha256": query_hash.hexdigest(),
            "query_partition_bucket_counts": partition_counts.tolist(),
            "sampling": dict(sampling),
            "bandit": None if stage == "supervised" else {
                "sampled_actions": int(totals["sampled_actions"]),
                "mean_sampled_reward": totals["sampled_reward_sum"] / totals["sampled_actions"],
                "mean_expected_reward": totals["expected_reward_sum"] / ledger.examples,
                "mean_policy_gradient_loss": totals["policy_gradient_loss"] / ledger.examples,
                "mean_entropy": totals["entropy"] / ledger.examples,
                "mean_kl_to_supervised": totals["kl_to_supervised"] / ledger.examples,
                "nonzero_sampled_advantages": int(totals["nonzero_sampled_advantages"]),
            },
        })
    policy.eval()
    return {
        "stage": stage, "epochs_completed": len(histories), "epochs": histories,
        "n_queries": sum(epoch["coverage"]["n_queries"] for epoch in histories),
        "n_optimizer_steps": sum(epoch["coverage"]["n_optimizer_steps"] for epoch in histories),
        "seconds": time.perf_counter() - started,
    }


def evaluation_cases(sampler: QuerySampler, rows: np.ndarray, *, split: str,
                     seed: int, batch_size: int) -> QueryBatch:
    rng = np.random.default_rng(seed)
    seen: set[bytes] = set()
    batches = [
        sampler.batch(rows[start:start + batch_size], rng, split=split, used_signatures=seen)
        for start in range(0, len(rows), batch_size)
    ]
    statistics = Counter()
    for batch in batches:
        statistics.update(batch.statistics)
    lengths = np.concatenate([np.diff(batch.pantry_offsets) for batch in batches])
    return QueryBatch(
        *(np.concatenate([getattr(batch, name) for batch in batches])
          for name in ("features", "valid", "positives", "candidate_rows", "source_rows")),
        np.concatenate([batch.pantry_flat for batch in batches]),
        np.r_[np.int64(0), np.cumsum(lengths)],
        *(np.concatenate([getattr(batch, name) for batch in batches])
          for name in ("signatures", "max_missing", "budgets")),
        dict(statistics))


def bootstrap_interval(values: np.ndarray, *, seed: int, repetitions: int = 2000) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or repetitions < 100:
        raise ValueError("bootstrap requires finite nonempty observations and at least 100 resamples")
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 64):
        count = min(64, repetitions - start)
        sample = rng.integers(len(values), size=(count, len(values)))
        means[start:start + count] = values[sample].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def recovery_metrics(scores: np.ndarray, cases: QueryBatch, *, seed: int,
                     bootstrap_repetitions: int) -> tuple[dict, np.ndarray]:
    if scores.shape != cases.valid.shape or not np.isfinite(scores).all():
        raise ValueError("evaluation scores must be finite and aligned with the fixed cases")
    maxima = np.where(cases.valid, scores, -np.inf).max(axis=1)
    top = cases.valid & (np.abs(scores - maxima[:, None]) <= TIE_ATOL)
    top_count = top.sum(axis=1)
    positive_top = (top & cases.positives).sum(axis=1)
    expected = positive_top / top_count
    deterministic = np.where(top, cases.candidate_rows, np.iinfo(np.int64).max).argmin(axis=1)
    informative = (cases.valid & ~cases.positives).any(axis=1)

    def subset(mask: np.ndarray) -> dict:
        n = int(mask.sum())
        if not n:
            return {"queries": 0, "expected_top1_recovery": None, "bootstrap_95_percent_interval": None}
        return {
            "queries": n, "expected_top1_recovery": float(expected[mask].mean()),
            "bootstrap_95_percent_interval": bootstrap_interval(
                expected[mask], seed=seed, repetitions=bootstrap_repetitions),
            "recipe_id_tiebroken_top1_recovery": float(
                cases.positives[np.arange(len(scores)), deterministic][mask].mean()),
            "optimistic_top1_recovery": float((positive_top[mask] > 0).mean()),
            "pessimistic_top1_recovery": float((positive_top[mask] == top_count[mask]).mean()),
            "top_score_tie_fraction": float((top_count[mask] > 1).mean()),
            "ambiguous_top_tie_fraction": float(
                ((positive_top[mask] > 0) & (positive_top[mask] < top_count[mask])).mean()),
            "mean_top_tie_size": float(top_count[mask].mean()),
        }

    return {"all_queries": subset(np.ones(len(scores), dtype=bool)),
            "nontrivial_queries": subset(informative)}, expected[informative]


def select_on_validation(values: dict[str, np.ndarray], *, seed: int,
                         bootstrap_repetitions: int) -> dict:
    """Prefer the simpler incumbent unless a paired validation interval clears zero."""
    if set(values) != {"heuristic", "supervised", "reinforce"}:
        raise ValueError("validation must compare heuristic, supervised, and reinforce")
    shapes = {array.shape for array in values.values()}
    if len(shapes) != 1 or not len(values["heuristic"]):
        raise ValueError("validation must include the same nontrivial queries for all policies")
    selected, comparisons = "heuristic", []
    for challenger in ("supervised", "reinforce"):
        differences = values[challenger] - values[selected]
        interval = bootstrap_interval(differences, seed=seed, repetitions=bootstrap_repetitions)
        replace = interval[0] > 0
        comparisons.append({
            "incumbent": selected, "challenger": challenger,
            "mean_paired_recovery_difference": float(differences.mean()),
            "paired_bootstrap_95_percent_interval": interval,
            "challenger_selected": replace,
        })
        if replace:
            selected = challenger
    return {
        "selected": selected, "split": "validation",
        "metric": "uniform-tie expected top1 source-set recovery on nontrivial sampled shortlists",
        "rule": ("Fixed order heuristic -> supervised -> reinforce; replace only if the lower "
                 "bound of the paired 95% query bootstrap interval is > 0. Test is not consulted."),
        "comparisons": comparisons,
        "interval_caveat": "Per-comparison query bootstrap, conditional on this synthetic generator; not human-preference confidence.",
    }


def _save_cases(path: Path, cases: QueryBatch) -> None:
    np.savez_compressed(path, **{
        name: getattr(cases, name) for name in (
            "features", "valid", "positives", "candidate_rows", "source_rows",
            "pantry_flat", "pantry_offsets", "signatures", "max_missing", "budgets")})


def experiment(corpus: NumericCorpus, sampler: QuerySampler, *, output: Path,
               population: np.ndarray, validation_rows: np.ndarray, test_rows: np.ndarray,
               args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    write_json(output / "feature_definitions.json", {
        "feature_names": list(FEATURE_NAMES), "definitions": FEATURE_DEFINITIONS,
        "idf_definition": IDF_DEFINITION,
    })
    write_json(output / "vocabulary.json", list(corpus.vocabulary))
    case_started = time.perf_counter()
    validation = evaluation_cases(
        sampler, validation_rows, split="validation", seed=args.seed + 101, batch_size=args.batch_size)
    test = evaluation_cases(
        sampler, test_rows, split="test", seed=args.seed + 202, batch_size=args.batch_size)
    if {value.tobytes() for value in validation.signatures} & {
            value.tobytes() for value in test.signatures}:
        raise RuntimeError("validation/test pantry signatures overlap")
    _save_cases(output / "private_validation_cases.npz", validation)
    _save_cases(output / "private_test_cases.npz", test)
    case_seconds = time.perf_counter() - case_started
    time_enabled = sampler.times is not None and bool(np.isfinite(sampler.times).any())
    supervised = RecipeRankingPolicy(
        args.hidden_dim, seed=args.seed, time_features_enabled=time_enabled)
    initialization_hash = array_digest(*[
        parameter.detach().cpu().numpy() for parameter in supervised.parameters()])
    supervised_training = train_stage(
        supervised, sampler, population, stage="supervised", epochs=args.supervised_epochs,
        batch_size=args.batch_size, seed=args.seed + 1000,
        learning_rate=args.supervised_lr, output=output, log_every=args.log_every)
    supervised.save(output / "supervised")
    reinforce = copy.deepcopy(supervised)
    reference = copy.deepcopy(supervised).requires_grad_(False)
    reinforce_training = train_stage(
        reinforce, sampler, population, stage="reinforce", epochs=args.reinforce_epochs,
        batch_size=args.batch_size, seed=args.seed + 2000, learning_rate=args.reinforce_lr,
        output=output, reference=reference, actions_per_query=args.actions_per_query,
        temperature=args.temperature, entropy_coefficient=args.entropy_coefficient,
        kl_coefficient=args.kl_coefficient, log_every=args.log_every)
    reinforce.save(output / "reinforce")
    checkpoint_files = {
        name: {filename: sha256_file(output / name / filename)
               for filename in ("recipe_ranker_config.json", "recipe_ranker.safetensors")}
        for name in ("supervised", "reinforce")
    }
    scorers = {"heuristic": deterministic_baseline_score,
               "supervised": supervised.score, "reinforce": reinforce.score}
    validation_metrics, validation_values = {}, {}
    for name, score in scorers.items():
        metrics, values = recovery_metrics(
            score_features(score, validation.features), validation, seed=args.seed + 301,
            bootstrap_repetitions=args.bootstrap_repetitions)
        validation_metrics[name], validation_values[name] = metrics, values
    selection = select_on_validation(
        validation_values, seed=args.seed + 302, bootstrap_repetitions=args.bootstrap_repetitions)
    deployment = {
        "schema_version": 1, "run_mode": args.mode,
        "deployed_ranker": "heuristic" if selection["selected"] == "heuristic" else "learned",
        "selected_candidate": selection["selected"],
        "policy_directory": None if selection["selected"] == "heuristic" else selection["selected"],
        "selection": selection, "corpus_sha256": corpus.corpus_sha256,
        "vocabulary_sha256": hashlib.sha256(json.dumps(
            list(corpus.vocabulary), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
        "production_or_human_preference_quality_established": False,
    }
    # Freeze selection on disk before computing any test policy score.
    write_json(output / "deployment.json", deployment)
    selection_hash = sha256_file(output / "deployment.json")
    test_metrics, test_values = {}, {}
    for name, score in scorers.items():
        metrics, values = recovery_metrics(
            score_features(score, test.features), test, seed=args.seed + 401,
            bootstrap_repetitions=args.bootstrap_repetitions)
        test_metrics[name], test_values[name] = metrics, values
    paired_test = {}
    for name in ("supervised", "reinforce"):
        differences = test_values[name] - test_values["heuristic"]
        paired_test[f"{name}_minus_heuristic"] = {
            "mean_difference": float(differences.mean()),
            "paired_bootstrap_95_percent_interval": bootstrap_interval(
                differences, seed=args.seed + 402, repetitions=args.bootstrap_repetitions),
        }
    rl_minus_supervised = validation_values["reinforce"] - validation_values["supervised"]
    return {
        "schema_version": 1, "status": "completed", "mode": args.mode,
        "checkpoint_files_sha256": checkpoint_files,
        "is_full_corpus_optimizer_coverage": bool(
            len(population) == corpus.n_recipes
            and all(epoch["coverage"]["every_catalog_row_exactly_once"]
                    for stage in (supervised_training, reinforce_training) for epoch in stage["epochs"])),
        "production_or_human_preference_quality_established": False,
        "population": {
            "catalog_records": corpus.n_recipes, "catalog_ingredient_slots": len(corpus.flat),
            "training_source_rows_per_epoch": len(population),
            "training_population_sha256": array_digest(np.sort(population)),
            "minimum_source_length": int(corpus.sizes[population].min()),
            "maximum_source_length": int(corpus.sizes[population].max()),
            "source_row_exclusions": "none" if len(population) == corpus.n_recipes else "explicit pilot sampling",
        },
        "model": {
            "architecture": "20 -> tanh(hidden_dim) -> 1", "hidden_dim": args.hidden_dim,
            "learned_parameter_count": sum(parameter.numel() for parameter in supervised.parameters()),
            "initialized_from_scratch": True, "initialization_sha256": initialization_hash,
            "pretrained_parameters_used": False, "time_features_enabled": time_enabled,
        },
        "reward": {
            "definition": "1 iff selected candidate's complete canonical ingredient set equals the source set; otherwise 0",
            "equivalent_sets_are_positive": True, "human_feedback": False,
            "not_measured": ["human taste or satisfaction", "food safety", "full-catalog retrieval recall",
                             "unseen-recipe generalization", "production readiness"],
        },
        "protocol": {
            "name": "held-out synthetic pantry queries over a known full catalog",
            "pantry_partition": "BLAKE2b-128(person=llmmm-query-v1, sorted unique uint16-LE IDs); first 8 bytes LE modulo 10; 0..7 train, 8 validation, 9 test",
            "all_rows_may_be_training_sources": True,
            "validation_test_pantry_contexts_never_used_for_training": True,
            "ingredient_set_disjoint_recipes": False,
            "catalog_idf_and_postings_use_all_rows": True,
            "source_insertion_in_candidate_set": True,
            "evaluation_scope": "sampled-candidate ranking with forced feasible source, NOT exhaustive retrieval",
            "negative_sampling": "7/8 pool draws from IDF-weighted pantry-ingredient postings; 1/8 uniform rows; keep hardest fixed-baseline feasible non-equivalent negatives plus available equivalent positives",
            "missing_constraint": f"uniform integer 0..{sampler.max_missing}, enforced before ranking",
            "pantry_corruption": f"remove uniformly 0..min(max_missing, source_length-1); add 0..{sampler.max_noise} distinct non-source IDs sampled proportional to sqrt(document frequency)",
            "time_constraint": "When source time is known, half of queries use a budget 1.0..1.5 times source time (minimum 1 minute); only known in-budget candidates survive",
            "tie_rule": f"absolute logit/score tolerance {TIE_ATOL}; primary reward expectation is uniform over all tied best candidates",
            "selection": selection, "deployment_sha256_frozen_before_test_scoring": selection_hash,
        },
        "training": {"supervised": supervised_training, "reinforce": reinforce_training},
        "evaluation": {
            "case_generation_seconds": case_seconds,
            "validation": {"case_sha256": validation.digest(), "queries": len(validation_rows),
                           "sampling": validation.statistics, "policies": validation_metrics},
            "test": {"case_sha256": test.digest(), "queries": len(test_rows),
                     "sampling": test.statistics, "policies": test_metrics,
                     "paired_nontrivial_comparisons_to_heuristic": paired_test},
            "rl_validation_mean_difference_from_supervised": float(rl_minus_supervised.mean()),
            "rl_validation_result": ("underperformed supervised" if rl_minus_supervised.mean() < 0 else
                                     "tied supervised" if rl_minus_supervised.mean() == 0 else
                                     "higher point estimate than supervised; see validation intervals"),
            "test_did_not_select_deployment": True,
            "bootstrap_repetitions": args.bootstrap_repetitions,
        },
        "seconds_excluding_corpus_and_index_setup": time.perf_counter() - started,
        "private_artifacts_not_for_publication": [
            "private_validation_cases.npz", "private_test_cases.npz", "private_coverage/"],
        "public_artifact_scope": "Learned safetensors, strict configuration, vocabulary/feature definitions, and aggregate reports only; no recipe or query datasets.",
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--corpus", type=Path, default=_ROOT / "data/recipes/recipe_ids.npz")
    result.add_argument("--generation", type=Path, default=_ROOT / "data/GENERATION.json")
    result.add_argument("--catalog", type=Path, help="Optional completed, matching private SQLite catalog for numeric time metadata")
    result.add_argument("--output", type=Path, required=True, help="New private run directory; existing contents are never overwritten")
    result.add_argument("--index-cache", type=Path, default=_ROOT / "training/recipe_ranker_index")
    result.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    result.add_argument("--max-train-rows", type=int, default=2048,
                        help="Pilot source-row cap. Full mode explicitly requires 0 (all canonical rows).")
    result.add_argument("--validation-queries", type=int, default=256)
    result.add_argument("--test-queries", type=int, default=256)
    result.add_argument("--candidates", type=int, default=16)
    result.add_argument("--pool-size", type=int, default=128)
    result.add_argument("--max-missing", type=int, default=2)
    result.add_argument("--max-noise", type=int, default=4)
    result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--threads", type=int, default=4)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--hidden-dim", type=int, default=32)
    result.add_argument("--supervised-epochs", type=int, default=1)
    result.add_argument("--reinforce-epochs", type=int, default=1)
    result.add_argument("--supervised-lr", type=float, default=0.003)
    result.add_argument("--reinforce-lr", type=float, default=0.0003)
    result.add_argument("--actions-per-query", type=int, default=4)
    result.add_argument("--temperature", type=float, default=1.0)
    result.add_argument("--entropy-coefficient", type=float, default=0.001)
    result.add_argument("--kl-coefficient", type=float, default=0.02)
    result.add_argument("--bootstrap-repetitions", type=int, default=2000)
    result.add_argument("--log-every", type=int, default=200)
    return result


def validate_args(args: argparse.Namespace) -> None:
    if args.mode == "full" and args.max_train_rows != 0:
        raise ValueError("full mode requires --max-train-rows 0; a sampling cap is not full-corpus training")
    if args.mode == "pilot" and not 1 <= args.max_train_rows < CANONICAL_RECIPES:
        raise ValueError("pilot mode requires an explicit positive cap smaller than the full corpus")
    if (not 1 <= args.threads <= 4 or not 1 <= args.batch_size <= 1024
            or args.batch_size * (args.pool_size + 1) > 32_768):
        raise ValueError("use at most 4 CPU threads and at most 32768 pool slots per batch")
    if (not 2 <= args.candidates <= 64 or not args.candidates <= args.pool_size <= 1024
            or not 0 <= args.max_missing <= 8 or not 1 <= args.max_noise <= 16):
        raise ValueError("invalid bounded candidate-pool or pantry-corruption configuration")
    if (not 1 <= args.validation_queries <= 100_000 or not 1 <= args.test_queries <= 100_000
            or args.validation_queries + args.test_queries > CANONICAL_RECIPES
            or not 1 <= args.supervised_epochs <= 100 or not 1 <= args.reinforce_epochs <= 100
            or not 1 <= args.hidden_dim <= 256 or not 0 <= args.seed < 2**31
            or not 1 <= args.actions_per_query <= 64 or not 100 <= args.bootstrap_repetitions <= 10_000
            or args.log_every < 1):
        raise ValueError("invalid bounded training/evaluation count, architecture, or seed")
    for name in ("supervised_lr", "reinforce_lr", "temperature"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("entropy_coefficient", "kl_coefficient"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")


def main(argv: list[str] | None = None) -> int:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as error:
        argument_parser.error(str(error))
    if args.output.exists() and any(args.output.iterdir()):
        argument_parser.error(f"{args.output}: output must be new or empty; existing artifacts are immutable")
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    torch.set_num_threads(args.threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    total_started = time.perf_counter()
    configuration = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "code_sha256": {
            "train_recipe_ranker.py": sha256_file(Path(__file__).resolve()),
            "recipe_ranker.py": sha256_file(Path(ranker_module.__file__).resolve()),
        },
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__, "safetensors": safetensors.__version__,
            "platform": platform.platform(), "device": "cpu",
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "foreign_pretrained_weights": False,
        },
    }
    write_json(args.output / "run_config.json", configuration)
    corpus_started = time.perf_counter()
    corpus = NumericCorpus.load_canonical(args.corpus, args.generation)
    corpus_seconds = time.perf_counter() - corpus_started
    index_metadata = corpus.build_postings(args.index_cache)
    times = None
    time_metadata = {"enabled": False, "reason": "No catalog supplied; time feature inputs are disabled in learned policies."}
    if args.catalog:
        times, time_metadata = load_catalog_times(args.catalog, corpus)
    sampler = QuerySampler(corpus, candidates=args.candidates, pool_size=args.pool_size,
                           max_missing=args.max_missing, max_noise=args.max_noise, times=times)
    rng = np.random.default_rng(args.seed)
    population = (np.arange(corpus.n_recipes, dtype=np.int32) if args.mode == "full" else
                  np.sort(rng.choice(corpus.n_recipes, args.max_train_rows, replace=False)).astype(np.int32))
    evaluation_rows = np.random.default_rng(args.seed + 23).choice(
        corpus.n_recipes, args.validation_queries + args.test_queries, replace=False).astype(np.int32)
    report = experiment(
        corpus, sampler, output=args.output, population=population,
        validation_rows=evaluation_rows[:args.validation_queries],
        test_rows=evaluation_rows[args.validation_queries:], args=args)
    completed_utc = datetime.now(timezone.utc).isoformat()
    configuration.update(status="completed", completed_utc=completed_utc)
    report.update({
        "corpus": {"sha256": corpus.corpus_sha256, "generation_sha256": corpus.generation_sha256,
                   "n_recipes": corpus.n_recipes, "n_ingredient_slots": len(corpus.flat),
                   "n_vocab": corpus.n_vocab, "minimum_length": int(corpus.sizes.min()),
                   "maximum_length": int(corpus.sizes.max()), "numeric_load_seconds": corpus_seconds},
        "numeric_postings": index_metadata, "time_metadata": time_metadata,
        "run_configuration": configuration,
        "total_wall_seconds": time.perf_counter() - total_started,
        "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * (1 if sys.platform == "darwin" else 1024),
        "completed_utc": completed_utc,
    })
    write_json(args.output / "report.json", report)
    write_json(args.output / "run_config.json", configuration)
    write_json(args.output / "progress.json", {"status": "completed", "report_sha256": sha256_file(args.output / "report.json")})
    print(json.dumps({
        "status": "completed", "mode": args.mode, "report": str(args.output / "report.json"),
        "selected": report["protocol"]["selection"]["selected"],
        "full_corpus_optimizer_coverage": report["is_full_corpus_optimizer_coverage"],
        "training_queries": sum(stage["n_queries"] for stage in report["training"].values()),
        "optimizer_steps": sum(stage["n_optimizer_steps"] for stage in report["training"].values()),
        "total_wall_seconds": report["total_wall_seconds"],
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
