"""Bounded recipe retrieval with hard constraints and a separately learned ranker."""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Protocol, Sequence
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from ._hashing import file_sha256


class RankingPolicy(Protocol):
    def score(self, features: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class RecipeQuery:
    available_ingredients: Sequence[str]
    must_use: Sequence[str] = ()
    exclude: Sequence[str] = ()
    max_total_minutes: float | None = None
    max_missing: int | None = 2
    min_servings: float | None = None
    language: str | None = None
    top_k: int = 10


@dataclass(frozen=True)
class RecipeMatch:
    recipe_id: int
    title: str
    source: str
    language: str
    source_url: str | None
    ingredients: tuple[str, ...]
    matched_ingredients: tuple[str, ...]
    missing_ingredients: tuple[str, ...]
    total_minutes: float | None
    servings: float | None
    time_status: str
    raw_ingredients: tuple[str, ...]
    steps: tuple[str, ...]
    ingredient_quantities: tuple[str | None, ...]
    score: float
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SearchResult:
    recipes: tuple[RecipeMatch, ...]
    ranking: str
    candidates_scanned: int
    feasible_candidates: int
    retrieval_truncated: bool
    elapsed_ms: float
    warnings: tuple[str, ...] = ()
    candidate_recipe_ids: tuple[int, ...] | None = None
    metadata_index_used: bool = False
    ingredient_index_used: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class _Candidate:
    row: sqlite3.Row
    ids: frozenset[int]
    total_minutes: float | None
    servings: float | None


def _number(value: object, name: str, *, minimum: float = 0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return float(value)


def _optional_number(value: object, name: str) -> float | None:
    return None if value is None else _number(value, name)


def _source_url(value: str) -> str | None:
    value = value.strip()
    if not value:
        return None
    try:
        parts = urlsplit(value if "://" in value else f"https://{value}")
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username:
        return None
    if "." not in parts.hostname or any(character.isspace() for character in parts.hostname):
        return None
    return urlunsplit(parts)


class RecipeFinder:
    """Search a local authorized catalog; model weights do not contain recipes.

    Each request opens its own read-only SQLite connection. Constraint checks
    precede ranking, so a learned score cannot override a time or ingredient limit.
    Scores order candidates within a policy; they are not calibrated probabilities.
    """

    def __init__(self, catalog_path: str | Path, *,
                 policy: RankingPolicy | None = None, ranking: str = "learned",
                 max_candidates: int = 2000, scan_limit: int = 50_000,
                 timeout_seconds: float = 5.0, allow_partial: bool = False,
                 metadata_index_path: str | Path | None = None,
                 require_metadata_index: bool = False,
                 corpus_path: str | Path | None = None, require_corpus_index: bool = False):
        self.catalog_path = Path(catalog_path).resolve()
        if not self.catalog_path.is_file():
            raise FileNotFoundError(
                f"{self.catalog_path}: build or restore the private recipe catalog first")
        if ranking not in ("learned", "heuristic"):
            raise ValueError("ranking must be 'learned' or 'heuristic'")
        if ranking == "learned" and policy is None:
            raise ValueError("learned ranking requires a trained policy; use ranking='heuristic' explicitly")
        for name, value, upper in (
            ("max_candidates", max_candidates, 10_000), ("scan_limit", scan_limit, 200_000),
        ):
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"{name} must be an integer between 1 and {upper}")
        if scan_limit < max_candidates:
            raise ValueError("scan_limit must be at least max_candidates")
        self.timeout_seconds = _number(timeout_seconds, "timeout_seconds", minimum=0.01)
        self.max_candidates, self.scan_limit = max_candidates, scan_limit
        self.policy, self.ranking = policy, ranking
        with closing(self._connect()) as connection:
            self.metadata = {
                row["key"]: json.loads(row["value"])
                for row in connection.execute("SELECT key, value FROM metadata")
            }
        if self.metadata.get("schema_version") != 1:
            raise ValueError("unsupported recipe catalog schema")
        if self.metadata.get("partial") is not False and not allow_partial:
            raise ValueError("a partial catalog requires explicit allow_partial=True")
        vocabulary = self.metadata.get("vocabulary")
        if (not isinstance(vocabulary, list) or not vocabulary
                or not all(isinstance(item, str) and item for item in vocabulary)
                or len(set(vocabulary)) != len(vocabulary)):
            raise ValueError("catalog vocabulary must contain unique nonempty names")
        self.vocabulary = tuple(vocabulary)
        self._index = {name: index for index, name in enumerate(vocabulary)}
        self.n_recipes = self.metadata.get("n_recipes")
        if type(self.n_recipes) is not int or self.n_recipes < 1:
            raise ValueError("catalog must record its positive recipe count")
        self.ingredient_frequency = np.asarray(
            self.metadata.get("ingredient_frequency"), dtype=np.float64)
        if (self.ingredient_frequency.shape != (len(vocabulary),)
                or not np.isfinite(self.ingredient_frequency).all()
                or (self.ingredient_frequency < 0).any()
                or (self.ingredient_frequency > self.n_recipes).any()):
            raise ValueError("catalog ingredient frequencies are invalid")
        stat = self.catalog_path.stat()
        self._catalog_stamp = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if type(require_metadata_index) is not bool:
            raise ValueError("require_metadata_index must be a boolean")
        from .data.recipe_search_metadata import RecipeSearchMetadata, default_search_metadata_path
        from .recipe_ranker import candidate_features, heuristic_scores

        selected_index = (Path(metadata_index_path) if metadata_index_path is not None
                          else default_search_metadata_path(self.catalog_path))
        self._metadata_index = None
        if selected_index.exists():
            self._metadata_index = RecipeSearchMetadata.load(self.catalog_path, selected_index)
        elif metadata_index_path is not None or require_metadata_index:
            raise FileNotFoundError(
                f"{selected_index}: build the private search metadata index first")
        if type(require_corpus_index) is not bool:
            raise ValueError("require_corpus_index must be a boolean")
        from .recipe_ingredients import CanonicalIngredientIndex

        selected_corpus = (Path(corpus_path) if corpus_path is not None
                           else self.catalog_path.parent / "recipe_ids.npz")
        self._ingredient_index = None
        if selected_corpus.exists():
            self._ingredient_index = CanonicalIngredientIndex.load(
                selected_corpus, corpus_sha256=self.metadata.get("corpus_sha256"),
                n_recipes=self.n_recipes, n_slots=self.metadata.get("n_slots"),
                n_vocab=len(self.vocabulary))
        elif corpus_path is not None or require_corpus_index:
            raise FileNotFoundError(
                f"{selected_corpus}: restore the canonical ingredient corpus first")
        self._candidate_features, self._heuristic_scores = candidate_features, heuristic_scores

    @classmethod
    def from_directory(cls, policy_path: str | Path, *, catalog_path: str | Path,
                       **search_options) -> "RecipeFinder":
        from .recipe_ranker import RecipeRankingPolicy

        return cls(catalog_path, policy=RecipeRankingPolicy.load(Path(policy_path)),
                   **search_options)

    @classmethod
    def from_pretrained(
            cls, model_id: str, *, catalog_path: str | Path, revision: str | None = None,
            cache_dir: str | Path | None = None, local_files_only: bool = False,
            token: str | bool | None = None, **search_options) -> "RecipeFinder":
        from huggingface_hub import hf_hub_download

        local = Path(model_id)

        def asset(name: str) -> Path:
            if local.is_dir():
                return local / name
            return Path(hf_hub_download(
                repo_id=model_id, filename=name, revision=revision,
                cache_dir=cache_dir, local_files_only=local_files_only, token=token))

        configuration = json.loads(asset("recipe_search_config.json").read_text())
        if configuration.get("schema_version") != 1:
            raise ValueError("unsupported recipe-search model configuration")
        defaults = configuration.get("search_defaults", {})
        if not isinstance(defaults, dict) or set(defaults) - {
                "max_candidates", "scan_limit", "timeout_seconds"}:
            raise ValueError("unsupported recipe-search serving defaults")
        search_options = {**defaults, **search_options}
        required_index = configuration.get("requires_metadata_index", False)
        if type(required_index) is not bool:
            raise ValueError("requires_metadata_index must be a boolean")
        if required_index:
            search_options["require_metadata_index"] = True
        required_corpus = configuration.get("requires_corpus_index", False)
        if type(required_corpus) is not bool:
            raise ValueError("requires_corpus_index must be a boolean")
        if required_corpus:
            search_options["require_corpus_index"] = True
        files = configuration.get("policy_files")
        if (not isinstance(files, dict) or not 1 <= len(files) <= 8
                or not all(isinstance(name, str) and
                           re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name)
                           and Path(name).suffix in (".json", ".safetensors")
                           for name in files)):
            raise ValueError("recipe policy must declare a bounded safe file inventory")
        policy_directory = configuration.get("policy_directory", "recipe_policy")
        if policy_directory not in (
                "recipe_policy", "recipe_policies/supervised", "recipe_policies/reinforce"):
            raise ValueError("unsupported recipe policy directory")
        policy_root = None
        for name, expected in files.items():
            path = asset(f"{policy_directory}/{name}")
            actual = file_sha256(path)
            if actual != expected:
                raise ValueError(f"{name}: recipe policy checksum differs from its configuration")
            policy_root = path.parent
        if configuration.get("deployed_ranker") == "heuristic":
            finder = cls(catalog_path, ranking="heuristic", **search_options)
        elif configuration.get("deployed_ranker") == "learned":
            assert policy_root is not None
            finder = cls.from_directory(
                policy_root, catalog_path=catalog_path, **search_options)
        else:
            raise ValueError("recipe-search configuration must declare the deployed ranker")
        vocabulary_hash = hashlib.sha256(json.dumps(
            list(finder.vocabulary), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        if (finder.metadata.get("corpus_sha256") != configuration.get("corpus_sha256")
                or vocabulary_hash != configuration.get("vocabulary_sha256")):
            raise ValueError("recipe catalog and ranking release use different corpus/vocabulary identities")
        return finder

    def _connect(self) -> sqlite3.Connection:
        ingredient_index = getattr(self, "_ingredient_index", None)
        if ingredient_index is not None:
            ingredient_index.check_unchanged()
        expected = getattr(self, "_catalog_stamp", None)
        if expected is not None:
            stat = self.catalog_path.stat()
            if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != expected:
                raise ValueError("recipe catalog changed after initialization; reopen and verify its index")
        connection = sqlite3.connect(
            self.catalog_path.as_uri() + "?mode=ro", uri=True, timeout=2.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        return connection

    def _ids(self, values: Sequence[str], name: str) -> frozenset[int]:
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise ValueError(f"{name} must be a sequence of ingredient names")
        if len(values) > 100 or not all(isinstance(value, str) for value in values):
            raise ValueError(f"{name} must contain at most 100 ingredient-name strings")
        normalized = [value.strip().lower().replace(" ", "_") for value in values]
        unknown = sorted({value for value in normalized if value not in self._index})
        if unknown:
            raise ValueError(f"{name} contains unknown canonical ingredients: {', '.join(unknown)}")
        return frozenset(self._index[value] for value in normalized)

    def _recipe_ids(self, row: sqlite3.Row) -> frozenset[int]:
        values = np.frombuffer(row["ingredient_ids"], dtype="<u2")
        if (not len(values) or (values >= len(self.vocabulary)).any()
                or np.any(values[1:] <= values[:-1])):
            raise ValueError(f"recipe {row['id']}: invalid sorted ingredient IDs")
        ids = frozenset(int(value) for value in values)
        if self._ingredient_index is not None and ids != self._ingredient_index.recipe_ids(row["id"]):
            raise ValueError(f"recipe {row['id']}: catalog ingredient IDs differ from the canonical corpus")
        return ids

    def _retrieval_rows(self, connection: sqlite3.Connection, expression: str,
                        conditions: list[str], parameters: list[object], *,
                        max_total_minutes: float | None, min_servings: float | None,
                        language: str | None, max_ingredients: int | None,
                        available: frozenset[int], max_missing: int | None,
                        ) -> Iterator[sqlite3.Row | None]:
        cursor = connection.execute(
            "SELECT rowid FROM recipe_fts WHERE recipe_fts MATCH ? "
            "ORDER BY rank LIMIT ?", (expression, self.scan_limit + 1))
        while ids := [row[0] for row in cursor.fetchmany(512)]:
            eligible = (ids if self._metadata_index is None else self._metadata_index.filter_ids(
                ids, max_total_minutes=max_total_minutes, min_servings=min_servings,
                language=language, max_ingredients=max_ingredients))
            if self._ingredient_index is not None and max_missing is not None and eligible:
                eligible = self._ingredient_index.filter_ids(eligible, available, max_missing)
            by_id = {}
            if eligible:
                placeholders = ",".join("?" for _ in eligible)
                rows = connection.execute(
                    f"SELECT r.* FROM recipes r WHERE r.id IN ({placeholders}) AND "
                    + " AND ".join(conditions), [*eligible, *parameters])
                by_id = {row["id"]: row for row in rows}
            for recipe_id in ids:
                yield by_id.get(recipe_id)

    def search(self, query: RecipeQuery, *, include_candidate_ids: bool = False) -> SearchResult:
        if type(include_candidate_ids) is not bool:
            raise ValueError("include_candidate_ids must be a boolean")
        start = time.monotonic()
        available = self._ids(query.available_ingredients, "available_ingredients")
        required = self._ids(query.must_use, "must_use")
        excluded = self._ids(query.exclude, "exclude")
        if not available:
            raise ValueError("at least one available ingredient is required")
        if required & excluded:
            raise ValueError("must_use and exclude cannot contain the same ingredient")
        if not (available - excluded or required):
            raise ValueError("no searchable ingredients remain after exclusions")
        if type(query.top_k) is not int or not 1 <= query.top_k <= min(100, self.max_candidates):
            raise ValueError("top_k must be between 1 and min(100, max_candidates)")
        if query.max_missing is not None and (
                type(query.max_missing) is not int or query.max_missing < 0):
            raise ValueError("max_missing must be a nonnegative integer or None")
        budget = (None if query.max_total_minutes is None else
                  _number(query.max_total_minutes, "max_total_minutes"))
        servings = (None if query.min_servings is None else
                    _number(query.min_servings, "min_servings", minimum=1))
        if query.language is not None and (
                not isinstance(query.language, str) or not query.language.strip()
                or len(query.language) > 32):
            raise ValueError("language must be a nonempty code of at most 32 characters")
        terms = " OR ".join(f"i{value}" for value in sorted((available - excluded) | required))
        expression = f"({terms})"
        if required:
            expression += " AND (" + " AND ".join(f"i{value}" for value in sorted(required)) + ")"
        if excluded:
            expression += " NOT (" + " OR ".join(f"i{value}" for value in sorted(excluded)) + ")"
        conditions = ["length(trim(r.title)) > 0",
                      "length(trim(r.steps, ' '||char(9)||char(10)||char(13)||char(31))) > 0",
                      "coalesce(r.text_status, '') != 'unparsed_steps'"]
        parameters: list[object] = []
        if budget is not None:
            conditions.append("r.total_minutes IS NOT NULL AND r.total_minutes <= ?")
            parameters.append(budget)
        if servings is not None:
            conditions.append("r.servings IS NOT NULL AND r.servings >= ?")
            parameters.append(servings)
        if query.language is not None:
            conditions.append("r.language = ?")
            parameters.append(query.language)
        candidates: list[_Candidate] = []
        scanned, truncated = 0, False
        connection = self._connect()
        deadline = start + self.timeout_seconds
        connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
        try:
            for row in self._retrieval_rows(
                    connection, expression, conditions, parameters,
                    max_total_minutes=budget, min_servings=servings, language=query.language,
                    max_ingredients=(None if query.max_missing is None else
                                     len(available) + query.max_missing),
                    available=available, max_missing=query.max_missing):
                if time.monotonic() > deadline:
                    raise TimeoutError("recipe search exceeded its time budget; narrow the query")
                if scanned >= self.scan_limit:
                    truncated = True
                    break
                scanned += 1
                if row is None:
                    continue
                if not row["title"].strip() or not any(
                        step.strip() for step in row["steps"].split("\x1f")):
                    continue
                ids = self._recipe_ids(row)
                total = _optional_number(row["total_minutes"], f"recipe {row['id']} total_minutes")
                count = _optional_number(row["servings"], f"recipe {row['id']} servings")
                if total is not None and (total <= 0 or row["time_status"] != "source_total"):
                    raise ValueError(f"recipe {row['id']}: invalid source-total provenance")
                if count is not None and count <= 0:
                    raise ValueError(f"recipe {row['id']}: reported servings must be positive")
                if count is not None and row["servings_status"] != "source_servings":
                    raise ValueError(f"recipe {row['id']}: invalid source-serving provenance")
                if not required <= ids or excluded & ids:
                    raise ValueError(f"recipe {row['id']}: catalog search violated ingredient constraints")
                if budget is not None and (total is None or total > budget):
                    raise ValueError(f"recipe {row['id']}: catalog search violated the time constraint")
                if servings is not None and (count is None or count < servings):
                    raise ValueError(f"recipe {row['id']}: catalog search violated the serving constraint")
                if query.max_missing is not None and len(ids - available) > query.max_missing:
                    continue
                if len(candidates) >= self.max_candidates:
                    truncated = True
                    break
                candidates.append(_Candidate(row, ids, total, count))
                if time.monotonic() > deadline:
                    raise TimeoutError("recipe search exceeded its time budget; narrow the query")
        except sqlite3.OperationalError as error:
            if getattr(error, "sqlite_errorcode", None) == sqlite3.SQLITE_INTERRUPT:
                raise TimeoutError("recipe search exceeded its time budget; narrow the query") from error
            raise
        finally:
            connection.close()
        warnings = [
            "canonical_ingredient_matching: review source ingredients and quantities before cooking"
        ]
        if excluded:
            warnings.append("canonical_exclusions_only: not an allergen-safety assessment")
        if truncated:
            warnings.append("retrieval_budget_reached: results are ranked from a bounded shortlist")
        if self.metadata.get("partial"):
            warnings.append("partial_catalog: this search does not cover the full corpus")
        if not candidates:
            return SearchResult((), self.ranking, scanned, 0, truncated,
                                (time.monotonic() - start) * 1000, tuple(warnings),
                                () if include_candidate_ids else None,
                                self._metadata_index is not None, self._ingredient_index is not None)

        features = self._candidate_features(
            sorted(available), [sorted(candidate.ids) for candidate in candidates],
            ingredient_frequency=self.ingredient_frequency, n_recipes=self.n_recipes,
            total_minutes=[candidate.total_minutes for candidate in candidates],
            max_total_minutes=budget)
        if self.ranking == "learned":
            assert self.policy is not None
            scores = np.asarray(self.policy.score(features), dtype=np.float64)
        else:
            scores = np.asarray(self._heuristic_scores(features), dtype=np.float64)
        if scores.shape != (len(candidates),) or not np.isfinite(scores).all():
            raise ValueError("ranking policy returned invalid scores")
        if time.monotonic() > deadline:
            raise TimeoutError("recipe ranking exceeded its time budget; narrow the query")
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i], candidates[i].row["id"]))
        matches = []
        for index in order[:query.top_k]:
            candidate = candidates[index]
            row, ids = candidate.row, candidate.ids
            quantities = json.loads(row["ingredient_quantities"])
            if not isinstance(quantities, list) or not all(
                    value is None or isinstance(value, str) for value in quantities):
                raise ValueError(f"recipe {row['id']}: invalid separate quantity values")
            recipe_warnings = []
            if row["quantity_status"] in ("values_without_units", "count_mismatch"):
                recipe_warnings.append(row["quantity_status"])
            if candidate.total_minutes is None:
                recipe_warnings.append("total_time_unknown")
            url = _source_url(row["url"])
            if url is None:
                recipe_warnings.append("source_url_unavailable")
            if row["text_status"]:
                recipe_warnings.append(row["text_status"])
            matches.append(RecipeMatch(
                recipe_id=row["id"], title=row["title"], source=row["source"],
                language=row["language"], source_url=url,
                ingredients=tuple(self.vocabulary[value] for value in sorted(ids)),
                matched_ingredients=tuple(self.vocabulary[value] for value in sorted(ids & available)),
                missing_ingredients=tuple(self.vocabulary[value] for value in sorted(ids - available)),
                total_minutes=candidate.total_minutes, servings=candidate.servings,
                time_status=row["time_status"],
                raw_ingredients=tuple(row["raw_ingredients"].split("\x1f")),
                steps=tuple(value.strip() for value in row["steps"].split("\x1f") if value.strip()),
                ingredient_quantities=tuple(quantities), score=float(scores[index]),
                warnings=tuple(recipe_warnings)))
        if time.monotonic() > deadline:
            raise TimeoutError("recipe response exceeded its time budget; narrow the query")
        return SearchResult(tuple(matches), self.ranking, scanned, len(candidates), truncated,
                            (time.monotonic() - start) * 1000, tuple(warnings),
                            tuple(candidate.row["id"] for candidate in candidates)
                            if include_candidate_ids else None, self._metadata_index is not None,
                            self._ingredient_index is not None)
