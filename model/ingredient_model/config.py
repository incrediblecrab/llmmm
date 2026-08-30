"""Filesystem layout and run-time configuration.

Every path is resolvable from an environment variable so the identical code runs
unchanged on a laptop and inside an Azure ML job, where inputs and outputs are
mounted at paths chosen by the platform rather than by us.

    IM_DATA     where datasets live      (default <repo>/data)
    IM_RESULTS  where runs are written   (default <repo>/results)
    IM_CORPUS   the raw recipe corpus    (optional; only bulk rebuilds need it)
    IM_PRIOR_STUDY  the prior study tree (default <repo>/../prior-study)

Nothing else in the package may construct a path by hand. A module that hardcodes
``../data`` works locally and silently reads an empty directory on the cluster,
which is the failure mode this file exists to prevent.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _p(env: str, default: Path) -> Path:
    return Path(os.environ.get(env, default)).expanduser()


@dataclass(frozen=True)
class Paths:
    """Resolved locations for one process.

    Built once at import as ``PATHS``. Tests and tools that need a different
    root construct their own instance rather than mutating the global.
    """

    data: Path
    results: Path
    corpus: Path | None
    prior_study: Path

    @property
    def prior_tools(self) -> Path:
        """The prior study's corpus readers and normaliser.

        ``corpus.py`` and ``normalize.py`` built the corpus every model here is
        trained on, so re-deriving or auditing it means importing them rather
        than reimplementing them — a reimplementation that disagrees would be
        indistinguishable from a corpus bug.
        """
        return self.prior_study / "tools"

    @property
    def graphs(self) -> Path:
        """Derived graph artefacts: the ingredient-ingredient NPMI graph, its
        train/held-out splits, and the ingredient-compound chemistry graph."""
        return self.data / "graphs"

    @property
    def recipes(self) -> Path:
        """Tokenised corpus: every recipe as a sorted set of vocabulary ids."""
        return self.data / "recipes"

    @property
    def catalog(self) -> Path:
        """Evaluation labels that no model may train on."""
        return self.data / "catalog"

    @property
    def runs(self) -> Path:
        return self.results / "runs"

    @property
    def reports(self) -> Path:
        return self.results / "reports"

    def run_dir(self, run_id: str) -> Path:
        return self.runs / run_id

    def ensure(self) -> "Paths":
        for d in (self.data, self.graphs, self.recipes, self.catalog,
                  self.results, self.runs, self.reports):
            d.mkdir(parents=True, exist_ok=True)
        return self


def load_paths() -> Paths:
    corpus = os.environ.get("IM_CORPUS")
    return Paths(
        data=_p("IM_DATA", REPO / "data"),
        results=_p("IM_RESULTS", REPO / "results"),
        corpus=Path(corpus).expanduser() if corpus else None,
        prior_study=_p("IM_PRIOR_STUDY", REPO.parent / "prior-study"),
    )


PATHS = load_paths()

# Fixed across every experiment. Changing it invalidates comparisons against
# results already recorded, so it is a constant rather than a flag.
SEED = 20260805
