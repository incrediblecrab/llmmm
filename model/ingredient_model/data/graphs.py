"""Graph datasets: ingredient-ingredient co-occurrence and ingredient-compound
chemistry.

The ingredient-ingredient graph is weighted by normalised pointwise mutual
information rather than raw counts. Raw co-occurrence is dominated by
near-universal ingredients — salt appears with everything, so counts rank
"salt + onion" above every genuinely informative pair. NPMI divides that
popularity out and lands in [-1, 1].

Three splits ship, and choosing the wrong one silently invalidates results:

``full``     every edge. Use for production artefacts only.
``train``    full minus a 10% held-out sample. Use for anything scored on M4.
``heldout``  the removed 10%. Labels — never a training input.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import PATHS

GRAPH_FULL = "ii_graph.npz"
GRAPH_TRAIN = "ii_graph_train.npz"
GRAPH_HELDOUT = "ii_graph_heldout.npz"
CHEM_GRAPH = "flavor_graph.npz"


def _readonly(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    a.flags.writeable = False
    return a


@dataclass(frozen=True)
class IIGraph:
    """Undirected weighted graph over the ingredient vocabulary.

    Stored one-directional (each pair appears once, ``src < dst``). Consumers
    that need both directions call :meth:`symmetric`; forgetting to symmetrise
    is the single most common source of a silently halved degree.
    """

    src: np.ndarray
    dst: np.ndarray
    npmi: np.ndarray
    count: np.ndarray
    unigram: np.ndarray
    itos: list[str]
    n_recipes: int
    name: str

    @property
    def n_vocab(self) -> int:
        return len(self.itos)

    @property
    def n_edges(self) -> int:
        return len(self.src)

    def symmetric(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Both directions of every edge, with NPMI shifted positive.

        Random walks sample proportionally to weight, so a negative NPMI would
        be an invalid probability. Shifting preserves the ordering, which makes
        a weakly-associated pair rare rather than impossible — dropping them
        instead would disconnect exactly the sparse region the walk needs to
        explore.
        """
        w = self.npmi.astype(np.float64)
        w = w - w.min() + 1e-3
        return (np.concatenate([self.src, self.dst]),
                np.concatenate([self.dst, self.src]),
                np.concatenate([w, w]))

    def degree(self) -> np.ndarray:
        """Symmetric degree — both endpoints counted, as the walk sees it."""
        n = self.n_vocab
        return (np.bincount(self.src, minlength=n)
                + np.bincount(self.dst, minlength=n)).astype(np.float64)

    def edge_key_set(self) -> set[int]:
        """Undirected edges as packed integers, for O(1) non-edge tests when
        sampling link-prediction negatives."""
        n = self.n_vocab
        lo = np.minimum(self.src, self.dst).astype(np.int64)
        hi = np.maximum(self.src, self.dst).astype(np.int64)
        return set((lo * n + hi).tolist())

    def dense(self, weight: str = "count") -> np.ndarray:
        """Symmetric dense matrix. At n=1,790 this is 25 MB — small enough that
        closed-form factorisation is a seconds-long operation, which is why the
        factorisation compartment can exist at all."""
        n = self.n_vocab
        vals = {"count": self.count, "npmi": self.npmi}[weight].astype(np.float64)
        M = np.zeros((n, n), np.float64)
        np.add.at(M, (self.src.astype(int), self.dst.astype(int)), vals)
        np.add.at(M, (self.dst.astype(int), self.src.astype(int)), vals)
        return M


@dataclass(frozen=True)
class ChemGraph:
    """Bipartite ingredient -> flavour-compound incidence, with compound classes.

    ``ctype`` assigns each compound to one of 15 chemical categories. Keeping
    the category is what allows a *typed* metapath: a walk confined to
    organosulfur compounds connects garlic/onion/leek, while one confined to
    lactones connects dairy and stone fruit. An untyped bipartite walk averages
    those channels together and loses the structure.
    """

    src: np.ndarray
    dst: np.ndarray
    ctype: np.ndarray
    class_names: list[str]
    itos: list[str]

    @property
    def n_compounds(self) -> int:
        return len(self.ctype)

    def incidence(self, n_vocab: int) -> np.ndarray:
        A = np.zeros((n_vocab, self.n_compounds), np.float64)
        A[self.src.astype(int), self.dst.astype(int)] = 1.0
        return A


@functools.lru_cache(maxsize=8)
def load_ii_graph(name: str = GRAPH_TRAIN) -> IIGraph:
    path = _resolve(PATHS.graphs / name)
    z = np.load(path, allow_pickle=True)
    return IIGraph(
        src=_readonly(z["src"].astype(np.int64)),
        dst=_readonly(z["dst"].astype(np.int64)),
        npmi=_readonly(z["npmi"].astype(np.float64)),
        count=_readonly(z["count"].astype(np.float64)),
        unigram=_readonly(z["uni"].astype(np.float64)),
        itos=[str(x) for x in z["itos"]],
        n_recipes=int(z["n_recipes"]), name=name)


@functools.lru_cache(maxsize=2)
def load_heldout(name: str = GRAPH_HELDOUT) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(_resolve(PATHS.graphs / name), allow_pickle=True)
    return _readonly(z["src"].astype(np.int64)), _readonly(z["dst"].astype(np.int64))


@functools.lru_cache(maxsize=2)
def load_chem_graph(name: str = CHEM_GRAPH) -> ChemGraph:
    z = np.load(_resolve(PATHS.graphs / name), allow_pickle=True)
    return ChemGraph(
        src=_readonly(z["src"].astype(np.int64)),
        dst=_readonly(z["dst"].astype(np.int64)),
        ctype=_readonly(z["ctype"].astype(np.int64)),
        class_names=[str(x) for x in z["class_names"]],
        itos=[str(x) for x in z["itos"]])


@functools.lru_cache(maxsize=1)
def vocab(name: str = GRAPH_FULL) -> list[str]:
    """The canonical ingredient vocabulary, in id order.

    Read from the full graph, not a split: splits remove edges, never terms, so
    every split shares one vocabulary and embeddings stay row-comparable.
    """
    return load_ii_graph(name).itos


def _resolve(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Populate the workspace first:\n"
            f"    python scripts/import_data.py --from <llmmm-checkout>")
    return path
