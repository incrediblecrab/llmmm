"""The Reasoner — blends evidence and shows its working."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..artifacts import Manifest, iter_runs, load_embedding
from ..data.graphs import load_chem_graph, load_ii_graph
from ..data.splits import DEFAULT_SPLIT, get_split
from ..eval.metrics import unit
from .dedup import canonical, dedup_ranking, is_near_duplicate

#: Co-occurrence dominates because it is direct evidence and measurably better
#: at ranking real pairs (AUC 0.984 vs the best embedding's 0.873). The
#: embedding earns its weight on pairs that have never co-occurred, where
#: counting is silent. Chemistry is zero by default: H5 was falsified, and
#: carrying a signal at zero weight documents that it was tested and rejected,
#: which a deleted signal would not.
DEFAULT_WEIGHTS = {"cooccurrence": 0.6, "embedding": 0.4, "chemistry": 0.0}

#: Below this NPMI a pair is treated as unobserved rather than as evidence
#: against. NPMI is defined for pairs that co-occur at all; absence of a pair
#: from a 4.6M-recipe corpus is weak evidence of incompatibility, not strong.
UNOBSERVED = -1.0


@dataclass
class Evidence:
    """What each witness said about one pair, before any blending."""

    a: str
    b: str
    cooccurrence: float | None = None
    count: int = 0
    embedding: float | None = None
    chemistry: float | None = None
    shared_compounds: int = 0
    score: float = 0.0
    bridges: list[tuple[str, float]] = field(default_factory=list)

    @property
    def disagreement(self) -> float:
        """Spread between the witnesses that actually spoke.

        Reported rather than resolved. A high value means the model has learned
        something the corpus does not directly show — which is either the
        generalisation the model exists to provide or an artefact of its
        geometry, and nothing in the numbers distinguishes those.
        """
        vals = [v for v in (self.cooccurrence, self.embedding) if v is not None]
        return float(max(vals) - min(vals)) if len(vals) > 1 else 0.0

    def render(self) -> str:
        def fmt(v):
            return "     —" if v is None else f"{v:+6.3f}"

        lines = [
            f"  {self.a}  <->  {self.b}",
            f"    blended score        {self.score:+6.3f}",
            "",
            f"    co-occurrence NPMI   {fmt(self.cooccurrence)}"
            f"   ({self.count:,} recipes together)"
            if self.count else
            f"    co-occurrence NPMI   {fmt(self.cooccurrence)}   (never together)",
            f"    embedding cosine     {fmt(self.embedding)}",
            f"    chemistry overlap    {fmt(self.chemistry)}"
            f"   ({self.shared_compounds} shared compounds)",
        ]
        if self.disagreement > 0.5:
            direction = ("the model rates this above the corpus"
                         if (self.embedding or 0) > (self.cooccurrence or 0)
                         else "the corpus rates this above the model")
            lines += ["", f"    ! witnesses disagree by {self.disagreement:.2f}"
                          f" — {direction}."]
        if self.bridges:
            lines += ["", "    connected through:"]
            lines += [f"      {n:<24}{s:+.3f}" for n, s in self.bridges]
        return "\n".join(lines)


class Reasoner:
    """Answers questions about a trained model using every available signal."""

    def __init__(self, W: np.ndarray | None, itos: list[str], graph,
                 chem=None, weights: dict | None = None,
                 run_id: str = "<none>"):
        self.itos = list(itos)
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.run_id = run_id
        self.W = W
        self.U = unit(W) if W is not None else None
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.n = len(self.itos)
        # The real vocabulary drives duplicate detection: whether `peanut` is an
        # ingredient in its own right is a fact about this corpus, not a guess.
        self.vocab_set = frozenset(canonical(s) for s in self.itos)
        self._chem_incidence: np.ndarray | None = None
        self._npmi, self._count = self._dense_graph(graph)
        self._chem = self._chem_similarity(chem) if chem is not None else None

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, run_dir: Path | None = None, *, split: str | None = None,
             weights: dict | None = None) -> "Reasoner":
        """Load a run, or fall back to corpus statistics alone.

        ``run_dir=None`` is a supported mode, not a degraded one: co-occurrence
        is the strongest single signal, so a Reasoner with no model still
        answers well and serves as the control that any model must beat.
        """
        W, run_id = None, "<statistics only>"
        if run_dir is None:
            runs = list(iter_runs())
            if runs:
                run_dir = runs[0]
        if run_dir is not None:
            man = Manifest.load(run_dir)
            W, run_id = load_embedding(run_dir), man.run_id
            split = split or man.params.get("split", DEFAULT_SPLIT)
        sp = get_split(split or DEFAULT_SPLIT)
        graph = load_ii_graph(sp.graph)
        try:
            chem = load_chem_graph()
        except FileNotFoundError:
            chem = None
        return cls(W, list(graph.itos), graph, chem, weights, run_id)

    # ------------------------------------------------------------- signal prep
    def _dense_graph(self, g) -> tuple[np.ndarray, np.ndarray]:
        n = self.n
        npmi = np.full((n, n), UNOBSERVED, np.float32)
        cnt = np.zeros((n, n), np.int64)
        src, dst = np.asarray(g.src, int), np.asarray(g.dst, int)
        npmi[src, dst] = npmi[dst, src] = np.asarray(g.npmi, np.float32)
        cnt[src, dst] = cnt[dst, src] = np.asarray(g.count, np.int64)
        np.fill_diagonal(npmi, UNOBSERVED)
        return npmi, cnt

    def _chem_similarity(self, chem) -> np.ndarray | None:
        try:
            M = np.asarray(chem.incidence(self.n), np.float32)
        except (AttributeError, ValueError):
            return None
        if M.shape[0] != self.n or not M.any():
            return None
        self._chem_incidence = M
        return unit(M) @ unit(M).T

    def _idx(self, name: str) -> int:
        if name in self.stoi:
            return self.stoi[name]
        alt = name.strip().lower().replace(" ", "_")
        if alt in self.stoi:
            return self.stoi[alt]
        near = [s for s in self.itos if alt in s][:8]
        hint = f" Did you mean: {', '.join(near)}?" if near else ""
        raise KeyError(f"unknown ingredient {name!r}.{hint}")

    # ------------------------------------------------------------------ public
    def explain_pair(self, a: str, b: str, top_bridges: int = 5) -> str:
        i, j = self._idx(a), self._idx(b)
        ev = Evidence(a=self.itos[i], b=self.itos[j], count=int(self._count[i, j]))
        parts: list[tuple[float, float]] = []

        npmi = float(self._npmi[i, j])
        if npmi > UNOBSERVED:
            ev.cooccurrence = npmi
            parts.append((self.weights["cooccurrence"], npmi))
        if self.U is not None:
            ev.embedding = float(self.U[i] @ self.U[j])
            parts.append((self.weights["embedding"], ev.embedding))
        if self._chem is not None:
            ev.chemistry = float(self._chem[i, j])
            if self._chem_incidence is not None:
                ev.shared_compounds = int(np.count_nonzero(
                    self._chem_incidence[i] * self._chem_incidence[j]))
            parts.append((self.weights["chemistry"], ev.chemistry))

        # Renormalise over the witnesses that spoke, so a pair with no
        # co-occurrence is scored on the remaining evidence rather than being
        # penalised for a missing term it could never have supplied.
        wsum = sum(w for w, _ in parts) or 1.0
        ev.score = sum(w * v for w, v in parts) / wsum
        ev.bridges = self.bridges(a, b, k=top_bridges)
        return ev.render()

    def bridges(self, a: str, b: str, k: int = 5) -> list[tuple[str, float]]:
        """Ingredients that go with both — the shortest culinary path between
        two things that do not themselves pair."""
        i, j = self._idx(a), self._idx(b)
        both = np.minimum(self._npmi[i], self._npmi[j])
        both[[i, j]] = UNOBSERVED
        order = np.argsort(-both)[: k * 4]
        cand = [(self.itos[t], float(both[t])) for t in order
                if both[t] > UNOBSERVED
                and not is_near_duplicate(self.itos[t], self.itos[i], self.vocab_set)
                and not is_near_duplicate(self.itos[t], self.itos[j], self.vocab_set)]
        return cand[:k]

    def neighbors(self, ingredient: str, k: int = 15, dedup: bool = True,
                  signal: str = "blend") -> list[tuple[str, float]]:
        i = self._idx(ingredient)
        if signal == "cooccurrence" or self.U is None:
            scores = self._npmi[i].astype(np.float64).copy()
        elif signal == "embedding":
            scores = (self.U @ self.U[i]).astype(np.float64)
        else:
            emb = (self.U @ self.U[i]).astype(np.float64)
            co = self._npmi[i].astype(np.float64)
            seen = co > UNOBSERVED
            w_co, w_e = self.weights["cooccurrence"], self.weights["embedding"]
            scores = np.where(seen, (w_co * co + w_e * emb) / (w_co + w_e), emb)
        scores[i] = -np.inf
        order = np.argsort(-scores)[: k * 6 if dedup else k]
        names = [self.itos[t] for t in order]
        vals = [scores[t] for t in order]
        return (dedup_ranking(names, vals, k, self.vocab_set) if dedup
                else list(zip(names, vals)))
