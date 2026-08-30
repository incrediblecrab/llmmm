"""Evaluation labels — the substitution catalogue.

210,612 human-voted "you can use B instead of A" pairs, scraped independently of
the recipe corpus and never used in training. They are the only source of ground
truth here that is not itself a co-occurrence statistic, which is what makes
them able to falsify a model rather than merely agree with it.

Two vote tiers are reported for every model, fixed in advance:

``broad``   votes >= 1   — maximum statistical power
``strict``  votes >= 10  — requires community agreement

Reporting both is deliberate. The vote distribution has median 1, so a single
high threshold discards ~93% of usable labels; a single low one admits noise.
Fixing two tiers up front removes the option of choosing the flattering one
after seeing results.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

from ..config import PATHS

TIERS = {"broad": 1, "strict": 10}
SUBSTITUTIONS = "substitutions.parquet"


@dataclass(frozen=True)
class Substitutions:
    pairs: dict[str, list[tuple[int, int]]]
    by_anchor: dict[str, dict[int, set[int]]]

    def tier(self, name: str) -> list[tuple[int, int]]:
        return self.pairs[name]

    def anchors(self, name: str, min_subs: int = 3) -> dict[int, set[int]]:
        return {k: v for k, v in self.by_anchor[name].items() if len(v) >= min_subs}

    def summary(self) -> dict:
        return {t: {"pairs": len(self.pairs[t]),
                    "anchors_ge3": len(self.anchors(t))} for t in TIERS}


@functools.lru_cache(maxsize=1)
def load_substitutions(itos: tuple[str, ...]) -> Substitutions:
    """Map the catalogue onto the model vocabulary.

    ``itos`` is a tuple rather than a list purely so this can be cached; the
    mapping depends on the vocabulary, so caching on it is correct.
    """
    import pandas as pd

    path = PATHS.catalog / SUBSTITUTIONS
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Populate the workspace first:\n"
            f"    python scripts/import_data.py --from <llmmm-checkout>")
    stoi = {s: i for i, s in enumerate(itos)}
    df = pd.read_parquet(path)
    df = df[df.ingredient_vocab.isin(stoi) & df.alternative_vocab.isin(stoi)]

    pairs: dict[str, list[tuple[int, int]]] = {}
    by_anchor: dict[str, dict[int, set[int]]] = {}
    for tier, min_votes in TIERS.items():
        s = df[df.votes >= min_votes]
        uniq = {(stoi[a], stoi[b])
                for a, b in zip(s.ingredient_vocab, s.alternative_vocab)
                if stoi[a] != stoi[b]}
        pairs[tier] = sorted(uniq)
        m: dict[int, set[int]] = {}
        for ia, ib in uniq:
            m.setdefault(ia, set()).add(ib)
        by_anchor[tier] = m
    return Substitutions(pairs=pairs, by_anchor=by_anchor)


def as_arrays(pairs: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    if not pairs:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    a = np.fromiter((p[0] for p in pairs), np.int64, len(pairs))
    b = np.fromiter((p[1] for p in pairs), np.int64, len(pairs))
    return a, b
