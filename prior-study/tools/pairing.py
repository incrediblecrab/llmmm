#!/usr/bin/env python3
"""Pairing engine — "I have X, can it pair with Y?"

Applies the paper's own reasoning rather than trying to match its tables.
The three siblings are three INDEPENDENT kinds of evidence:

    cooc  — do people actually cook these together?   (co-occurrence)
    core  — do they play the same culinary role?      (distilled/denoised)
    chem  — do they share flavour compounds?          (molecular)

Agreement across all three = confident pairing. DISAGREEMENT is the interesting
signal, and it is what makes this different from a recipe search:

    chem yes / cooc no  -> novel pairing with a molecular basis but no tradition
                           (this is the Blumenthal "food pairing hypothesis" move)
    cooc yes / chem no  -> traditional pairing that works for cultural reasons

Everything here is pure geometry: no LLM, microseconds, offline.

CALIBRATION MATTERS. Raw cosines are NOT comparable across siblings — mean
pairwise cosine is 0.098 (cooc), 0.348 (core), 0.116 (chem). A raw 0.4 is
extraordinary in cooc and mediocre in core. Every score is therefore reported as
a percentile against that sibling's own full pairwise distribution.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import re

import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SIBLINGS = ["cooc", "core", "chem"]
EVIDENCE = {
    "cooc": "people cook these together",
    "core": "similar culinary role",
    "chem": "shared flavour compounds",
}


# Semantic substitutes that share no name token and no food group, so neither
# heuristic catches them. FINDINGS 5 showed the paper's own near-duplicate filter
# is inconsistent and unrecoverable, so this is a deliberate UX list, not an
# attempt at fidelity. Extend it as real users hit cases.
SYNONYMS = {
    frozenset(x) for x in [
        ("lamb", "mutton"), ("cilantro", "coriander"), ("scallion", "green_onion"),
        ("chickpea", "garbanzo_bean"), ("aubergine", "eggplant"),
        ("courgette", "zucchini"), ("rocket", "arugula"), ("prawn", "shrimp"),
        ("maize", "corn"), ("swede", "rutabaga"), ("beetroot", "beet"),
        ("capsicum", "bell_pepper"), ("soda", "baking_soda"),
    ]
}


def unit(v, axis=-1, eps=1e-9):
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), eps)


@dataclass
class Sibling:
    name: str
    E: np.ndarray
    vocab: dict
    itos: list
    modes: list
    poles: dict
    quantiles: np.ndarray  # 1001-point CDF of the pairwise cosine distribution

    def pct(self, cos: float) -> float:
        """Where does this cosine sit in this sibling's own distribution?"""
        return float(np.searchsorted(self.quantiles, cos) / len(self.quantiles) * 100)


@dataclass
class Verdict:
    x: str
    y: str
    scores: dict = field(default_factory=dict)
    overall: float = 0.0
    label: str = ""
    headline: str = ""
    recipes: int = 0            # times the pair co-occurs in 2.1M real recipes
    npmi: float = float("nan")  # normalised PMI over those recipes
    evidence: str = "embedding"  # "recipes" when real corpus data was available
    shared_modes: list = field(default_factory=list)
    cuisine: tuple = ()
    bridges: list = field(default_factory=list)
    substitute: bool = False


def load_sibling(name: str, rng) -> Sibling:
    d = RAW / f"epicure-{name}"
    E = unit(load_file(d / "embeddings.safetensors")["embeddings"].astype(np.float32))
    vocab = json.loads((d / "vocab.json").read_text())
    itos = [None] * len(vocab)
    for k, v in vocab.items():
        itos[v] = k
    modes = json.loads((d / "modes.json").read_text())
    poles = {k: unit(np.asarray(v, np.float32))
             for k, v in json.loads((d / "supervised_poles.json").read_text()).items()}
    # empirical CDF from a large random sample of distinct pairs
    n = len(E)
    i = rng.integers(0, n, 400_000)
    j = rng.integers(0, n, 400_000)
    m = i != j
    sample = np.sum(E[i[m]] * E[j[m]], axis=1)
    return Sibling(name, E, vocab, itos, modes, poles,
                   np.quantile(sample, np.linspace(0, 1, 1001)))


# How much to trust real recipes over the embedding when the corpus has seen a
# pair, and how to split that between the two recipe statistics. Swept in
# tools/tune_recipe_weight.py. The sweep's argmax was W_RECIPE=1.0 (ignore the
# embedding entirely) at AUC 0.992, but that is one pair better than 0.976 on a
# 42-pair set and it breaks obvious cases -- vanilla+onion rises to 69 because
# both are common. Keeping 30% embedding and weighting the two recipe stats
# equally is materially more robust for a difference well inside the noise.
#
#   count  "people really do cook these together"
#   nPMI   "and it is not just that both are everywhere"
W_RECIPE = 0.7
W_NPMI = 0.5

# PMI is a *relative* measure and it breaks down for background ingredients. Butter
# appears in 26% of all recipes, so garlic+butter co-occurs slightly LESS than chance
# and scores negative nPMI -- despite 56,496 recipes using both. Statistically true
# (garlic skews savoury, butter skews baking), culinarily absurd.
#
# Above this many recipes, the absolute evidence outweighs the association measure:
# tens of thousands of published recipes doing something is not a clash. This is the
# top ~1% of pairs (p99 = 4,378), so it fires rarely and only where it is unarguable.
FLOOR_COUNT = 4_000
FLOOR_SCORE = 70.0


def _combine(pcts: list[float]) -> float:
    """Evidence combination. The three siblings are independent evidence types,
    so the strongest axis should dominate -- but a pairing supported on only one
    axis is weaker than one supported on all three, hence the median/min terms."""
    hi, mid, lo = sorted(pcts, reverse=True)
    return 0.5 * hi + 0.3 * mid + 0.2 * lo


class Pairing:
    def __init__(self, seed: int = 0, n_null: int = 30_000, recipes: bool = True):
        rng = np.random.default_rng(seed)
        self.S = {s: load_sibling(s, rng) for s in SIBLINGS}
        self.rec = self._load_recipes() if recipes else None
        self.cuisines = {s: [k for k in self.S[s].poles if k.startswith("cuisine:")]
                         for s in SIBLINGS}
        self.null = self._build_null(rng, n_null)
        lab = json.loads((RAW / "epicure-explorer" / "ingredient_labels.json").read_text())
        self.group = dict(zip(lab["names"], lab["food_groups"]))

    def _load_recipes(self):
        """Real co-occurrence counts from 2.1M recipes (tools/build_recipe_cooc.py).

        The `cooc` embedding is a 300-d compression of exactly this signal and
        it loses most of it -- rank correlation with true nPMI is only ~0.50,
        and it rates chicken+lemon (23,940 real recipes) a "clash". Where the
        corpus has an opinion we should use it; the embedding's real job is
        generalising to the 81% of pairs no recipe has ever tried.
        """
        f = ROOT / "data" / "derived" / "recipe_cooc.npz"
        if not f.exists():
            return None
        z = np.load(f, allow_pickle=True)
        pairs, cnt, npmi = z["pairs"], z["count"], z["npmi"]
        n = len(self.S["cooc"].vocab)
        keys = pairs[:, 0].astype(np.int64) * n + pairs[:, 1]
        order = np.argsort(keys)
        good = np.isfinite(npmi)
        lc = np.log1p(cnt.astype(np.float64))
        return {"n": n, "uni": z["uni"], "keys": keys[order], "cnt": cnt[order],
                "npmi": npmi[order],
                "q": np.quantile(npmi[good], np.linspace(0, 1, 1001)),
                "q_cnt": np.quantile(lc, np.linspace(0, 1, 1001))}

    def recipe_evidence(self, i: int, j: int):
        """(count, nPMI, percentile) for a pair, or None if never co-occurred."""
        if self.rec is None:
            return None
        a, b = (i, j) if i < j else (j, i)
        k = np.searchsorted(self.rec["keys"], a * self.rec["n"] + b)
        if k >= len(self.rec["keys"]) or \
                self.rec["keys"][k] != a * self.rec["n"] + b:
            return None
        v = float(self.rec["npmi"][k])
        c = int(self.rec["cnt"][k])
        cp = float(np.searchsorted(self.rec["q_cnt"], np.log1p(c))
                   / len(self.rec["q_cnt"]) * 100)
        pp = (float(np.searchsorted(self.rec["q"], v) / len(self.rec["q"]) * 100)
              if np.isfinite(v) else 50.0)
        return c, v, W_NPMI * pp + (1 - W_NPMI) * cp

    def _build_null(self, rng, n):
        """Empirical distribution of the combined score over random pairs.
        Without this, a random pair scores ~50 and looks 'plausible'. Every
        headline number is a percentile against THIS, i.e. 'better than X% of
        all 1.6M possible ingredient pairs'."""
        N = len(self.S["cooc"].vocab)
        i, j = rng.integers(0, N, n), rng.integers(0, N, n)
        m = i != j
        i, j = i[m], j[m]
        cols = []
        for s in SIBLINGS:
            sib = self.S[s]
            cos = np.sum(sib.E[i] * sib.E[j], axis=1)
            cols.append(np.searchsorted(sib.quantiles, cos) / len(sib.quantiles) * 100)
        M = np.sort(np.stack(cols, 1), axis=1)[:, ::-1]
        scores = 0.5 * M[:, 0] + 0.3 * M[:, 1] + 0.2 * M[:, 2]
        return np.sort(scores)

    # ---------------------------------------------------------------- helpers
    # Users type "cheddar", not "cheddar_cheese"; "chicken breast", not "chicken".
    # Exact lookup alone fails on all three and the caller sees a dead end, so the
    # resolver walks progressively looser strategies and stops at the first hit.
    _HEADS = ("cheese", "pepper", "oil", "sauce", "vinegar", "powder", "seed",
              "juice", "wine", "meat", "nut", "leaf", "root", "paste")
    _CUTS = ("breast", "thigh", "leg", "fillet", "filet", "loin", "chop", "shoulder",
             "mince", "ground", "steak", "wing", "rib", "shank", "drumstick")
    # Adjectives that name a style rather than a different ingredient. "greek
    # yogurt" is yogurt; note "green"/"red" are absent on purpose, since a green
    # pepper is a different ingredient from a pepper.
    _STYLE = ("greek", "plain", "whole", "skim", "low_fat", "nonfat", "fat_free",
              "unsalted", "salted", "raw", "cooked", "dried", "fresh", "frozen",
              "canned", "smoked", "extra", "virgin", "large", "small",
              "boneless", "skinless", "lean", "sweetened", "unsweetened")
    # Spellings and everyday words the 1,790-term vocabulary does not carry.
    _ALIAS = {"chilli": "chili", "chile": "chili", "yoghurt": "yogurt",
              "spaghetti": "pasta", "macaroni": "pasta", "penne": "pasta",
              "fusilli": "pasta", "linguine": "pasta", "noodles": "noodle",
              "aubergine": "eggplant", "courgette": "zucchini",
              "coriander": "cilantro", "rocket": "arugula", "prawn": "shrimp",
              "capsicum": "bell_pepper", "spring_onion": "scallion",
              "green_onion": "scallion", "confectioners_sugar": "powdered_sugar",
              "caster_sugar": "sugar", "double_cream": "heavy_cream",
              "minced_meat": "beef", "hamburger": "beef", "soda": "soft_drink",
              "green_pepper": "bell_pepper", "red_pepper": "bell_pepper",
              "prawn": "shrimp", "chili": "chili_pepper", "chilli": "chili_pepper",
              "chile": "chili_pepper", "scallions": "scallion"}

    def resolve(self, name: str) -> str | None:
        """Map free text to a vocabulary term, loosest-strategy-last.

        Every stage runs through _hit() so an alias can rescue a form that
        plural- or modifier-stripping produced ("prawns" -> "prawn" -> shrimp).
        """
        key = re.sub(r"[^a-z]+", "_", name.strip().lower()).strip("_")
        if (r := self._hit(key)):
            return r
        for stripped in self._depluralise(key):
            if (r := self._hit(stripped)):
                return r
            key = stripped
        for h in self._HEADS:                      # cheddar -> cheddar_cheese
            if f"{key}_{h}" in self.S["cooc"].vocab:
                return f"{key}_{h}"
        parts = key.split("_")
        while len(parts) > 1 and (parts[0] in self._STYLE or parts[0] in self._CUTS):
            parts = parts[1:]
            if (r := self._hit("_".join(parts))):
                return r
        while len(parts) > 1 and parts[-1] in self._CUTS:
            parts = parts[:-1]
            if (r := self._hit("_".join(parts))):
                return r
        # last resort: a unique vocab entry containing the query as a whole token
        hits = [v for v in self.S["cooc"].vocab if parts[-1] in v.split("_")]
        return hits[0] if len(hits) == 1 else None

    def _hit(self, key: str) -> str | None:
        vocab = self.S["cooc"].vocab
        if key in vocab:
            return key
        a = self._ALIAS.get(key)
        return a if a in vocab else None

    @staticmethod
    def _depluralise(key: str):
        if key.endswith("ies"):
            yield key[:-3] + "y"
        if key.endswith("es"):
            yield key[:-2]
        if key.endswith("s"):
            yield key[:-1]

    def _shared_modes(self, x, y, limit=4):
        out = []
        for s in SIBLINGS:
            sib = self.S[s]
            for m in sib.modes:
                mem = {n.replace(" ", "_") for n in m["members"]}
                if x in mem and y in mem:
                    # only the tight core of a mode is meaningful (REPLICATION.md)
                    idx = [sib.vocab[n] for n in mem if n in sib.vocab]
                    pole = unit(np.asarray(m["pole"], np.float32))
                    top8 = {sib.itos[i] for i in
                            sorted(idx, key=lambda i: -float(sib.E[i] @ pole))[:8]}
                    out.append((s, m["label"], x in top8 and y in top8))
        out.sort(key=lambda r: not r[2])
        return out[:limit]

    # A "novel" pairing claims cooks have overlooked something. That claim is
    # only meaningful when cooks had the chance to try it. Both ingredients must
    # be common enough in the corpus that never seeing them together is
    # informative -- otherwise absence of evidence is read as evidence of
    # absence, and the app confidently recommends pike + horse_meat.
    MIN_UNI = 400

    def _absence_is_evidence(self, x, y) -> bool:
        if self.rec is None:
            return True
        v = self.S["cooc"].vocab
        uni = self.rec["uni"]
        return bool(uni[v[x]] >= self.MIN_UNI and uni[v[y]] >= self.MIN_UNI)

    def _cuisine_lean(self, x, y):
        agree = []
        for s in SIBLINGS:
            sib = self.S[s]
            if not self.cuisines[s]:
                continue
            def best(n):
                return max(self.cuisines[s], key=lambda c: float(sib.E[sib.vocab[n]] @ sib.poles[c]))
            bx, by = best(x), best(y)
            if bx == by:
                agree.append((s, bx.split(":", 1)[1].replace("_", " ")))
        return agree

    def _bridges(self, x, y, k=5):
        """Ingredients close to BOTH — the 'add this and it works' move.
        Scored on min() so a bridge must genuinely serve both sides."""
        sib = self.S["core"]
        sx = sib.E @ sib.E[sib.vocab[x]]
        sy = sib.E @ sib.E[sib.vocab[y]]
        both = np.minimum(sx, sy)
        both[sib.vocab[x]] = both[sib.vocab[y]] = -np.inf
        out = []
        for i in np.argsort(-both)[:40]:
            n = sib.itos[i]
            # a bridge must not be a near-duplicate of either endpoint
            if n in x or x in n or n in y or y in n:
                continue
            out.append((n, float(both[i])))
            if len(out) >= k:
                break
        return out

    # ------------------------------------------------------------------ main
    def pair(self, a: str, b: str) -> Verdict | str:
        x, y = self.resolve(a), self.resolve(b)
        if x is None:
            return f"'{a}' is not in the 1,790-ingredient vocabulary"
        if y is None:
            return f"'{b}' is not in the 1,790-ingredient vocabulary"
        if x == y:
            return "same ingredient"

        v = Verdict(x, y)
        for s in SIBLINGS:
            sib = self.S[s]
            cos = float(sib.E[sib.vocab[x]] @ sib.E[sib.vocab[y]])
            v.scores[s] = (cos, sib.pct(cos))

        pcts = {s: p for s, (_, p) in v.scores.items()}
        raw = _combine(list(pcts.values()))
        overall = float(np.searchsorted(self.null, raw) / len(self.null) * 100)

        # Real recipes outrank the embedding when they have an opinion:
        # nPMI separates held-out human-judged pairings at AUC 0.984 vs the
        # embedding's 0.873 (tools/ground_truth.py).
        ev = self.recipe_evidence(self.S["cooc"].vocab[x], self.S["cooc"].vocab[y])
        if ev is not None:
            v.recipes, v.npmi, rp = ev
            v.evidence = "recipes"
            overall = W_RECIPE * rp + (1 - W_RECIPE) * overall
            if v.recipes >= FLOOR_COUNT:
                overall = max(overall, FLOOR_SCORE)
        v.overall = overall
        gap_novel = pcts["chem"] - pcts["cooc"]
        gap_trad = pcts["cooc"] - pcts["chem"]

        # Disagreement is more informative than the average, so test it first.
        # Tiers below are calibrated empirically: 20 canonical pairings score
        # min 65.5 / median 95.6; 7 implausible ones score max 66.9. See
        # tools/calibrate_pairing.py. The disagreement labels additionally
        # require a floor of 65 so nonsense can't be sold as a "discovery".
        # Two ingredients that share a culinary role AND a compound profile are
        # interchangeable, not complementary -- that is the substitute
        # relationship, and calling it a "novel pairing" is how the engine ends
        # up recommending quail + tilapia. A pairing needs contrast somewhere.
        substitutes = pcts["core"] >= 90 or pcts["chem"] >= 97
        if (overall >= 65 and gap_novel >= 30 and pcts["chem"] >= 70
                and not substitutes and self._absence_is_evidence(x, y)):
            v.label = "novel"
            v.headline = ("Shares flavour compounds but is rarely cooked together — "
                          "a real discovery rather than a safe bet.")
        elif overall >= 65 and gap_trad >= 30 and pcts["cooc"] >= 70:
            v.label = "traditional"
            v.headline = ("An established culinary pairing that is NOT explained by "
                          "shared compounds — it works for cultural reasons.")
        elif overall >= 97:
            v.label, v.headline = "excellent", "As strong as any classic pairing."
        elif overall >= 92:
            v.label, v.headline = "strong", "Comparable to established pairings."
        elif overall >= 80:
            v.label, v.headline = "plausible", "Reasonable — worth trying."
        elif overall >= 65:
            v.label, v.headline = "stretch", "Thin; use a bridge ingredient."
        else:
            v.label, v.headline = "clash", "Weaker than almost every known pairing."

        # Similarity is not the same as complementarity. Two ingredients with
        # nearly identical embeddings are SUBSTITUTES (rice/brown_rice), not a
        # pairing -- same root cause as the near-duplicate problem in FINDINGS 5.
        tx, ty = set(x.split("_")), set(y.split("_"))
        gx, gy = self.group.get(x, "?"), self.group.get(y, "?")
        same_group = gx == gy and gx not in ("?", "Other")
        if (tx & ty) or frozenset((x, y)) in SYNONYMS or (
                same_group and pcts["core"] >= 99.5):
            v.substitute = True
            v.label = "substitute"
            v.headline = ("These are near-interchangeable, not complementary. "
                          "Use one OR the other, not both.")

        v.shared_modes = self._shared_modes(x, y)
        v.cuisine = self._cuisine_lean(x, y)
        v.bridges = self._bridges(x, y)
        return v


def render(v: Verdict) -> str:
    if isinstance(v, str):
        return f"  {v}"
    L = [f"\n{'='*66}", f"  {v.x}  +  {v.y}", f"{'='*66}"]
    for s in SIBLINGS:
        cos, pct = v.scores[s]
        bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
        mark = "YES" if pct >= 90 else ("~  " if pct >= 60 else "no ")
        L.append(f"  {s:<5} {bar} p{pct:5.1f}  cos={cos:+.3f}  {mark} {EVIDENCE[s]}")
    L.append(f"\n  VERDICT: {v.label.upper()}  (better than {v.overall:.1f}% of all "
             f"1.6M ingredient pairs) — {v.headline}")
    if v.substitute:
        L.append("  note: flagged as a SUBSTITUTE pair, not a complementary pairing")
    if v.shared_modes:
        L.append("  shared contexts:")
        for s, lab, core in v.shared_modes:
            L.append(f"     [{s}] {lab}{'  (both in mode core)' if core else ''}")
    if v.cuisine:
        L.append("  cuisine agreement: " +
                 ", ".join(f"{c} ({s})" for s, c in v.cuisine))
    if v.bridges:
        L.append("  bridge via: " + ", ".join(f"{n} ({c:.2f})" for n, c in v.bridges))
    return "\n".join(L)


if __name__ == "__main__":
    P = Pairing()
    print(f"loaded {len(P.S['cooc'].vocab)} ingredients x 3 siblings")
    for a, b in [("strawberry", "basil"), ("pork", "apple"),
                 ("white chocolate", "caviar"), ("chocolate", "blue cheese"),
                 ("tomato", "basil"), ("banana", "parsley"),
                 ("coffee", "garlic"), ("lamb", "mint")]:
        print(render(P.pair(a, b)))
