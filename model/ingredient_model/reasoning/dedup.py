"""Near-duplicate detection over the ingredient vocabulary.

``rice`` and ``brown_rice`` are not a useful recommendation pair, and neither
are ``lamb`` and ``mutton``. Any similarity model will rank them at the top
because they *are* similar — that is the model working correctly and the product
failing. Suppressing them is a presentation decision, so it lives here rather
than being trained away.

Detection is lexical rather than learned, deliberately: a learned duplicate
detector would be fitted on the same co-occurrence statistics that produce the
problem, so it would confidently agree that the duplicates are distinct.
"""
from __future__ import annotations

import re

_SPLIT = re.compile(r"[_\s\-]+")

#: Modifiers that describe a *form* of an ingredient rather than a different
#: ingredient. Stripping these collapses ``ground_beef``/``beef`` but leaves
#: ``beef``/``chicken`` alone.
FORM_WORDS = frozenset({
    "fresh", "frozen", "dried", "dry", "ground", "chopped", "sliced", "diced",
    "minced", "grated", "shredded", "whole", "raw", "cooked", "canned",
    "unsalted", "salted", "sweet", "unsweetened", "low", "fat", "free",
    "reduced", "light", "extra", "virgin", "pure", "large", "small", "medium",
    "boneless", "skinless", "lean", "ripe", "toasted", "roasted", "smoked",
    "powdered", "crushed", "peeled", "seedless", "instant", "quick", "plain",
    "white", "brown", "red", "green", "yellow", "black", "baby", "wild",
    "organic", "natural", "hot", "cold", "warm", "thick", "thin", "fine",
    "coarse", "granulated", "confectioners", "all", "purpose",
})

#: Pairs that survive stemming but still refer to the same thing. Kept short and
#: explicit: a long list here is a sign the vocabulary needs fixing upstream.
SYNONYMS: dict[str, str] = {
    "mutton": "lamb", "scallion": "green_onion", "spring_onion": "green_onion",
    "coriander": "cilantro", "aubergine": "eggplant", "courgette": "zucchini",
    "capsicum": "bell_pepper", "garbanzo": "chickpea", "maize": "corn",
    "prawn": "shrimp", "rocket": "arugula", "swede": "rutabaga",
    "confectioners_sugar": "powdered_sugar", "caster_sugar": "sugar",
    "soda": "baking_soda", "bicarbonate": "baking_soda",
}


def _stem(word: str) -> str:
    for suffix in ("ies", "es", "s"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def canonical(name: str) -> str:
    """Reduce a name to the thing it is, dropping form and quantity words."""
    name = SYNONYMS.get(name, name)
    parts = [_stem(w) for w in _SPLIT.split(name.lower()) if w]
    core = [w for w in parts if w not in FORM_WORDS]
    # Everything was a modifier ("extra virgin"), so the modifiers *are* the
    # name and dropping them would collapse it into every other such term.
    return "_".join(core or parts)


#: Modifier words that are themselves ingredients. `peanut_butter` shares a head
#: with `butter` but is not a kind of it, whereas `cheddar_cheese` is a kind of
#: cheese — the difference is that "peanut" names an ingredient in its own right
#: and "cheddar" does not. Callers holding the real vocabulary should pass it to
#: :func:`is_near_duplicate`, which subsumes this list; it exists so the rule
#: still works when no vocabulary is at hand.
STANDALONE = frozenset({
    "peanut", "almond", "coconut", "soy", "cashew", "walnut", "pecan",
    "hazelnut", "sesame", "apple", "orange", "lemon", "lime", "coffee",
    "chocolate", "vanilla", "honey", "garlic", "onion", "tomato", "potato",
    "corn", "rice", "oat", "wheat", "barley", "chicken", "beef", "pork",
    "fish", "shrimp", "egg", "cheese", "butter", "cream", "milk", "vinegar",
    "wine", "beer", "mint", "ginger", "banana", "mango", "olive",
})


def is_near_duplicate(a: str, b: str, vocab: frozenset[str] | None = None) -> bool:
    """True when two names denote the same ingredient at different precision.

    ``vocab`` is the set of canonical ingredient names in use. When supplied it
    replaces the built-in :data:`STANDALONE` list, so the rule adapts to the
    actual vocabulary instead of a guess about it.
    """
    ca, cb = canonical(a), canonical(b)
    if ca == cb:
        return True
    ta, tb = ca.split("_"), cb.split("_")
    if not (set(ta) <= set(tb) or set(tb) <= set(ta)):
        return False
    # One name contains the other, but containment alone is not enough. In an
    # English compound the *head* is the last word, so `rice_vinegar` is a
    # vinegar and not a kind of rice, while `brown_rice` is a kind of rice.
    if ta[-1] != tb[-1]:
        return False
    # Heads agree, so the longer name is "head, qualified". That is only a
    # duplicate if the qualifier is a description; if the qualifier is itself an
    # ingredient the compound names a different thing (`peanut_butter`).
    known = vocab if vocab is not None else STANDALONE
    extra = set(ta) ^ set(tb)
    return not (extra & known)


def dedup_ranking(names, scores, k: int, vocab: frozenset[str] | None = None):
    """Keep the highest-scoring member of each near-duplicate group."""
    kept: list[tuple[str, float]] = []
    for name, score in zip(names, scores):
        if any(is_near_duplicate(name, prev, vocab) for prev, _ in kept):
            continue
        kept.append((name, float(score)))
        if len(kept) >= k:
            break
    return kept
