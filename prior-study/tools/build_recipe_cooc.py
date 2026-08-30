#!/usr/bin/env python3
"""Build a REAL ingredient co-occurrence matrix from actual recipes.

Everything in this project so far has been validated against Epicure's derived
artifacts -- the 1,790x300 embeddings and their published result tables. Those
are a lossy compression of 4.14M recipes that we never actually looked at. This
script closes that gap for the largest single source (RecipeNLG, 53.9% of the
Epicure corpus) so we can ask ground-truth questions:

  - does the `cooc` embedding actually preserve recipe co-occurrence?
  - when the engine says pork+apple is a "clash", is that a compression
    artefact, or do they genuinely not co-occur?

Reads the NER column (RecipeNLG's own pre-extracted ingredient entities),
matches to the Epicure vocabulary, and writes counts + PMI.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"          # Epicure model artifacts (vocab, embeddings)
CORPUS = ROOT / "recipe"             # verified Table A1 corpora, one dir per source
OUT = ROOT / "data" / "derived"
# The real RecipeNLG full_dataset.csv, with the pre-extracted NER ingredient
# column the paper's pipeline consumes. 2,230,569 recipes after dropping the
# 573 rows with an empty NER list.
SRC = CORPUS / "01-recipenlg" / "RecipeNLG_dataset.csv"

csv.field_size_limit(1 << 30)


def load_vocab() -> tuple[dict, list]:
    v = json.loads((RAW / "epicure-cooc" / "vocab.json").read_text())
    itos = [None] * len(v)
    for k, i in v.items():
        itos[i] = k
    return v, itos


def build_alias(vocab: dict) -> dict:
    """Map surface forms seen in recipes onto Epicure vocab ids.

    RecipeNLG NER strings are already lemmatised-ish ("ground beef", "olive
    oil"), so the main gap is the underscore convention plus a few very common
    culinary synonyms. Kept deliberately small and auditable -- an aggressive
    fuzzy matcher would manufacture co-occurrences that aren't real.
    """
    alias = {}
    for name, i in vocab.items():
        alias[name] = i
        alias[name.replace("_", " ")] = i
    extra = {
        "scallions": "scallion", "green onion": "scallion",
        "green onions": "scallion", "spring onion": "scallion",
        "cilantro": "coriander", "coriander leaves": "coriander",
        "garbanzo beans": "chickpea", "confectioners sugar": "powdered_sugar",
        "all purpose flour": "flour", "all-purpose flour": "flour",
        "plain flour": "flour", "unsalted butter": "butter",
        "salted butter": "butter", "extra virgin olive oil": "olive_oil",
        "kosher salt": "salt", "sea salt": "salt", "table salt": "salt",
        "freshly ground black pepper": "black_pepper", "pepper": "black_pepper",
        "ground beef": "beef", "beef broth": "beef", "chicken broth": "chicken",
        "chicken breasts": "chicken", "chicken breast": "chicken",
        "boneless skinless chicken breasts": "chicken",
        "granulated sugar": "sugar", "white sugar": "sugar",
        "brown sugar": "brown_sugar", "vanilla": "vanilla_extract",
        "eggs": "egg", "egg whites": "egg", "egg yolks": "egg",
        "milk": "milk", "whole milk": "milk", "heavy cream": "cream",
        "whipping cream": "cream", "sour cream": "sour_cream",
        "parmesan": "parmesan_cheese", "cheddar": "cheddar_cheese",
        "mozzarella": "mozzarella_cheese", "tomatoes": "tomato",
        "onions": "onion", "potatoes": "potato", "carrots": "carrot",
        "mushrooms": "mushroom", "apples": "apple", "lemon juice": "lemon",
        "lime juice": "lime", "orange juice": "orange",
        "garlic cloves": "garlic", "cloves garlic": "garlic",
        "clove garlic": "garlic", "fresh garlic": "garlic",
        "olive oil": "olive_oil", "vegetable oil": "vegetable_oil",
        "soy sauce": "soy_sauce", "worcestershire sauce": "worcestershire_sauce",
        # top unmatched NER surface forms, checked individually against vocab
        "soda": "baking_soda", "oleo": "margarine", "catsup": "ketchup",
        "hamburger": "beef", "cocoa": "cocoa_powder", "crisco": "shortening",
        "salad oil": "vegetable_oil", "cooking oil": "vegetable_oil",
        "green pepper": "bell_pepper", "red pepper": "bell_pepper",
        "grated cheese": "cheese", "boiling water": "water",
        "cold water": "water", "warm water": "water", "hot water": "water",
        "dry mustard": "mustard", "garlic salt": "garlic",
        "vanilla extract": "vanilla", "chocolate chips": "chocolate",
        "garlic powder": "garlic", "onion powder": "onion",
    }
    for k, target in extra.items():
        if target in vocab:
            alias.setdefault(k, vocab[target])
    # Plurals must be explicit surface forms: the matcher uses word boundaries,
    # so \bpecan\b will not fire on "pecans".
    for name in list(alias):
        i = alias[name]
        if name.endswith(("s", "x", "z", "ch", "sh")):
            alias.setdefault(name + "es", i)
        elif name.endswith("y") and len(name) > 3:
            alias.setdefault(name[:-1] + "ies", i)
        else:
            alias.setdefault(name + "s", i)
    return alias


# Modifiers that describe preparation or state, never identity. "green" and
# "red" are deliberately absent -- "green pepper" is a bell pepper, not a
# peppercorn, and stripping the colour would silently corrupt the counts.
_MODS = ("ground ", "fresh ", "dried ", "chopped ", "minced ", "grated ",
         "shredded ", "sliced ", "whole ", "large ", "small ", "medium ",
         "frozen ", "canned ", "cooked ", "raw ", "finely ", "coarsely ",
         "peeled ", "crushed ", "melted ", "softened ", "beaten ", "toasted ")


def lookup(alias: dict, t: str):
    """Resolve one NER surface form, trying progressively looser forms."""
    for cand in (t, t.replace(" ", "_")):
        if cand in alias:
            return alias[cand]
    changed = True
    while changed:                                   # strip stacked modifiers
        changed = False
        for m in _MODS:
            if t.startswith(m) and len(t) > len(m) + 2:
                t, changed = t[len(m):], True
        if t in alias:
            return alias[t]
    if t.endswith("es") and t[:-2] in alias:         # tomatoes -> tomato
        return alias[t[:-2]]
    if t.endswith("s") and t[:-1] in alias:          # pecans -> pecan
        return alias[t[:-1]]
    if t.endswith("ies") and t[:-3] + "y" in alias:  # berries -> berry
        return alias[t[:-3] + "y"]
    return None


def build_matcher(alias: dict):
    """One regex over every known surface form, longest alternative first.

    Longest-first matters: Python alternation is first-match-wins, so without it
    "buttermilk" would match "butter" and corrupt the counts. Word boundaries
    stop "pear" matching inside "spearmint".
    """
    forms = sorted({k for k in alias if len(k) > 2}, key=len, reverse=True)
    pat = re.compile(r"\b(" + "|".join(re.escape(f) for f in forms) + r")\b")
    return pat


# Strip a leading quantity + unit: "3 cloves garlic" -> "garlic".
#
# The unit group MUST end with a (?=\s) lookahead. Without it the single-letter
# alternatives g / l / t match the first letter of the ingredient itself, so a
# bare "lemon" became "emon" and "garlic" became "arlic" -- silently deleting
# every g- and l-initial ingredient (garlic, ginger, lemon, lime, lettuce, lamb,
# leek) from any line that had no explicit unit. That bug cost ~5% of all
# ingredient lines before it was caught by tools/audit_corpus_coverage.py.
_QTY = re.compile(
    r"^[\s\-\u2022*]*[\d/\u00bc\u00bd\u00be\u2153\u2154\u215b\.\s]*"
    r"(?:(?:cups?|c\.|tbsp?\.?|tablespoons?|tsp\.?|teaspoons?|oz\.?|ounces?"
    r"|lbs?\.?|pounds?|pkg\.?|packages?|cans?|qt\.?|quarts?|pt\.?|pints?"
    r"|gal\.?|g|kg|ml|l|t\.?|T\.?|sticks?|cloves?|slices?|pieces?"
    r"|dash(?:es)?|pinch(?:es)?)(?=\s))?\s*")


def iter_allrecipes(limit=None):
    """Yield ingredient-line lists from the corbt/all-recipes text dump.

    DEPRECATED. This is a 2,147,248-row derived mirror, not a Table A1 source;
    it under-counts RecipeNLG by 83,321 recipes and has no NER column, so
    ingredient lines must be recovered with the _QTY regex. Prefer
    source="ner", which reads the real full_dataset.csv. Kept for A/B only.
    """
    import pyarrow.parquet as pq
    n = 0
    src = CORPUS / "_superseded" / "corbt-all-recipes-NOT-table-a1"
    for f in sorted(src.glob("ar_*.parquet")):
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            for txt in pf.read_row_group(g, columns=["input"]).column(0).to_pylist():
                if not txt:
                    continue
                body = txt.split("Ingredients:", 1)
                if len(body) < 2:
                    continue
                body = body[1].split("Directions:", 1)[0]
                yield [_QTY.sub("", ln).strip().lower()
                       for ln in body.splitlines() if ln.strip()]
                n += 1
                if limit and n >= limit:
                    return


# Chinese and Russian ingredient strings carry quantities the English regex
# cannot see: "2大片生菜" (2 large leaves lettuce), "半个紫洋葱" (half a red onion).
_ZH_QTY = re.compile(
    r"[0-9０-９.·/]+"
    r"|[适少半一二三四五六七八九十两]?"
    r"[大小]?[勺匙杯个只根片块条把颗"
    r"粒盒袋包碗斤克千毫升汤茶量许适]+"
    r"|\(.*?\)|（.*?）")


def _zh_matcher(zh_map):
    """Longest-first alternation. Chinese has no word boundaries, so a plain
    substring match is correct here -- but 低筋面粉 must beat 面粉, and
    生抽 (light soy) must beat 抽, or the counts silently collapse."""
    keys = sorted(zh_map, key=len, reverse=True)
    return re.compile("|".join(re.escape(k) for k in keys))


def iter_xiachufang(limit=None):
    """1.55M Chinese recipes -- 37.4% of the paper's corpus."""
    import json
    n = 0
    f = CORPUS / "02-xiachufang" / "recipe_corpus_full.json"
    if not f.exists():
        return
    with f.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            ing = d.get("recipeIngredient") or []
            if not ing:
                continue
            yield [_ZH_QTY.sub("", s).strip() for s in ing]
            n += 1
            if limit and n >= limit:
                return


def iter_povarenok(limit=None):
    """147k Russian recipes. Ingredients are already a normalised dict."""
    import ast
    csv.field_size_limit(1 << 30)
    n = 0
    f = CORPUS / "03-povarenok" / "povarenok.csv"
    if not f.exists():
        return
    with f.open(encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                d = ast.literal_eval(row["ingredients"])
            except Exception:
                continue
            if not d:
                continue
            yield [k.strip().lower() for k in d]
            n += 1
            if limit and n >= limit:
                return


def main(limit: int | None = None, source: str = "ner"):
    vocab, itos = load_vocab()
    alias = build_alias(vocab)
    n = len(vocab)
    multi = source in ("allrecipes", "all")
    pat = build_matcher(alias) if multi else None
    zh_map = ru_map = zh_pat = None
    if source == "all":
        from multilingual import build_maps
        maps, _ = build_maps(vocab)
        zh_map, ru_map = maps["zh"], maps["ru"]
        zh_pat = _zh_matcher(zh_map)

    uni = np.zeros(n, np.int64)
    co = Counter()
    n_recipes = matched_recipes = 0
    tokens_total = tokens_matched = 0

    def records():
        if source == "all":
            # tag each record with its language so the right matcher runs
            for lines in iter_allrecipes(limit):
                yield "en", lines
            for lines in iter_xiachufang(limit):
                yield "zh", lines
            for lines in iter_povarenok(limit):
                yield "ru", lines
        elif source == "allrecipes":
            for lines in iter_allrecipes(limit):
                yield "en", lines
        else:
            with SRC.open(newline="", encoding="utf-8", errors="replace") as fh:
                for row in csv.DictReader(fh):
                    if row.get("NER"):
                        yield "en", [x.strip().strip('"[]\' ').lower()
                                     for x in row["NER"].split(",")]

    per_lang = Counter()
    for lang, toks in records():
            n_recipes += 1
            per_lang[lang] += 1
            ids = set()
            for t in toks:
                if not t:
                    continue
                tokens_total += 1
                if lang == "zh":
                    hits = {vocab[zh_map[h.group(0)]]
                            for h in zh_pat.finditer(t)}
                    if hits:
                        tokens_matched += 1
                        ids |= hits
                    continue
                if lang == "ru":
                    name = ru_map.get(t)
                    if name is None:                 # "лук репчатый крупный"
                        for k in sorted(ru_map, key=len, reverse=True):
                            if t.startswith(k):
                                name = ru_map[k]
                                break
                    if name is not None:
                        tokens_matched += 1
                        ids.add(vocab[name])
                    continue
                if pat is not None:
                    # every non-overlapping match, so "salt and pepper" and
                    # "butter or margarine" both contribute two ingredients
                    hits = {alias[h.group(1)] for h in pat.finditer(t)}
                    if hits:
                        tokens_matched += 1
                        ids |= hits
                    continue
                i = lookup(alias, t)
                if i is not None:
                    tokens_matched += 1
                    ids.add(i)
            if len(ids) >= 2:
                matched_recipes += 1
                for i in ids:
                    uni[i] += 1
                for a, b in combinations(sorted(ids), 2):
                    co[(a, b)] += 1
            if limit and n_recipes >= limit:
                break
            if n_recipes % 250_000 == 0:
                print(f"  {n_recipes:,} recipes, {len(co):,} distinct pairs",
                      flush=True)

    print(f"\nrecipes read      {n_recipes:,}")
    for lg, c_ in per_lang.most_common():
        print(f"    {lg}: {c_:,}")
    print(f"usable (>=2 hits) {matched_recipes:,} "
          f"({100 * matched_recipes / max(n_recipes, 1):.1f}%)")
    print(f"NER tokens        {tokens_total:,}, matched to vocab "
          f"{tokens_matched:,} ({100 * tokens_matched / max(tokens_total, 1):.1f}%)")
    print(f"distinct pairs    {len(co):,}")
    print(f"vocab covered     {int((uni > 0).sum()):,}/{n}")

    # PMI over the recipe population
    tot = matched_recipes
    pairs = np.array(list(co.keys()), np.int32)
    cnt = np.array(list(co.values()), np.int64)
    pi = uni[pairs[:, 0]] / tot
    pj = uni[pairs[:, 1]] / tot
    pij = cnt / tot
    pmi = np.log(pij / (pi * pj))
    npmi = pmi / -np.log(pij)

    np.savez_compressed(
        OUT / "recipe_cooc.npz", pairs=pairs, count=cnt, pmi=pmi.astype(np.float32),
        npmi=npmi.astype(np.float32), uni=uni, n_recipes=tot,
        itos=np.array(itos, dtype=object))
    print(f"\nwrote {(OUT / 'recipe_cooc.npz').relative_to(ROOT)}")


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "-" else None
    src = sys.argv[2] if len(sys.argv) > 2 else "ner"
    main(lim, src)
