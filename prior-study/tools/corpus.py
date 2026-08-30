#!/usr/bin/env python3
"""Unified reader over every recipe corpus we hold.

`tools/build_recipe_cooc.py` only ever wired three sources (RecipeNLG,
XiaChuFang, Povarenok) and hard-coded their quirks inside `main()`. That covered
92% of recipes but 0% of Spanish, Vietnamese, Turkish, Indian, Indonesian and
German -- and none of the 19 expansion corpora.

This module puts every corpus behind one registry so the graph builder can ask
for "all recipes" and get back `(source, lang, [ingredient strings])` without
caring how any individual file is shaped.

Ingredient *strings* are returned raw (minus obvious quantity noise); mapping
them onto the 1,790-term canonical vocabulary is `normalize.py`'s job.
"""
from __future__ import annotations

import ast
import csv
import glob
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: The corpus is not duplicated into ``prior-study/``. It lives once, at the
#: repository root, and this points at it. ``IM_CORPUS`` overrides for a
#: corpus held outside the repository.
BASE = Path(os.environ.get("IM_CORPUS") or ROOT.parent / "raw-data")
EXP = BASE  # expansion sources were flattened into recipe/ as NN-name

csv.field_size_limit(1 << 30)

# --------------------------------------------------------------------------
# splitters
# --------------------------------------------------------------------------

def _lines(s):
    return [x for x in str(s).splitlines() if x.strip()]

def _comma(s):
    return [x for x in str(s).split(",") if x.strip()]

def _dashdash(s):
    return [x for x in str(s).split("--") if x.strip()]

def _pylist(s):
    """'[a, b]' / "c('a','b')" / real list / list-of-dicts -> list[str]."""
    if isinstance(s, (list, tuple)):
        return [_name_of(x) for x in s]
    if hasattr(s, "tolist"):
        return [_name_of(x) for x in s.tolist()]
    t = str(s).strip()
    if t.startswith("c(") and t.endswith(")"):
        return re.findall(r'"([^"]*)"', t)
    try:
        v = ast.literal_eval(t)
        if isinstance(v, (list, tuple)):
            return [_name_of(x) for x in v]
        if isinstance(v, dict):
            return [_name_of(v)]
    except Exception:
        pass
    return _multi(t)


def _name_of(x):
    """Rows often ship {'name': 'Батон', 'count': '1 бан.'} instead of a string."""
    if isinstance(x, dict):
        return str(x.get("name") or x.get("ingredient") or x.get("title") or "")
    t = str(x).strip()
    if t.startswith("{") and "name" in t:
        try:
            d = ast.literal_eval(t)
            if isinstance(d, dict):
                return str(d.get("name") or "")
        except Exception:
            pass
    return t


_SEPS = re.compile(r"\n|\s;\s|\s\|\s|●|\u2022|--")

def _multi(s):
    """Split on whichever of newline / ; / | / bullet / -- the source uses."""
    return [x.strip() for x in _SEPS.split(str(s)) if x and x.strip()]


def _recipe_text(s):
    """thefoodprocessor: title line, ingredient lines, then 'Instructions:'."""
    t = str(s).split("Instructions:")[0]
    parts = [x.strip() for x in t.splitlines() if x.strip()]
    return parts[1:] if len(parts) > 1 else parts


def _ja(s):
    return [x.strip() for x in re.split(r"[、,\n]", str(s)) if x.strip()]


def _jsonlist(s):
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [x if isinstance(x, str) else
                    (x.get("name") or x.get("ingredient") or "") for x in v]
    except Exception:
        pass
    return _pylist(s)

_HTML_TH = re.compile(r"<th>(.*?)</th>", re.S)

def _html_table(s):
    return _HTML_TH.findall(str(s))

# --------------------------------------------------------------------------
# generic file readers
# --------------------------------------------------------------------------

def _read_csv_col(path, col, split, lang, key, limit=None):
    import pandas as pd
    n = 0
    for chunk in pd.read_csv(path, chunksize=100_000, engine="c",
                             on_bad_lines="skip", dtype=str):
        if col not in chunk.columns:
            return
        for v in chunk[col]:
            if v is None or v != v:
                continue
            items = split(v)
            if items:
                yield key, lang, items
                n += 1
                if limit and n >= limit:
                    return


def _read_parquet_col(pattern, col, split, lang, key, limit=None):
    import pandas as pd
    n = 0
    for f in sorted(glob.glob(str(pattern))):
        df = pd.read_parquet(f, columns=[col]) if col else pd.read_parquet(f)
        for v in df[col]:
            if v is None or (isinstance(v, float) and v != v):
                continue
            items = split(v)
            if items:
                yield key, lang, items
                n += 1
                if limit and n >= limit:
                    return


# --------------------------------------------------------------------------
# source-specific readers
# --------------------------------------------------------------------------

def src_recipenlg(limit=None):
    """2.23M English. NER = the corpus's own pre-extracted ingredient entities."""
    f = BASE / "01-recipenlg" / "RecipeNLG_dataset.csv"
    n = 0
    with f.open(newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            ner = row.get("NER")
            if not ner:
                continue
            items = [x.strip().strip('"[]\' ').lower() for x in ner.split(",")]
            items = [x for x in items if x]
            if items:
                yield "01-recipenlg", "en", items
                n += 1
                if limit and n >= limit:
                    return


_ZH_QTY = re.compile(
    r"[0-9０-９.·/]+"
    r"|[适少半一二三四五六七八九十两]?"
    r"[大小]?[勺匙杯个只根片块条把颗粒盒袋包碗斤克千毫升汤茶量许适]+"
    r"|\(.*?\)|（.*?）")


def src_xiachufang(limit=None):
    """1.55M Chinese, JSONL."""
    f = BASE / "02-xiachufang" / "recipe_corpus_full.json"
    n = 0
    with f.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            ing = d.get("recipeIngredient") or []
            if not ing:
                continue
            items = [_ZH_QTY.sub("", s).strip() for s in ing]
            items = [x for x in items if x]
            if items:
                yield "02-xiachufang", "zh", items
                n += 1
                if limit and n >= limit:
                    return


def src_povarenok(limit=None):
    """147k Russian. `ingredients` is a dict literal {name: qty}."""
    f = BASE / "03-povarenok" / "povarenok.csv"
    n = 0
    with f.open(encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                d = ast.literal_eval(row["ingredients"])
            except Exception:
                continue
            if not d:
                continue
            yield "03-povarenok", "ru", [k.strip().lower() for k in d]
            n += 1
            if limit and n >= limit:
                return


def src_spanish(limit=None):
    d = BASE / "04-spanish"
    yield from _read_csv_col(d / "somosnlp-recetas-cocina" / "dataset.csv",
                             "ingredients", _lines, "es", "04-spanish", limit)
    for part in ("train", "valid", "test"):
        p = d / "frorozcol-recetas-cocina" / "data" / f"{part}.csv"
        if p.exists():
            yield from _read_csv_col(p, "ingredients", _lines, "es",
                                     "04-spanish", limit)
    yield from _read_csv_col(d / "somosnlp-recetasdelaabuela" / "main.csv",
                             "Ingredientes", _comma, "es", "04-spanish", limit)


_VI_ING = re.compile(r"(?:nguyên liệu|bạn cần chuẩn bị)\s*:?\s*(.+?)(?:\n\n|\Z)",
                     re.I | re.S)


def src_vietnamese(limit=None):
    """64k Vietnamese hiding inside a multimodal SFT chat file."""
    f = BASE / "05-vietnamese" / "cooking_multimodal_local_fixed_v1.json"
    data = json.load(f.open(encoding="utf-8"))
    n = 0
    seen = set()
    for rec in data:
        msgs = rec.get("messages") or []
        txt = "".join(m.get("content", "") for m in msgs
                      if m.get("role") == "assistant")
        if not txt:
            continue
        m = _VI_ING.search(txt)
        if not m:
            continue
        body = m.group(1)
        items = [re.sub(r"^[-*\u2022\d.\s]+", "", x).strip()
                 for x in body.splitlines() if x.strip()]
        items = [x for x in items if 1 < len(x) < 80]
        if len(items) < 2:
            continue
        key = txt[:60]
        if key in seen:                      # file is 2 identical halves x3 prompts
            continue
        seen.add(key)
        yield "05-vietnamese", "vi", items
        n += 1
        if limit and n >= limit:
            return


def src_turkish(limit=None):
    yield from _read_parquet_col(BASE / "06-turkish" / "turkish_recipe_v3.parquet",
                                 "malzemeler", _comma, "tr", "06-turkish", limit)


def src_indian(limit=None):
    d = BASE / "07-indian"
    yield from _read_csv_col(d / "jain-mendeley-xsphgmmh7b" / "IndianFoodDatasetCSV.csv",
                             "TranslatedIngredients", _comma, "en", "07-indian", limit)
    yield from _read_csv_col(d / "singh-indian-food-101" / "indian_food.csv",
                             "ingredients", _comma, "en", "07-indian", limit)
    p = d / "ahsan-10k-south-asian" / "recipe_ingredients.csv"
    if p.exists():
        import pandas as pd
        df = pd.read_csv(p, dtype=str, on_bad_lines="skip")
        # Prefer an explicit *_name column: this file also has "ingredient_number",
        # which matches a naive "ingredient in colname" test and yields the
        # digits 1,2,3... instead of food -- a 17% match rate instead of 99%.
        icol = next((c for c in df.columns if "ingredient" in c.lower()
                     and "name" in c.lower()), None)
        if icol is None:
            icol = next((c for c in df.columns if "ingredient" in c.lower()
                         and not re.search(r"id|number|no|idx|qty|quantity",
                                           c.lower())), None)
        idcol = next((c for c in df.columns if c.lower().endswith("recipe_id")
                      or c.lower() == "recipe_id"), None)
        if icol and idcol:
            for _, g in df.groupby(idcol)[icol]:
                items = [str(x) for x in g if str(x) != "nan"]
                if items:
                    yield "07-indian", "en", items


def src_indonesian(limit=None):
    for f in sorted(glob.glob(str(BASE / "08-indonesian" / "*.csv"))):
        yield from _read_csv_col(f, "Ingredients", _dashdash, "id",
                                 "08-indonesian", limit)


def src_chefkoch(limit=None):
    f = BASE / "09-chefkoch" / "recipes.json"
    data = json.load(f.open(encoding="utf-8"))
    n = 0
    for r in data:
        items = r.get("Ingredients") or []
        if not items:
            continue
        yield "09-chefkoch", "de", [str(x) for x in items]
        n += 1
        if limit and n >= limit:
            return


# --------------------------------------------------------------------------
# expansion registry: (key, lang, kind, path, column, splitter)
# --------------------------------------------------------------------------

EXPANSION = [
    ("foodcom-522k", "en", "csv", "10-foodcom-canonical/recipes.csv",
     "RecipeIngredientParts", _pylist),
    ("foodcom-raw-231k", "en", "parquet", "11-foodcom-raw/food_recipes.parquet",
     "ingredients", _pylist),
    ("povarenok-detail", "ru", "parquet", "12-povarenok-detail/data/*.parquet",
     "ingredients", _pylist),
    ("turkish-102k", "tr", "parquet", "13-turkish/data/*.parquet",
     "Malzemeler", _multi),
    ("thefoodprocessor-74k", "en", "parquet", "14-thefoodprocessor/data/*.parquet",
     "recipe", _recipe_text),
    ("allrecipes-33k", "en", "csv", "15-allrecipes/recipe.csv",
     "ingredients", _pylist),
    ("bhuvii-17k", "en", "jsonl", "16-bhuvii/pretokenization.json",
     "Ingredients", _pylist),
    ("kaggle-food-13k", "en", "csv",
     "17-kaggle-food/Food Ingredients and Recipe Dataset with Image Name Mapping.csv",
     "Cleaned_Ingredients", _pylist),
    ("hebrew-9.7k", "he", "hebrew", "18-hebrew/recipes.parquet", None, None),
    ("indian-7k", "en", "csv", "19-indian/Food_Recipe.csv",
     "ingredients_name", _comma),
    ("persian-6k", "fa", "csv", "20-persian/train.csv", "table", _html_table),
    ("greek-5k", "el", "csv", "21-greek/recipes_greek.csv", "Ingredients", _comma),
    ("moroccan-4.6k", "en", "moroccan", "22-moroccan/moroccan_recipes_dataset.csv",
     "prompt", None),
    ("japanese-3k", "ja", "csv", "23-japanese/recipe_jap.csv", "材料", _ja),
    ("filipino-2k", "fil", "parquet", "24-filipino/data/*.parquet",
     "ingredients", _pylist),
    ("halal-2k", "en", "jsonl", "25-halal/data/recipes.jsonl", "ingredients", _pylist),
    ("thai-1k", "th", "csv", "26-thai/thai_food_dataset.csv",
     "วัตถุดิบ (Ingredients)", _comma),
    ("thai-1k-seasoning", "th", "csv", "26-thai/thai_food_dataset.csv",
     "เครื่องปรุง (Seasonings)", _comma),
    ("taiwan-1.8k", "zh", "parquet", "27-taiwan/data/*.parquet",
     "ingredients", _pylist),
    ("romanian-881", "ro", "csv", "28-romanian/train.csv", "1", _multi),
]

_MA_ING = re.compile(r"Ingredients\s*:\s*(\[.*?\])", re.S)


def _src_expansion(key, lang, kind, rel, col, split, limit=None):
    p = EXP / rel
    if kind == "csv":
        yield from _read_csv_col(p, col, split, lang, key, limit)
    elif kind == "parquet":
        yield from _read_parquet_col(p, col, split, lang, key, limit)
    elif kind == "jsonl":
        n = 0
        for line in p.open(encoding="utf-8"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            items = split(d.get(col))
            if items:
                yield key, lang, items
                n += 1
                if limit and n >= limit:
                    return
    elif kind == "json":
        data = json.load(p.open(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("data", [])
        for d in data:
            items = split(d.get(col))
            if items:
                yield key, lang, items
    elif kind == "hebrew":
        import pandas as pd
        df = pd.read_parquet(p, columns=["json_ld"])
        for s in df["json_ld"]:
            try:
                j = json.loads(s)
                if isinstance(j, list):
                    j = j[0]
                g = j.get("@graph")
                if g:
                    j = next((x for x in g if x.get("@type") == "Recipe"), j)
                items = j.get("recipeIngredient") or []
            except Exception:
                continue
            if items:
                yield key, lang, [str(x) for x in items]
    elif kind == "moroccan":
        import pandas as pd
        df = pd.read_csv(p, dtype=str, on_bad_lines="skip")
        for v in df[col].dropna():
            m = _MA_ING.search(str(v))
            if not m:
                continue
            items = _pylist(m.group(1))
            if items:
                yield key, lang, items


BASELINE = [
    ("01-recipenlg", src_recipenlg), ("02-xiachufang", src_xiachufang),
    ("03-povarenok", src_povarenok), ("04-spanish", src_spanish),
    ("05-vietnamese", src_vietnamese), ("06-turkish", src_turkish),
    ("07-indian", src_indian), ("08-indonesian", src_indonesian),
    ("09-chefkoch", src_chefkoch),
]


def iter_all(limit_per_source=None, only=None):
    """Yield (source_key, lang, [ingredient strings]) across every corpus."""
    for key, fn in BASELINE:
        if only and key not in only:
            continue
        yield from fn(limit_per_source)
    for key, lang, kind, rel, col, split in EXPANSION:
        if only and key not in only:
            continue
        try:
            yield from _src_expansion(key, lang, kind, rel, col, split,
                                      limit_per_source)
        except Exception as e:                       # never kill a 4M-row run
            print(f"  [warn] {key}: {type(e).__name__} {e}", flush=True)


ALL_KEYS = [k for k, _ in BASELINE] + [k for k, *_ in EXPANSION]


if __name__ == "__main__":
    import sys
    from collections import Counter
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    c, ex = Counter(), {}
    for key, lang, items in iter_all(lim):
        c[(key, lang)] += 1
        ex.setdefault(key, items[:3])
    print(f"{'source':24}{'lang':6}{'recipes':>10}   sample")
    for (k, lg), n in c.items():
        print(f"{k:24}{lg:6}{n:>10,}   {str(ex[k])[:60]}")
    print(f"\nsources yielding data: {len(c)}/{len(ALL_KEYS)}")
