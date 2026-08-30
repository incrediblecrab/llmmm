"""Recount every raw recipe corpus on disk and compare against Table A1 of the
Epicure supplement (arXiv:2605.22391), which reports 4,135,189 recipes.

Paper-implied inclusion rule, recovered empirically: a record counts as a recipe
when it carries a non-empty ingredient list. That rule reproduces RecipeNLG and
Povarenok to the row.

Run:  .venv/bin/python tools/verify_corpus_table_a1.py
"""

import ast
import collections
import csv
import json
import os
import re
import sys

csv.field_size_limit(10**9)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# One directory per Table A1 row, numbered in the paper's order. The corpus is
# not duplicated into prior-study/; it lives once at the repository root.
RAW = os.environ.get("IM_CORPUS") or os.path.join(
    os.path.dirname(ROOT), "raw-data")
DERIVED = os.path.join(ROOT, "data", "derived")

# Table A1, verbatim.
TARGETS = [
    ("RecipeNLG", "en", 2230569),
    ("XiaChuFang", "zh", 1548405),
    ("Povarenok", "ru", 146564),
    ("Spanish", "es", 75680),
    ("Vietnamese", "vi", 64454),
    ("Turkish", "tr", 25496),
    ("Indian", "en", 16190),
    ("Indonesian", "id", 15641),
    ("Chefkoch", "de", 12190),
]
TOTAL_TARGET = 4135189


def _dictrows(path, encoding="utf-8-sig"):
    with open(path, newline="", encoding=encoding, errors="replace") as fh:
        yield from csv.DictReader(fh)


def _nonempty(value):
    return bool((value or "").strip()) and (value or "").strip() != "-1"


def count_recipenlg():
    """full_dataset.csv; drop rows whose NER (extracted ingredient) list is empty."""
    path = os.path.join(RAW, "01-recipenlg", "RecipeNLG_dataset.csv")
    total = kept = 0
    for row in _dictrows(path, encoding="utf-8"):
        total += 1
        try:
            ner = json.loads(row.get("NER") or "[]")
        except ValueError:
            ner = []
        if ner:
            kept += 1
    return kept, {"raw_rows": total, "dropped_empty_NER": total - kept}


def count_xiachufang():
    """recipe_corpus_full.json (JSONL); drop records with no recipeIngredient."""
    path = os.path.join(RAW, "02-xiachufang", "recipe_corpus_full.json")
    total = kept = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total += 1
            if (json.loads(line).get("recipeIngredient") or []):
                kept += 1
    return kept, {"raw_lines": total, "dropped_empty_ingredients": total - kept}


def count_povarenok():
    """povarenok.csv; ingredients is a python-literal dict, drop empties."""
    path = os.path.join(RAW, "03-povarenok", "povarenok.csv")
    total = kept = 0
    for row in _dictrows(path, encoding="utf-8"):
        total += 1
        raw = (row.get("ingredients") or "").strip()
        try:
            parsed = ast.literal_eval(raw) if raw else {}
        except (ValueError, SyntaxError):
            parsed = {}
        if parsed:
            kept += 1
    return kept, {"raw_rows": total, "dropped_empty_ingredients": total - kept}


def count_spanish():
    """Three corpora: somosnlp/recetas-cocina + Frorozcol/recetas-cocina (all
    splits) + somosnlp/RecetasDeLaAbuela (main.csv)."""
    parts = collections.OrderedDict()
    base = os.path.join(RAW, "04-spanish")
    parts["somosnlp_recetas-cocina"] = sum(
        1 for _ in _dictrows(os.path.join(base, "somosnlp-recetas-cocina", "dataset.csv"), "utf-8")
    )
    frorozco = 0
    for split in ("train", "valid", "test"):
        frorozco += sum(
            1 for _ in _dictrows(
                os.path.join(base, "frorozcol-recetas-cocina", "data", f"{split}.csv"), "utf-8")
        )
    parts["Frorozcol_recetas-cocina"] = frorozco
    parts["somosnlp_RecetasDeLaAbuela"] = sum(
        1 for _ in _dictrows(os.path.join(base, "somosnlp-recetasdelaabuela", "main.csv"), "utf-8")
    )
    return sum(parts.values()), dict(parts)


def count_vietnamese():
    """anhnq1130/cooking: a multimodal SFT file, not a recipe table. It holds
    2 identical halves x 3 prompt variants x 32,137 dish images. The
    ingredient-bearing variant ("De lam mon X, ban can chuan bi:") is the
    recipe-shaped record, giving len(file)/3."""
    path = os.path.join(RAW, "05-vietnamese", "cooking_multimodal_local_fixed_v1.json")
    data = json.load(open(path, encoding="utf-8"))
    images, ingredient_records = set(), 0
    for rec in data:
        imgs = rec.get("images") or []
        if imgs:
            images.add(os.path.basename(imgs[0]))
        for msg in rec["messages"]:
            if msg["role"] == "assistant":
                if msg["content"].startswith("Để làm món"):
                    ingredient_records += 1
                break
    return ingredient_records, {
        "conversation_entries": len(data),
        "distinct_dish_images": len(images),
        "prompt_variants_per_dish": len(data) // max(len(images), 1),
    }


def count_turkish():
    import pyarrow.parquet as pq

    path = os.path.join(RAW, "06-turkish", "turkish_recipe_v3.parquet")
    n = pq.ParquetFile(path).metadata.num_rows
    return n, {"raw_rows": n}


def count_indian():
    """Three corpora: Jain (Mendeley) + Singh (indian-food-101) + Ahsan (10k South Asian)."""
    parts = collections.OrderedDict()
    base = os.path.join(RAW, "07-indian")
    jain = [
        r for r in _dictrows(os.path.join(base, "jain-mendeley-xsphgmmh7b", "IndianFoodDatasetCSV.csv"))
        if _nonempty(r.get("TranslatedIngredients"))
    ]
    parts["Jain_mendeley_xsphgmmh7b"] = len(jain)
    parts["Singh_indian-food-101"] = sum(
        1 for _ in _dictrows(os.path.join(base, "singh-indian-food-101", "indian_food.csv"))
    )
    parts["Ahsan_10k-south-asian"] = sum(
        1 for _ in _dictrows(os.path.join(base, "ahsan-10k-south-asian", "recipes_master.csv"))
    )
    return sum(parts.values()), dict(parts)


def count_indonesian():
    parts = collections.OrderedDict()
    folder = os.path.join(RAW, "08-indonesian")
    for name in sorted(os.listdir(folder)):
        if name.endswith(".csv"):
            parts[name] = sum(1 for _ in _dictrows(os.path.join(folder, name), "utf-8"))
    return sum(parts.values()), dict(parts)


def count_chefkoch():
    path = os.path.join(RAW, "09-chefkoch", "recipes.json")
    data = json.load(open(path, encoding="utf-8"))
    kept = sum(1 for r in data if r.get("Ingredients"))
    return kept, {"raw_records": len(data), "dropped_empty_ingredients": len(data) - kept}


COUNTERS = {
    "RecipeNLG": count_recipenlg,
    "XiaChuFang": count_xiachufang,
    "Povarenok": count_povarenok,
    "Spanish": count_spanish,
    "Vietnamese": count_vietnamese,
    "Turkish": count_turkish,
    "Indian": count_indian,
    "Indonesian": count_indonesian,
    "Chefkoch": count_chefkoch,
}


def main():
    results, breakdowns = {}, {}
    for source, _, _ in TARGETS:
        sys.stderr.write(f"counting {source} ...\n")
        sys.stderr.flush()
        results[source], breakdowns[source] = COUNTERS[source]()

    print()
    print("Table A1 reproduction — Epicure (arXiv:2605.22391)")
    print("=" * 78)
    print(f"{'Source':<14}{'Lang':<6}{'Paper':>12}{'Ours':>12}{'Delta':>10}{'Status':>12}")
    print("-" * 78)
    exact = 0
    ours_total = 0
    for source, lang, target in TARGETS:
        got = results[source]
        ours_total += got
        delta = got - target
        if delta == 0:
            exact += 1
            status = "EXACT"
        else:
            status = f"{delta / target * 100:+.2f}%"
        print(f"{source:<14}{lang:<6}{target:>12,}{got:>12,}{delta:>+10,}{status:>12}")
    print("-" * 78)
    d = ours_total - TOTAL_TARGET
    print(f"{'TOTAL':<20}{TOTAL_TARGET:>12,}{ours_total:>12,}{d:>+10,}{d / TOTAL_TARGET * 100:>11.3f}%")
    print("=" * 78)
    print(f"Exact source matches: {exact}/{len(TARGETS)}")
    print()
    for source, _, _ in TARGETS:
        print(f"{source}: {breakdowns[source]}")

    out = os.path.join(DERIVED, "table_a1_verification.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "paper_total": TOTAL_TARGET,
                "our_total": ours_total,
                "exact_matches": exact,
                "per_source": [
                    {"source": s, "lang": l, "paper": t, "ours": results[s], "delta": results[s] - t,
                     "breakdown": breakdowns[s]}
                    for s, l, t in TARGETS
                ],
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
