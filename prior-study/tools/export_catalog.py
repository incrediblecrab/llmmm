#!/usr/bin/env python3
"""Export a single organised ingredient catalogue for upload to Azure.

The project's ingredient knowledge is currently scattered across four files in
three formats -- embeddings in safetensors, labels in JSON, graph statistics in
npz, and substitutions as free text inside a recipe parquet. Nothing joins them,
so there is no one place that answers "what do we know about garlic?".

This builds that join:

  ingredients.parquet   one row per vocabulary entry, with food group, corpus
                        frequency, graph degree and substitute availability
  substitutions.parquet the free-text `Ingredient: a, b, c` lines parsed into
                        (ingredient, alternative, votes) triples, with a flag
                        for whether each side resolves to the vocabulary
  catalog.json          manifest describing both, so the upload is documented

Substitution keys are matched against the vocabulary rather than trusted as-is,
because the source is recipe text and contains phrasings ("fried apples") that
are not ingredients at all.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pairing  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "catalog"
SUBS = (ROOT / "recipe" / "expansion" / "ingredient-substitutions-74k" /
        "data" / "train-00000-of-00001.parquet")


def build_substitutions(P) -> pd.DataFrame:
    df = pd.read_parquet(SUBS, columns=["ingredients_alternatives"])
    idx: dict[str, Counter] = {}
    for blob in df["ingredients_alternatives"].dropna():
        for line in str(blob).splitlines():
            if ":" not in line:
                continue
            head, tail = line.split(":", 1)
            key = head.strip().lower()
            if not key or len(key) > 40:
                continue
            alts = [a.strip().lower() for a in tail.split(",") if a.strip()]
            idx.setdefault(key, Counter()).update(a for a in alts if len(a) < 40)

    rows = []
    for ing, c in idx.items():
        rv = P.resolve(ing)
        for alt, n in c.items():
            rows.append((ing, alt, n, rv, P.resolve(alt)))
    return pd.DataFrame(rows, columns=["ingredient", "alternative", "votes",
                                       "ingredient_vocab", "alternative_vocab"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    P = pairing.Pairing()
    vocab = list(P.S["cooc"].vocab)

    z = np.load(ROOT / "data" / "derived" / "ii_graph.npz", allow_pickle=True)
    uni = z["uni"]
    deg = Counter(np.concatenate([z["src"], z["dst"]]).tolist())

    sub = build_substitutions(P)
    # only substitutions whose source term names a real vocabulary ingredient
    named = sub[sub.ingredient_vocab.notna()]
    by_vocab = named.groupby("ingredient_vocab")

    have = {k: v for k, v in by_vocab["votes"].sum().items()}
    top = {k: ", ".join(g.sort_values("votes", ascending=False)
                        .alternative.head(3))
           for k, g in by_vocab}

    ing = pd.DataFrame({
        "id": range(len(vocab)),
        "name": vocab,
        "food_group": [P.group.get(v, "Unknown") for v in vocab],
        "recipe_count": [int(uni[i]) for i in range(len(vocab))],
        "graph_degree": [int(deg.get(i, 0)) for i in range(len(vocab))],
        "substitute_votes": [int(have.get(v, 0)) for v in vocab],
        "top_substitutes": [top.get(v, "") for v in vocab],
    })

    ing.to_parquet(OUT / "ingredients.parquet", index=False)
    ing.to_csv(OUT / "ingredients.csv", index=False)
    sub.to_parquet(OUT / "substitutions.parquet", index=False)

    manifest = {
        "name": "Cookbook Regression ingredient catalog",
        "ingredients": {"rows": len(ing),
                        "food_groups": sorted(ing.food_group.unique().tolist()),
                        "with_substitutes": int((ing.substitute_votes > 0).sum())},
        "substitutions": {"rows": len(sub),
                          "distinct_ingredients": int(sub.ingredient.nunique()),
                          "resolved_to_vocab": int(sub.ingredient_vocab.notna().sum())},
        "graphs": {"ii_graph.npz": "ingredient-ingredient NPMI",
                   "flavor_graph.npz": "typed ingredient-compound"},
    }
    (OUT / "catalog.json").write_text(json.dumps(manifest, indent=2))

    print(json.dumps(manifest, indent=2))
    print(f"\ncoverage: {(ing.substitute_votes>0).sum()}/{len(ing)} vocabulary "
          f"ingredients have at least one substitute")
    print(ing.sort_values('recipe_count', ascending=False).head(8).to_string(index=False))


if __name__ == "__main__":
    main()
