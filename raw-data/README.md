# `raw-data/` — Epicure training corpus, by source

This tracked file records source provenance. The dataset bytes are not
redistributed with the public repository. Counts below describe the original
paper-reproduction audit and expansion; the current normalized corpus is
identified by [`model/data/GENERATION.json`](../model/data/GENERATION.json).

Raw recipe corpora backing **Table A1** of *Epicure: Navigating the Emergent Geometry of
Food Ingredient Embeddings* ([arXiv:2605.22391](https://arxiv.org/abs/2605.22391)).
One folder per Table A1 row, numbered in the paper's order (descending recipe count).

**5.4 GB · 4,137,626 recipes · 9 source groups / 11 corpora · 7 languages.**

Recount at any time:

```bash
model/.venv/bin/python prior-study/tools/verify_corpus_table_a1.py     # ~6 min
```

> **The numbered `01-…09-` folders are frozen.** They are the verified 1:1 reproduction of
> Table A1 and must not be added to — otherwise the reproduction claim stops being checkable.
> Corpora beyond the paper live in numbered folders `10-…29-` (documented in [`_expansion_README.md`](_expansion_README.md)), which add
> **177,718 net-new deduplicated recipes** and 8 new languages
> (combined unique total: **4,315,344**).

## Layout

```
raw-data/
  01-recipenlg/ … 09-chefkoch/   Table A1 baseline — frozen, verified
  10-…/ … 28-…/                  19 corpora beyond the paper (+177,718 net new)
  29-ingredient-substitutions/   a substitution table, not a recipe corpus
  _duplicates/                   verified-redundant sets, kept for provenance
  _expansion_README.md           documentation for the 10-29 expansion corpora
  _superseded/                   corbt/all-recipes mirror — NOT Table A1
```

The expansion corpora were previously in an `expansion/` subdirectory, and this whole tree was previously called `recipe/`; both were flattened and renamed in 08/2026. See `prior-study/_layout_migration.json` for the exact old→new mapping if you encounter old path references elsewhere — note that its entries still use the old `recipe/` root.

Results are written to `prior-study/data/derived/table_a1_verification.json`.

---

## Reproduction status

| # | Source | Lang | Paper | Ours | Delta | Status |
|---|---|---|---:|---:|---:|---|
| 01 | RecipeNLG | en | 2,230,569 | 2,230,569 | +0 | **EXACT** |
| 02 | XiaChuFang | zh | 1,548,405 | 1,550,095 | +1,690 | +0.11% |
| 03 | Povarenok | ru | 146,564 | 146,564 | +0 | **EXACT** |
| 04 | Spanish | es | 75,680 | 75,680 | +0 | **EXACT** |
| 05 | Vietnamese | vi | 64,454 | 64,274 | −180 | −0.28% |
| 06 | Turkish | tr | 25,496 | 25,496 | +0 | **EXACT** |
| 07 | Indian | en | 16,190 | 17,117 | +927 | +5.73% |
| 08 | Indonesian | id | 15,641 | 15,641 | +0 | **EXACT** |
| 09 | Chefkoch | de | 12,190 | 12,190 | +0 | **EXACT** |
| | **TOTAL** | | **4,135,189** | **4,137,626** | **+2,437** | **0.059%** |

**6 of 9 sources reproduce exactly. The total is within 0.059% of the paper.**

### The counting rule

Recovered empirically, not documented in the paper:

> A record counts as a recipe when it carries a **non-empty ingredient list**.

This reproduces two sources to the row, which is what makes it credible rather
than a guess:

- RecipeNLG — 2,231,142 raw − **573** empty `NER` = 2,230,569 ✓
- Povarenok — 146,582 raw − **18** empty ingredient dicts = 146,564 ✓

---

## Sources

### 01 · RecipeNLG — en — 2,230,569
`RecipeNLG_dataset.csv` (2.1 GB) · Bień et al., 2020

Columns `title, ingredients, directions, link, source, NER`. The `NER` column is the
pre-extracted ingredient list the paper's canonicalisation consumes.

> Obtained from HF `SandhyaKilari/RecipeNLG_dataset`, which is a byte-faithful copy of
> the official gated `full_dataset.csv` (2,294,981,083 bytes; 2,231,142 rows).
> `mbien/recipe_nlg` on HF ships only a loader script with no data.

### 02 · XiaChuFang — zh — 1,548,405 (ours 1,550,095)
`recipe_corpus_full.json` (1.9 GB) · Liu et al., 2022 · HF `xzm1999/XiaChuFang_Recipe_Corpus`

JSONL, one recipe per line. Keys: `name, dish, description, recipeIngredient,
recipeInstructions, author, keywords`. 1,550,151 lines, only 56 with no ingredients.

### 03 · Povarenok — ru — 146,564
`povarenok.csv` (56 MB) · HF `rogozinushka/povarenok-recipes`

Columns `url, name, ingredients`; `ingredients` is a Python-literal dict of
`{name: quantity}`. 146,582 rows − 18 empty dicts.

### 04 · Spanish — es — 75,680
Three corpora that sum exactly:

| Folder | Origin | Recipes |
|---|---|---:|
| `somosnlp-recetas-cocina/dataset.csv` | HF `somosnlp/recetas-cocina` | 28,238 |
| `frorozcol-recetas-cocina/data/{train,valid,test}.csv` | HF `Frorozcol/recetas-cocina` | 27,206 |
| `somosnlp-recetasdelaabuela/main.csv` | HF `somosnlp/RecetasDeLaAbuela` | 20,236 |

> The Abuela repo ships **two** tables. `main.csv` (20,236) is the one that makes the
> group total 75,680; `recetasdelaabuela.csv` (20,085) does not. Kept both.

### 05 · Vietnamese — vi — 64,454 (ours 64,274)
`cooking_multimodal_local_fixed_v1.json` (268 MB) · Nguyen, 2024 · HF `anhnq1130/cooking`

Not a recipe table — a multimodal SFT file of 192,822 chat turns, structured as
**2 identical halves × 3 prompt variants × 32,137 dish images**. The ingredient-bearing
variant (`"Để làm món X, bạn cần chuẩn bị:"`) is the recipe-shaped record: 192,822 / 3 = 64,274.

### 06 · Turkish — tr — 25,496 ✓
`turkish_recipe_v3.parquet` (11 MB) · Al, 2023 · HF `SedatAl/Turkish_Recipe_v3`

Matches Table A1 exactly at raw row count, no filtering needed.

### 07 · Indian — en — 16,190 (ours 17,117)
Three corpora:

| Folder | Origin | Recipes |
|---|---|---:|
| `jain-mendeley-xsphgmmh7b/` | Mendeley `xsphgmmh7b/1` (Jain, 2020) | 6,865 |
| `singh-indian-food-101/` | Kaggle `nehaprabhavalkar/indian-food-101` (Singh, 2019) | 255 |
| `ahsan-10k-south-asian/` | Kaggle `ahsanneural/10k-south-asian-...` (Ahsan, 2022) | 9,997 |

### 08 · Indonesian — id — 15,641 ✓
8 CSVs by protein (`ayam, ikan, kambing, sapi, tahu, telur, tempe, udang`) · Dzikri, 2020 ·
Kaggle `canggih/indonesian-food-recipes`. Sums to 15,641 exactly.

### 09 · Chefkoch — de — 12,190 ✓
`recipes.json` (16 MB) · Sterby, 2021 · Kaggle `sterby/german-recipes-dataset`

Keys `Url, Instructions, Ingredients, Day, Name, Year, Month, Weekday`. Exactly 12,190
records, zero empty. Independently corroborated: the Kaggle dataset is described in
third-party docs as "12190 german recipes... crawled from chefkoch.de".

---

## Provenance notes

Kaggle's API requires authentication, so the three Kaggle-origin corpora (08, 09, and
Singh/Ahsan in 07) were sourced from public mirrors. **Two of them land on the paper's
exact count** (Indonesian 15,641; Chefkoch 12,190), which is strong evidence the mirrors
are faithful — a truncated copy would not hit the published number precisely.
Jain (07) came directly from the Mendeley public API, the source the paper cites.

## Open deltas

| Source | Delta | Why it is unresolved |
|---|---|---|
| XiaChuFang | +1,690 | Only 56 records lack ingredients. Dedup on `(name, ingredients)` overshoots to 1,542,420; no filter lands on 1,548,405. |
| Vietnamese | −180 | Repo has only ever held one file (both commits checked). The extra 180 is not recoverable from this source. |
| Indian | +927 | Ahsan is fully covered (9,997 across ingredients/steps/nutrition), so the drop must be in Jain — but no cuisine, ASCII, or dedup filter yields 5,938. |

All three are structural (a duplicated SFT file, an undocumented dedup step), not
access problems. Adding Kaggle credentials would not close any of them.

## `_superseded/`

`corbt-all-recipes-NOT-table-a1/` — a 2,147,248-row derived RecipeNLG mirror
(`ar_*.parquet`, single `input` column) that was on disk before this audit. It is **not**
a Table A1 source and is excluded from all counts. Superseded by `01-recipenlg/`.
Safe to delete (801 MB).
