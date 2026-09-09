# recipe/expansion/ — corpora beyond the Epicure paper

The numbered `recipe/01-…09-` folders are **frozen** as the verified 1:1 reproduction of supplement Table A1
(4,137,626 recipes, 6/9 sources exact). Nothing in this folder may be moved into it.

This folder holds corpora the paper did **not** use. Every source below was counted with the
same rule recovered from the paper (`tools/verify_corpus_table_a1.py`):

> a record counts as a recipe when it carries a non-empty ingredient list.

Net-new figures come from `tools/audit_expansion.py`, which normalises titles
(Unicode-aware `\w`, casefolded) and subtracts (a) titles already in `recipe/` and
(b) titles already claimed by an earlier row in this table.

## Recipe corpora

| # | source | lang | rows | w/ ingredients | net new | HF repo |
|---|--------|------|------|----------------|---------|---------|
| 1 | `foodcom-522k-canonical/` | en | 522,517 | 522,517 | **30,042** | `untitledwebsite123/food-recipes` |
| 2 | `foodcom-raw-231k/` | en | 231,637 | 231,637 | 1,210 | `jojogo9/Food_Recipes` |
| 3 | `povarenok-detail/` | ru | 154,158 | 154,155 | **8,276** | `d0rj/povarenok_recipes_detail` |
| 4 | `turkish-102k/` | tr | 101,993 | 101,811 | **71,422** | `AnIl-c/yemek_tarifleri` |
| 5 | `thefoodprocessor-74k/` | en | 74,465 | 74,465 | **19,997** | `Thefoodprocessor/recipes` |
| 6 | `allrecipes-33k/` | en | 32,722 | 32,722 | 3,557 | `arya123321/recipes` |
| 7 | `bhuvii-17k/` | en | 17,357 | 17,052 | 2,749 | `Bhuvii19/recipes` |
| 8 | `kaggle-food-13k/` | en | 13,501 | 13,501 | 1,047 | `Hieu-Pham/kaggle_food_recipes` |
| 9 | `hebrew-9.7k/` | **he** | 9,730 | 8,731 | **8,506** | `Wissotsky/HebrewRecipes` |
| 10 | `indian-7k/` | en-IN | 7,101 | 7,096 | **6,959** | `BhavaishKumar112/Food_Recipe` |
| 11 | `persian-6k/` | **fa** | 6,660 | 6,660 | **4,805** | `dadashzadeh/Persian_Cooking` |
| 12 | `greek-5k/` | **el** | 5,434 | 5,434 | **5,418** | `Depie/Recipes_Greek` |
| 13 | `moroccan-4.6k/` | **en-MA** | 4,627 | 4,627 | 2,589 | `idrisskh/Moroccan-Meal-Recipes-Dataset` |
| 14 | `japanese-3k/` | **ja** | 3,756 | 3,756 | 3,742 | `xiashuaxia/japanese_recipe` |
| 15 | `filipino-2k/` | **fil** | 1,988 | 1,988 | 1,827 | `joackimagno/FILIPINO_RECIPES_2K` |
| 16 | `halal-2k/` | en | 2,016 | 2,016 | 2,004 | `sdamoolp/optimal-recipes-halal` |
| 17 | `thai-1k/` | **th** | 1,023 | 1,023 | 1,022 | `ChokF/thai_food` |
| 18 | `taiwan-1.8k/` | zh-TW | 1,799 | 1,799 | 1,697 | `AWeirdDev/zh-tw-recipes-sm` |
| 19 | `romanian-881/` | **ro** | 881 | 881 | 849 | `BlackKakapo/recipes-ro` |
| | **TOTAL** | | **1,193,365** | **1,191,871** | **177,718** | |

Combined unique corpus: **4,137,626 + 177,718 = 4,315,344 recipes.**

### Language coverage
The paper covers 7 languages (en, zh, ru, vi, es, tr, id, de + Indian-English).
This expansion adds **8 new languages/locales**: Hebrew, Persian, Greek, Japanese,
Thai, Filipino, Romanian, and Moroccan/North-African — the first Sub-/North-African
coverage in the project, a region the paper **excluded entirely** (Table A2:
Sub-Saharan African had only 324 backing recipes, below the inclusion threshold).

Turkish is the single biggest gain: 25,496 → 96,918 (**+280%**).

## Non-recipe resource

`ingredient-substitutions-74k/` — `Thefoodprocessor/ingredients_alternatives`.
Not a recipe corpus (it re-uses the same 74,465 recipes as #5) but a **ground-truth
ingredient-substitution table**: 43,267 distinct source ingredients and
**218,354 substitution pairs** (`Parsley: cilantro, basil, dill`).
This is direct supervision/eval data for the app's "replace an ingredient" feature —
the only labelled substitution ground truth we hold. Do **not** count it as recipes.

## `_duplicates/` — verified redundant, retained for provenance

| dir | rows | net new | why rejected |
|-----|------|---------|--------------|
| `foodcom-490k/` | 490,457 | **0** | 100% contained in `foodcom-522k-canonical` (`Karo8870/food.com-parsed-dataset`). Kept only because its ingredients are pre-parsed to `{quantity, unit, name}` — useful as a parsing aid, never as a count. |
| `umarigan-82k/` | 82,303 | **0** | fully covered by RecipeNLG + food.com; 3.4 GB of image embeddings |
| `formido-20k/` | 20,000 | **0** | instruction-tuning reformat of RecipeNLG |
| `turkish-21k/` | 21,075 | 25 | subset of `turkish-102k` |
| `lebanese-37k/` | 36,800 | **6** | `WissMah/lebanese_aug_set`: 36,800 rows are image augmentations of only **9 unique dishes**. Clean ingredient lists, but no corpus value. |

## Rejected before download

- `AkashPS11`, `mdivanisova`, `Vatazh0k` `/recipes_data_food.com` — advertised 1,048,543 rows;
  actually an **Excel export padded to the 1,048,576-row limit with only 1,228 real records**.
  Verified by parsing: `distinct RecipeId = 1228`. Do not trust the datasets-server row count here.
- `CodeKapital`, `ahmedtra`, `Kaiser1308`, `skadewdl3`, `Mango008` `/CookingRecipes` etc. —
  byte-identical re-uploads of RecipeNLG (2,231,142).
- `Sultannn/id_recipe` (15,641), `SinclairSchneider/deutsche_rezepte` (12,190),
  `J555`/`Frorozcol`/`somosnlp` `recetas-cocina`, `rogozinushka/povarenok-recipes` (146,582) —
  identical to corpora already in `recipe/`.
- `Zappandy/recipe_nlg` (500k), `rk404/recipe_short` (350k), `KingName1/food.com` (87k),
  `tiagomosantos`/`OdinMeng` (496k) — strict subsets.
- `yemalin/african-food` (2,771) — VLM-generated **image captions**, not recipes
  ("visible ingredients include sliced okra…"). Hallucination risk; no ingredient list.
- `Tinsae/Ethiopian-foods` (1,097) — image + dish-name label only.
- `devkyle/ghanaian-food-dataset` (149 images), `infinite-dataset-hub/GhanaianEats`
  (91 synthetic nutrition rows) — too small / synthetic.
- `Dashyash/indian_cuisine_dataset` (15,039) — `image` column only.
- `tiptoghosh/food-recipes-15k` / `rahul7star/food-recipes` — 15.6 GB of images for
  15,698 food.com recipes already covered.

## Reproducing

```bash
.venv/bin/python tools/audit_expansion.py          # ~6 min, writes data/derived/expansion_audit.json
.venv/bin/python tools/verify_corpus_table_a1.py   # ~6 min, the frozen baseline
```

## Known gaps

Sub-Saharan African remains unsolved. HF has no real African recipe corpus with
ingredient lists — every candidate is images or captions. Morocco (2,589) is the only
African foothold. Filling West/East African properly will need scraping
(e.g. allnigerianrecipes, kenyanfoodrecipes) rather than a dataset download.

## Sweep exhaustiveness

Hugging Face is exhausted for this purpose. The final sweep enumerated **1,836 distinct
datasets** across 68 cuisine/region/language queries; only **7** carried a real ingredient
column, and of those only `halal-2k` and `thai-1k` survived dedup. Earlier sweeps covered
761 and 231 candidates. Anything not listed in this file was checked and rejected.

GitHub was searched for the African gap (`gh search repos` across Nigerian/Ghanaian/Kenyan/
Ethiopian/pan-African terms, plus `gh search code`). Every hit is a tutorial web app, not a
corpus — the largest data file found is `Jogwums/nigerian-food-api/db.json` at **5.4 KB**
(~15 recipes). Confirmed: no downloadable African recipe corpus exists.
