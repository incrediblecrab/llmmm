# Epicure → LLMMM: Research Findings

Everything discovered while auditing the Epicure release for use in a shipped iOS app.
Each item is marked **VERIFIED** (reproduced locally) or **REPORTED** (from docs/web).

---

## 1. The headline: the whole engine is 2.1 MB

**VERIFIED.** `embeddings.safetensors` is a `[1790, 300]` float32 matrix = 2,148,000 bytes.
Every operator in the paper is small linear algebra:

| Operator | Work | Practical cost on-device |
|---|---|---|
| `neighbors` | `[1790,300] @ [300]` + top-K | microseconds via Accelerate |
| `closest_mode` | 150–200 dot products | negligible |
| `slerp` | Gram–Schmidt + 2 scalar mults | negligible |

**Implication:** no backend, no API key, no per-query cost, full offline. This is the
single most important architectural fact about the project.

Built bundle for all three siblings, including poles and mode atlases: **7.49 MB**.

---

## 2. Data integrity: 24/24 checks pass

**VERIFIED** via `tools/verify_epicure.py`:

| Check | cooc | core | chem |
|---|---|---|---|
| Shape `[1790,300]` | PASS | PASS | PASS |
| Participation ratio (published) | 178.6 (173.6) | 96.1 (94.2) | 186.5 (183.1) |
| Avg pairwise cosine | 0.098 | 0.348 | 0.116 |
| Mode count (published) | 150 (150) | 193 (193) | 200 (200) |
| Supervised pole keys | 97 | 113 | 120 |
| NaN/Inf | none | none | none |

My participation ratios run ~2–3% above published — consistent offset across all three,
so almost certainly a minor definitional difference (centering/normalisation) rather than
a data problem. Ordering and magnitude match exactly.

**Embeddings ship un-normalised** (mean L2 norm 1.87 / 2.20 / 2.26). L2-normalise once at load.

---

## 3. ⚠️ Only **7** cuisine poles ship, not 8 — and the sets differ per sibling

**VERIFIED.** This will break naive UI code.

| Sibling | Cuisine poles present | **Missing** |
|---|---|---|
| cooc | East_Asian, Japanese, Latin_American, Mediterranean, South_Asian, Southeast_Asian, Western_Atlantic | **Eastern_European** |
| core | East_Asian, **Eastern_European**, Latin_American, Mediterranean, South_Asian, Southeast_Asian, Western_Atlantic | **Japanese** |
| chem | East_Asian, Japanese, Latin_American, Mediterranean, South_Asian, Southeast_Asian, Western_Atlantic | **Eastern_European** |

The paper defines **8** macro-regions (`cuisine_macroregions.json` lists all 8). The
heuristic reconstruction only recovered 7 per sibling, and **core recovers a different 7**.

**Consequence:** if the app offers a cuisine picker *and* a sibling crossfade, the option set
must be recomputed per sibling, or the intersection (6 regions) used. Hardcoding 8 will crash.

---

## 4. ⚠️ The paper's SLERP table does not reproduce — even using the author's own recipe

**VERIFIED** via `tools/reproducibility_audit.py` and, decisively,
`tools/slerp_via_space_recipe.py`.

### 4a. The directions ARE reconstructible — the Space shows how

A first pass concluded 90% of `direction_arithmetic_full.parquet` was unreachable because no
`sweet` / `nutty` / `high protein` vector ships. That was a **coverage** problem, and the
`epicure-explorer` Space solves it. `app.py::_aggregate_pole` builds these directions on the fly
as the **unit-mean of every supervised pole whose key starts with a property prefix**:

```python
_SENSORY_SLIDER_KEYS = [("sweet", ["sweet_score/", "cf_sweet/"]),
                        ("umami", ["umami_score/", "cf_umami/", "cf_meaty/"]), ...]
```

This is sound: **98% of modes have `prop_z_mean > 0`**, so modes are the *high* end of each
property and their mean is a genuine "high X" direction. Applying it lifts coverage from
**10% → 81.2%** (351/432 cells).

### 4b. But steering still does not reproduce

| θ | mean top-5 overlap | exact top-5 match |
|---|---|---|
| 0° | **94.4%** | **76.9%** |
| 30° | 34.5% | **0%** |
| 60° | 9.4% | **0%** |

At θ=0 the direction is irrelevant — this is just `neighbors`, and it reproduces well. The
moment you rotate, agreement collapses, and it collapses *monotonically with angle*. That is
the exact signature of a **wrong pole**: the shipped `supervised_poles.json` are not the
vectors that produced the published table.

### 4c. Three directions have no pole family at all

`floral` (27 cells), `nutty` (27) and `high water` (18) match no prefix — not even in the
author's own Space. Properties `cf_floral`, `cf_nutty`, `usda_water` simply don't exist.
The remaining 9 uncovered cells are the cuisine asymmetry of §3: `Eastern_European` (6 cells,
missing from cooc+chem) and `Japanese` (3 cells, missing from core) — an independent
confirmation of that finding.

### 4d. Airtight: the author's own live service contradicts the paper

`tools/api_differential.py` settles this with 99 live API calls.

**Parity (54 calls):** our bundle is **100% identical** to the author's live service —
same ranking every time, max cosine delta **1e-06** (float32 JSON rounding).

**Replication (45 published cells re-queried live):**

| θ | mean top-5 overlap | exact match |
|---|---|---|
| 0° | 94.7% | 73.3% |
| 30° | 26.7% | **0%** |
| 60° | 28.0% | **0%** |

Worked example — `rice + South_Asian`, core, θ=30:

| rank | **published table** | **author's live API** (= our output) |
|---|---|---|
| 1 | chana_dal 0.6976 | turmeric 0.7607 |
| 2 | fenugreek_leaf 0.6919 | mustard_seed 0.7567 |
| 3 | urad_dal 0.6838 | fenugreek_seed 0.7468 |
| 4 | toor_dal 0.6741 | coriander 0.7428 |
| 5 | horse_gram 0.6684 | cumin 0.7388 |

```
our bundle   ≡  live service        (100% parity, 1e-6)
live service ≠  published tables    (0% exact at any θ>0)
--------------------------------------------------------
therefore    shipped weights ≠ the weights behind the paper
```

**Zero overlap, different cosine range.** This cannot be our bug. The shipped
`supervised_poles.json` are not the vectors behind the published tables, and the
author's own demo doesn't reproduce them.

**Consequence:** `direction_arithmetic_full.parquet` is **not** a valid acceptance fixture for
SLERP. It is valid only at θ=0, i.e. for testing `neighbors`. Use
`data/derived/golden_fixture.json`, which is generated from the shipped weights and is
self-consistent.

**Product consequence:** the θ dial will work and feel good — it moves smoothly and lands
somewhere plausible — but you **cannot claim it reproduces the paper's published examples**.
Don't put paper screenshots in marketing.

---

## 5. ⚠️ Raw neighbours return near-duplicates — the app must filter them

**VERIFIED.** The paper's published rows are post-filtered by logic that is **not shipped**.
Diagnosing the θ=0 mismatches showed every one is an extra near-duplicate we return and they dropped:

| Seed | We return | Paper drops it |
|---|---|---|
| `rice` | `brown_rice` | dropped |
| `lamb` | `mutton` | dropped |
| `tomato` | `red_pepper` (with `bell_pepper` present) | dropped |
| `tofu` | `sesame_oil` (with `oil` present) | dropped |
| `egg` | `egg_yolk`, `baking_soda` (with `baking_powder`) | dropped |
| `pasta` | `tortellini` | dropped |

I tried to recover the exact rule. **It is not consistently applied** — `chicken` retains both
`chicken_broth` and `cream_of_chicken_soup` (the Core model card even advertises this), which
contradicts any seed-token rule. A token-overlap filter raises these specific cases to exact but
*lowers* overall θ=0 reproduction from 76.4% → 68.1%. So the paper's filter is not recoverable.

**This does not matter for reproduction — it matters for product.** Recommending "brown rice"
to someone who has rice, or "mutton" for lamb, looks broken. **A near-duplicate filter is a
product requirement regardless of what the paper did.** Design it for user experience, not fidelity.

---

## 6. Data-joining gotcha: space vs underscore

**VERIFIED.** Mode atlas members use **spaces** (`"oyster sauce"`); `vocab.json` keys use
**underscores** (`"oyster_sauce"`). Naive joins silently drop every multi-word ingredient.
After normalising, member-name resolution is **100.00%** (0 of 46,244 unresolved across all three siblings).

---

## 7. The three siblings are genuinely different — this is the product

**VERIFIED.** Same query, `miso`, top-5:

| Sibling | Result | Reads as |
|---|---|---|
| **cooc** | sake, snow_pea, mirin, shichimi_togarashi, fried_tofu_puff | "what people cook it with" |
| **core** | mirin, tofu, bonito_flakes, dashi, sesame_oil | the Japanese pantry |
| **chem** | octopus, kombu, natto, tofu, wakame | shared umami/marine compounds |

This is the crossfade feature, working, for free. It is the most defensible UX in the project.

## 8. The dial works — verified hero example

**VERIFIED.** `core`, seed `rice`, steering toward `cuisine:South_Asian`:

| θ | Top-5 |
|---|---|
| 0° | shiitake_mushroom, nori, bok_choy, brown_rice, sesame_seed |
| 30° | turmeric, mustard_seed, fenugreek_seed, coriander, cumin |
| 60° | mustard_seed, turmeric, kashmiri_chili, garam_masala, fenugreek_seed |
| 90° | kashmiri_chili, mustard_seed, curry_leaf, panch_phoran, garam_masala |

Note θ=0: plain "rice" returns **shiitake, nori, bok choy** — a vivid demonstration of the
corpus being ~50% East Asian (see §9). The dial itself is compelling and demoable.

---

## 9. Corpus imbalance is severe and will show

**REPORTED** (supplement Table A2), backing recipes per macro-region:

| Region | Backing recipes |
|---|---|
| East_Asian | 1,549,034 |
| Western_Atlantic | 198,086 |
| Mediterranean | 164,107 |
| Eastern_European | 154,479 |
| Southeast_Asian | 107,964 |
| South_Asian | 47,462 |
| Latin_American | 40,618 |
| Japanese | 33,923 |
| *Sub_Saharan_African* | *324 — below inclusion threshold, excluded* |

East Asian has **38× the backing of Latin American**. Steering quality will be visibly
uneven. Sub-Saharan African cuisine is effectively absent. This is a product-honesty issue,
not just a technical one — consider surfacing confidence per region rather than hiding it.

---

## 10. Other constraints

- **523 of 1,790 ingredients are chemistry hubs.** The other 1,267 reach compound context only
  indirectly. `chem` sibling quality is uneven for non-hubs. **REPORTED**
- **1,790 ingredients total** — no brands, no prepared foods. Real pantries will miss.
- **Vocabulary is LLM-curated** (Claude Opus 4.6 + Gemini embeddings), reduced from ~200,000
  raw NER strings through 6 stages. Errors from that pipeline are baked in. **REPORTED**
- **Not shipped:** raw recipe text, the 203,508-edge NPMI graph, the 80,019-edge typed
  compound graph, per-ingredient cuisine tags, compound-node embeddings. The authors say the
  graphs may come in a v2.

---

## 10b. Wine pairing does NOT come free — VERIFIED

Tempting roadmap item, since the vocab has 153 `Beverage` entries and ~25 real
wines. **Tested against 15 textbook sommelier pairings: AUC 0.700**, versus
**0.995** for food-food. With n=15 that is not distinguishable from chance.

Individual failures show it isn't a threshold problem, it's the wrong signal:

| pair | sommelier | engine |
|---|---|---|
| `red_wine + oyster` | textbook clash | **84.7 plausible** |
| `white_wine + beef` | clash | **85.1 traditional** |
| `champagne + caviar` | the famous one | **57.4 clash** |
| `white_wine + sole` | textbook match | **27.2 clash** |
| `red_wine + mushroom` | — | 98.0 excellent |

**Root cause: the corpus is recipes, so wine appears as an ingredient you cook
*with*, never a beverage you drink *alongside*.** Look at the neighbourhoods:

```
cooc  red_wine -> thyme, rosemary, marjoram, olive_oil, savory   (a braise)
cooc  sherry   -> veal, butternut_squash, mushroom               (a pan sauce)
chem  red_wine -> port_wine, wine, white_wine, champagne, sherry, rose_wine
```

Two independent failures:
1. **`cooc` measures deglazing, not drinking.** `red_wine + mushroom` scores 98
   because of coq au vin — correct, and irrelevant to what's in your glass.
2. **`chem` collapses the whole category.** Every wine's nearest neighbours are
   just other wines, so chemistry cannot tell a Riesling from a Cabernet.

Also, **there are no drinking wines in the vocabulary** — no `pinot_noir`,
`chardonnay`, `cabernet_sauvignon`, `riesling`, `syrah`, `chianti`. Only generic
`red_wine`/`white_wine` plus the cooking shelf (marsala, madeira, shaoxing,
mirin). That is the cellar-vs-pantry distinction in one line.

**Implication:** wine pairing is a *new data problem*, not a feature toggle. It
needs a varietal vocabulary and pairing-labelled (not co-occurrence) supervision.
Do not ship it off these embeddings — it will confidently recommend red with
oysters.

## 11. Licensing — the serious one

Full detail in `LICENSE_AUDIT.md`. Summary of the conflict:

- **Epicure weights are CC BY 4.0** — explicitly *"even commercially"*, per the shipped `LICENSE`.
- **But its two largest inputs are NonCommercial:**
  - **RecipeNLG — 53.9% of the corpus** — non-commercial research terms.
  - **FlavorDB** (the entire chemistry layer) — CC BY-NC-SA 3.0.
  - Also **Recipe1M+** (via FlavorGraph lineage) — CC BY-NC-SA 4.0.

Whether trained embeddings are a "derivative work" of training data is legally unsettled.
The authors granted CC BY 4.0; the risk is whether that grant is theirs to make. **This is a
question for a lawyer, not for me** — but it must be resolved before charging money.

It also weakens my earlier suggestion of recomputing co-occurrence statistics from the
original corpora: those statistics would derive from the same NC sources.

**Cleanly permissive:** USDA FoodData Central (CC0), FlavorGraph code (Apache 2.0),
`somosnlp/recetas-cocina` (MIT), `Frorozcol/recetas-cocina` (MIT), Jain Indian 6000+ (CC BY 4.0).

---

## 12. Attribution obligation

CC BY 4.0 requires visible credit. Ship this in an About/Credits screen:

> Ingredient embeddings from **Epicure** by Jakub Radzikowski and Josef Chen (KAIKAKU.AI),
> licensed under CC BY 4.0. Paper: *Epicure: Navigating the Emergent Geometry of Food
> Ingredient Embeddings*, arXiv:2605.22391.

---

## 14. Ground truth: we validated against their artifacts, then finally against reality — VERIFIED

Everything in §1–§13 was checked against Epicure's *published artifacts*. That proves we
reproduce their pipeline's output; it says nothing about whether the output is right.
The repo named `epicure-corpus-resources` contains **no corpus** — the largest file is
2,160 rows, and `epicure_{cooc,core,chem}.csv` are just the same 1,790×300 embeddings
again. The real corpus (4,135,189 recipes over 11 sources, per the supplement) was never
released.

We substituted `corbt/all-recipes` — **2,147,248 English recipes** — and built our own
co-occurrence table: `tools/build_recipe_cooc.py` → `data/derived/recipe_cooc.npz`,
94.7% ingredient match rate, **297,543 distinct pairs** covering 1,614/1,790 of vocab.

### 14a. The raw statistics beat the embedding decisively

On the held-out human-judged set (disjoint from anything used to tune):

| signal | AUC |
|---|---|
| engine score, embeddings only | 0.873 |
| log recipe count | 0.971 |
| **raw nPMI** | **0.984** |

The `cooc` embedding is a 300-d compression of exactly this co-occurrence signal, and it
loses most of it: `spearman(cosine, nPMI) = +0.498`, and **only 7% of the top-100 pairs
by nPMI appear in the top-100 by cosine**. The sibling whose entire purpose is encoding
co-occurrence disagrees with the source data on 93% of its strongest claims.

This is not an artifact of sparsity. 0 of 29 held-out good pairs have zero co-occurrence
(counts 60–23,940); only 3 of 13 bad ones do. GOOD nPMI spans +0.011…+0.351, BAD spans
−0.650…−0.087.

### 14b. Methodological warning: a 17% subsample gave the opposite answer

We first used a 399,942-recipe subset and concluded the engine's misses were "genuinely
rare pairings". The full corpus reversed every case:

| pair | 400k subset | 2.15M full | old engine |
|---|---|---|---|
| pork + apple | 23 → "rare" | **3,760** | 51.8 clash |
| chicken + lemon | 2,863 | **23,940** | 51.6 clash |
| corn + butter | 1,534 | **19,262** | 61.2 clash |
| pineapple + ham | 99 | **1,432** | 25.0 clash |
| miso + butter | 0 | **272** | 2.8 clash |

Do not draw corpus conclusions from partial data. The subsample was not just noisier — it
was confidently wrong.

### 14c. The fix, and what it bought

`pairing.py` now consults real recipes first and falls back to the embedding only where
the corpus is silent (`W_RECIPE = 0.7`, count and nPMI weighted equally via `W_NPMI`).
Both statistics are needed: nPMI alone penalises pairs of individually-common ingredients
(chicken+lemon appears in 23,940 recipes but scores nPMI +0.011), while count alone
rewards "salt + everything".

| metric | embeddings only | + real recipes |
|---|---|---|
| held-out AUC | 0.873 | **0.976** |
| generalisation gap | +0.122 | **+0.024** |
| good pairs called "clash" | 9/29 (31%) | **4/29 (14%)** |
| bad pairs sold as plausible | 0/13 | **0/13** |

The sweep's argmax was actually `W_RECIPE=1.0` at AUC 0.992 — discard the embedding
entirely. We did not take it: that is one pair better on a 42-pair set, and it pushes
`vanilla+onion` to 69 and `salt+sugar` to 87. See `tools/tune_recipe_weight.py`.

The embedding still earns its place. Only **18.6% of all possible pairs** occur in any
recipe; for the other 81% the embedding is the only signal there is. Among *common*
ingredients (≥1,000 uses) corpus coverage rises to **81.3%**, which is where real user
queries land.

### 14d. Remaining misses are corpus bias, not model error

The 4 surviving held-out misses are `miso+butter`, `chocolate+chili_pepper`,
`pineapple+ham`, `rhubarb+ginger` — Japanese, Mexican, retro-American and British. Our
substitute corpus is American home cooking: `miso` appears 272 times against `butter`'s
87,215. Epicure's real corpus is 37.4% XiaChuFang (Chinese), which we do not have.
Adding a Chinese recipe source is the highest-value next data step.

### 14e. The siblings are not independent evidence

The paper's framing — three complementary views — is weaker than it reads. All three
predict recipe co-occurrence almost identically (cooc 0.947, core 0.945, **chem 0.941**;
a molecular embedding matching the behavioural one is a red flag). Pairwise Spearman in
percentile space: cooc–core +0.468, cooc–chem +0.455, core–chem +0.535.

They are *moderately* correlated, not redundant — siblings still disagree by >30
percentile points on ~32% of pairs, so the novel/traditional labels remain meaningful.
But "three independent kinds of evidence" overstates it, and any UI copy should say
"three views" rather than implying independence.

### 14f. The "novel pairing" label was broken, and ground truth exposed it

Sampling 40,000 random pairs, the highest-scoring "novel discoveries" were
`sturgeon+snail`, `pike+horse_meat`, `dumpling+frog_leg`, `camellia_oil+conch` — all with
**0 recipes**. Two failures compounded:

1. **Absence of data read as absence of tradition.** Rare ingredients never co-occur
   because they are rare. A "cooks have overlooked this" claim is only meaningful if cooks
   had the opportunity: `Pairing.MIN_UNI = 400` now requires both ingredients to be common
   enough that the silence is informative.
2. **High `chem` similarity means substitutable, not pairable.** Two ingredients sharing
   a compound profile *and* a culinary role are interchangeable — that is the substitute
   relationship, not a pairing. Every false novel had `core ≥ 80` and `chem ≥ 93`; real
   pairings like `tomato+basil` (88.6/75.9) sit lower. Pairs with `core ≥ 90` or
   `chem ≥ 97` are now excluded from the novel label.

This is the food-pairing hypothesis' known weak spot (cf. Ahn et al. 2011: East Asian
cuisines actively *avoid* compound sharing), and it hit the exact feature meant to be the
app's differentiator.

After both guards, novel fires on 0.21% of random pairs and the top hits read like real
suggestions: `peanut+tamari`, `ghee+curry_powder`, `clementine+mint`,
`red_curry_paste+alfredo_sauce`. Traditional fires on 2.98%.

### 14g. `strawberry + basil` is a clash, and the docs said otherwise

Documented in three files as the flagship "traditional" example. The engine returns
**46.5 / clash** (cooc p79, chem p1, 522 recipes). The gap criterion fires but the
`overall ≥ 65` floor blocks the label — and the two are anti-correlated by construction,
since a large gap means one sibling is low, which drags `overall` down through the
`0.3·mid + 0.2·min` terms. Corrected to `chicken+lemon` (66.7, traditional, 23,940
recipes), which is a genuine instance.

### 14h. PMI breaks on background ingredients — a count floor is required

`butter` appears in 26% of all recipes. So `garlic + butter` (56,496 recipes) co-occurs
slightly *less* than independence predicts and scores negative nPMI — statistically
correct (garlic skews savoury, butter skews baking) and culinarily absurd. The engine
called it a clash at 56.0.

`FLOOR_COUNT = 4_000` / `FLOOR_SCORE = 70.0`: above the ~99th percentile of pair counts,
absolute evidence overrides the association measure. It fires on 5,767 of 297,543 pairs
and changed no held-out result — no bad pair comes close to 4,000 recipes — while fixing
`garlic+butter`, `broccoli+butter` and `butter+soy_sauce`.

### 14i. The resolver was the real product blocker

A simulated end-to-end session (parent, 6:15pm, eight fridge items in natural language)
exposed something none of the metrics caught: `resolve()` did an exact lookup only, so
`cheddar`, `spaghetti` and `chicken breast` all returned `None` — and the caller
silently dropped those pairs. Three of eight fridge items were invisible.

`resolve()` now runs loosest-strategy-last: exact → plural (`-ies`/`-es`/`-s`) → alias →
head-noun completion (`cheddar` → `cheddar_cheese`) → style/cut modifier stripping
(`boneless skinless chicken breast` → `chicken`, `greek yogurt` → `yogurt`) → unique
whole-token containment. Aliases cover British spellings (`aubergine`, `courgette`,
`prawns`, `chilli`) and words absent from the 1,790-term vocabulary (`spaghetti` →
`pasta`). Household-term resolution went 11/20 → 19/20, and held-out AUC rose 0.976 →
**0.978** purely because fewer reference pairs were being skipped.

Lesson worth carrying into the app: **a vocabulary miss is indistinguishable from a bad
pairing unless the UI says so.** Never fail silently — always report the resolved term.
