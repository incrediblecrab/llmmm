# LLMMM — Implementation Plan (Fable 5 handoff)

**Purpose:** this document exists so a Fable 5 coding agent can build the app *without
exploring*. Every ambiguity that could be resolved cheaply has been resolved here.
At $10/$50 per MTok, discovery is the expensive failure mode — not typing.

**Read first:** `FINDINGS.md` (data gotchas), `LICENSE_AUDIT.md` (commercial risk).

---

## 0. Status: Phase 0 is DONE

Already built and verified in this repo — **do not redo**:

| Artefact | What it is |
|---|---|
| `data/raw/epicure-{cooc,core,chem}/` | Full model repos, 3× |
| `data/raw/epicure-corpus-resources/` | Mode atlases, probes, supplement |
| `data/derived/bundle/` | **iOS-ready blobs, 7.49 MB** |
| `data/derived/golden_fixture.json` | Swift acceptance-test values |
| `tools/*.py` | Verification + build pipeline, reproducible |

Bundle format (little-endian, row-major, **already L2-normalised**):

```
{sib}.emb.f32     1790 × 300 float32   — mmap directly, 2.1 MB
{sib}.poles.f32      P × 300 float32   — P = 97 / 113 / 120
{sib}.modes.f32      M × 300 float32   — M = 150 / 193 / 200
{sib}.meta.json                        — ingredients[], pole_keys[], modes[], member ids
manifest.json                          — sizes, counts, attribution
```

`meta.json.ingredients[i]` is the name of embedding row `i`. Modes reference members by
integer index. No string joining needed at runtime.

---

## 1. Architecture — the one decision that matters

```
┌─────────────────────────────────────────────┐
│  ON-DEVICE, OFFLINE, $0, microseconds       │
│  ─────────────────────────────────────────  │
│  neighbours · SLERP dial · closest-mode     │
│  basket centroid · bridge · substitute      │
│  sibling crossfade · mode atlas browsing    │
└─────────────────────────────────────────────┘
                     │  only when user taps "Make it real"
                     ▼
┌─────────────────────────────────────────────┐
│  FABLE 5 — one call, ~$0.10                 │
│  vector result ──► actual recipe            │
└─────────────────────────────────────────────┘
```

**Never** put an LLM in the retrieval path. The geometry is free and instant; that is the
entire competitive advantage. If a feature can be expressed as linear algebra, it must be.

---

## 2. Phase 1 — Swift core engine (`EpicureKit`)

Pure Swift package, **no UI, no dependencies**, 100% unit-tested. Build this first and
completely; everything else depends on it.

```
EpicureKit/
  Sources/EpicureKit/
    EpicureModel.swift      // mmap loader, Sibling enum
    Operators.swift         // neighbours, slerp, closestMode, centroid
    Dedup.swift             // near-duplicate filter (see §2.3)
    Bridge.swift            // basket bridge + substitution
  Tests/EpicureKitTests/
    GoldenTests.swift       // asserts against golden_fixture.json
```

### 2.1 Loading
- Load `.f32` blobs via `Data(contentsOf:options:.mappedIfSafe)` — do **not** parse into arrays.
- Bind memory to `Float`. Vectors are already unit-length; **do not normalise again**.
- Assert `data.count == n * 300 * 4` on load and fail loudly.

### 2.2 Operators — use Accelerate
- Top-K: `cblas_sgemv` for the `[n,300] × [300]` product, then a partial selection.
  Do **not** fully sort 1,790 elements per frame; use a bounded max-heap of size K.
- SLERP (must match exactly — this is the reference):
  ```
  d_perp = d - (d·v)v
  if ‖d_perp‖ < 1e-9 : return v
  q = normalize(cos θ · v + sin θ · normalize(d_perp))
  ```
  θ in **degrees** at the API boundary; convert internally. Seed excluded from results.
- `closestMode`: dot the seed against `modes.f32`, take top-K, filter by `kind`.
- Basket centroid: **mean of unit vectors, then re-normalise**.

### 2.3 Dedup filter — REQUIRED (see FINDINGS §5)
Raw neighbours return `rice → brown_rice`, `lamb → mutton`, `tomato → red_pepper`.
Filter a candidate if it shares a token with the seed **or** with any higher-ranked kept hit,
plus a small hand-curated synonym list (`lamb/mutton`, `pasta/tortellini`).
Make it a **toggleable parameter**, defaulted on, so it can be tuned against real output.
Do **not** try to match the paper's filter — it is inconsistent and unrecoverable.

### 2.4 Acceptance tests
`golden_fixture.json` has, per sibling: 10 neighbours for 6 seeds, and `rice → South_Asian`
SLERP at θ ∈ {0,30,60,90}. Assert names exactly and similarities to **1e-5**.
Also assert: all rows unit-length; `cuisine_poles` count is **7**, not 8; and that the
**cuisine pole sets differ between siblings** (core has `Eastern_European`, lacks `Japanese`).

### 2.5 Performance budget
Top-K over 1,790 × 300 must run in **< 2 ms** on an A15. The dial re-queries every frame;
budget 16 ms total. Add an XCTest performance assertion.

---

## 3. Phase 2 — App

Target **iOS 17+**, SwiftUI, MVVM. Screens:

1. **Pantry** — searchable list over 1,790 ingredients, multi-select basket, recents.
2. **Dial (hero)** — basket → live top-K, with two controls:
   - **θ slider**, 0–90°, steering toward a chosen cuisine/mode pole
   - **sibling crossfade**, cooc ↔ core ↔ chem
   Results must animate/reorder continuously as the slider moves. This is the app.
3. **Atlas** — browse 150–200 named modes (labels already ship; free content).
4. **Dish** — the single Fable 5 call.

### ⭐ "I have X — can it pair with Y?" — the flagship screen

Implemented and calibrated in `tools/pairing.py`; port it verbatim. This is the
clearest expression of the paper's actual insight, and it needs no LLM.

**The three siblings are three independent kinds of evidence:**

| sibling | question it answers |
|---|---|
| `cooc` | do people actually cook these together? |
| `core` | do they play a similar culinary role? |
| `chem` | do they share flavour compounds? |

Agreement = confidence. **Disagreement is the interesting part**, and it is what a
recipe search can never tell you:

- **chem high, cooc low → "novel"** — a molecular basis with no tradition behind it.
  This is the food-pairing-hypothesis move, and it's a genuine discovery surface.
- **cooc high, chem low → "traditional"** — a classic that works for cultural
  reasons, not chemical ones. (`chicken + lemon`, `lamb + mint`.) Note `strawberry +
  basil` scores 46.5/clash despite the classic gap signature — the `overall >= 65` floor
  blocks it, and only 522 of 2.1M recipes use both. Do not use it as the demo case.

**Two calibration steps are mandatory — skip either and the feature is nonsense:**

1. **Per-sibling percentile.** Raw cosines are NOT comparable across siblings
   (mean pairwise: cooc 0.098, core 0.348, chem 0.116). A raw 0.40 is
   extraordinary in `cooc` and mediocre in `core`. Convert to a percentile against
   each sibling's own pairwise distribution.
2. **An empirical null.** After step 1 a *random* pair still averages ~50, which
   would read as "plausible". Score against the distribution of ~30k random pairs
   so the headline is "better than X% of all 1.6M possible pairs". Ship both
   quantile tables in the bundle — they're a few KB.

**Tiers are empirically calibrated**, not guessed (`tools/calibrate_pairing.py`):
20 canonical pairings score min 65.5 / median 95.6; 10 implausible ones max 66.9.
**In-sample AUC = 1.000, held-out AUC = 0.976** (`tools/validate_holdout.py`; the
in-sample figure is tuned-on and meaningless alone). Keep both as CI regression tests —
if a scoring change
drops AUC or pushes a canonical pairing into "clash", the change is wrong.

**Similarity ≠ complementarity.** Very high similarity means *substitute*, not
pairing (`rice + brown_rice` scores 96.6 but is useless advice). Flag substitutes
via shared name token, a curated synonym list, or same-food-group + core p≥99.5.

**Supporting evidence to render:** shared modes (top-8 core only), cuisine
agreement, and **bridges** — ingredients scored by `min(sim to X, sim to Y)`, i.e.
"add this and it works". Bridges are the best answer when the verdict is weak, and
they turn a "no" into a useful suggestion.

### ⚠️ Sensory axes: reconstructible, but DO NOT ship as labelled sliders
`data/raw/epicure-explorer/app.py` shows the author reconstructs sensory directions at
runtime as the unit-mean of supervised poles matching a property prefix (`_aggregate_pole`).
The bundle now emits these as `{sib}.sensory.f32` (10 axes × 300).

**I tested them and they do not work as labelled axes.** The 10 axes have a mean pairwise
|cos| of **0.94–0.98** — they are very nearly the *same vector*. Raw top-K is identical
for every axis and just returns corpus hubs:

```
cooc  sweet  -> black_pepper, garlic, grapeseed_oil, salt, lemon
cooc  citrus -> black_pepper, garlic, salt, cayenne_pepper, bay_leaf
cooc  bitter -> black_pepper, garlic, salt, cayenne_pepper, grapeseed_oil
```

Cause: every aggregate pole is ~0.95 aligned with the corpus mean direction. Averaging mode
poles regresses to the global mean. **Mean-centring before scoring is mandatory** and drops
axis collinearity from 0.96 → 0.34, producing vivid, distinct results:

```
cooc  sweet  -> mint, mango, honeydew_melon, grapefruit, hibiscus   ✅
cooc  bitter -> orange, lime, sage, ricotta_cheese, mint            ❌ that's citrus
cooc  citrus -> bay_leaf, rosemary, black_pepper, thyme, cumin      ❌ that's herbal
```

Vivid but **mislabelled**. Ship this only as an unlabelled "drift / explore" control, or
not at all. Do not put the words "sweet" or "bitter" next to these results.

### Anisotropy is a global engineering constraint
`cos(pole, corpus mean)` is **0.81–0.89** for cuisine poles too. Any operation built on an
*averaged* pole inherits this. Cuisine poles are less degenerate (0.70–0.81 collinear) and
the hero dial does work raw — **so keep cuisine steering uncentred** (centring measurably
hurt θ=0 agreement, 0.944 → 0.786). Rule of thumb: **single poles raw, aggregated poles
centred**, and sanity-check any new aggregate before exposing it in UI.

### Mode rendering — show the top 8, never the full membership

`docs/REPLICATION.md` shows modes have a **tight core and a near-random tail** (median 89
members, max 254). Against size-matched random groups the true coherence margin is only
+0.06 to +0.09 — the published ~0.5 margin compares two different statistics.

**Rule:** rank members by cosine to the mode `pole`, render the **top 8**, and stop. The
published coherence figures are only achieved over roughly that many members. Treat modes
as *labels and steering targets*, not as browsable lists — a full-membership screen will
surface the tail and read as broken. Consider demoting Atlas from a top-level tab.

### Cuisine picker — do not hardcode
Read available cuisine poles **from the loaded sibling's meta**. The set changes when the
crossfade changes. Show 6-region intersection, or regenerate options per sibling.

### Honesty affordance
Region backing counts are in `cuisine_macroregions.json` (East Asian 1,549,034 vs Latin
American 40,618). Surface a confidence hint on thin regions rather than pretending parity.

---

### 🍷 Wine pairing — deferred, and NOT a small extension

Tested, not assumed: **AUC 0.700 vs 0.995 for food** (FINDINGS §10b). The corpus
is recipes, so wine is encoded as *an ingredient you cook with* — `cooc` gives
`red_wine -> thyme, rosemary, marjoram` (a braise), and `chem` collapses every
wine onto every other wine. The engine rates `red_wine + oyster` at 84.7
"plausible". Shipping that is a credibility loss on the exact axis where a food
app cannot afford one.

If it becomes a priority, the viable route reuses what already exists rather than
re-embedding: extract the dish's **flavour anchors** with the current engine, then
map anchors → wine descriptors through a small curated sommelier table. That keeps
the geometry doing what it is good at (finding anchors) and puts the wine
knowledge where it belongs (curated, auditable, ~200 rows). It also degrades
honestly: no anchor match means no recommendation.

## 4. Phase 3 — Fable 5 integration

### 4.1 Build-time content (do this once, Batch API, ~$7)
Pre-generate and **ship in the bundle**:
- ~1,790 ingredient blurbs (~150 tokens each)
- 150–200 mode descriptions

`1790 × 150 = 268k output tokens × $25/MTok (batch) ≈ $6.71.` Zero runtime cost, works offline.
This is the best dollar-for-value use of the credits.

### 4.2 Runtime "Make it real"
Input: basket + steered results + mode labels + θ. Output: recipe with ratios, technique, timing.
~2k in / 1.5k out ≈ **$0.10/call**. Cache aggressively by basket hash.

**Decide before building:** whose key pays. Options — (a) fully offline v1, no LLM at all;
(b) user supplies their own key; (c) you pay, requires subscription. **(a) is the cleanest v1**
and defers the whole billing/backend question.

---

## 5. Budget: spending $200 of Fable 5 on this

At $10/$50 per MTok with prompt caching (cache hits $1/MTok), figure **~$0.30–0.60 per agent
turn**, so roughly **350–600 turns**. Remember Fable 5's tokenizer produces **~30% more
tokens** for the same text — real budget is ~30% tighter than naive estimates.

**Spend it here:**

| Use Fable 5 for | Use a cheaper model for |
|---|---|
| `EpicureKit` operators + Accelerate perf | SwiftUI boilerplate, list views |
| The 60fps dial interaction | Settings, About, navigation |
| Dedup filter tuning against real output | Asset/colour plumbing |
| Debugging numerical mismatches | Test scaffolding |

**Rules for the agent:**
1. Phase 0 is done. Do not re-download or re-derive data.
2. Read `FINDINGS.md` before touching data — every known trap is listed.
3. Run `golden_fixture.json` tests after each engine change.
4. Never load an LLM into the retrieval path.

---

## 6. Blockers to settle before writing app code

| # | Question | Why it blocks |
|---|---|---|
| 1 | **Commercial licensing** (FINDINGS §11) | RecipeNLG (54% of corpus) and FlavorDB are NonCommercial. Epicure grants CC BY 4.0 on top. Needs a lawyer, or ship free. |
| 2 | **Name** | "LLMMM" is unpronounceable and misdescribes an app whose core loop is deliberately *not* an LLM. |
| 3 | **Runtime LLM: yes or no?** | Determines whether v1 needs any backend at all. |
| 4 | **Scope** | TestFlight prototype vs App Store submission. |

Items 1 and 3 change the architecture. Settle them first.

---

## 7. Attribution — required, non-optional

CC BY 4.0. Ship in About/Credits:

> Ingredient embeddings from **Epicure** by Jakub Radzikowski and Josef Chen (KAIKAKU.AI),
> licensed under CC BY 4.0. Paper: *Epicure: Navigating the Emergent Geometry of Food
> Ingredient Embeddings*, arXiv:2605.22391.
