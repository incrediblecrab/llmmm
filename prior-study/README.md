# LLMMM

An iOS app built on the **Epicure** ingredient embeddings — navigating flavour space as
measured geometry rather than LLM guesswork.

> Working name. See `docs/IMPLEMENTATION_PLAN.md` §6 — "LLMMM" is unpronounceable and
> names the one component the architecture deliberately minimises.

## The idea in one line

The entire flavour engine is **2.1 MB** and runs on-device in microseconds, so retrieval,
steering and substitution are free, instant and offline — and an LLM is called only at the
last step, to turn a vector into a dish.

## Layout

```
data/raw/          Epicure model repos + corpus resources, as published (29 MB)
data/derived/
  bundle/          iOS-ready float32 blobs + metadata (7.52 MB)
  golden_fixture.json   Swift acceptance-test values
tools/             Fetch, verify, audit and build pipeline
docs/
  FINDINGS.md            Research dossier — READ THIS FIRST
  REPLICATION.md         Independent replication of the paper's numbers
  IMPLEMENTATION_PLAN.md Fable 5 handoff plan
  LICENSE_AUDIT.md       Per-source licensing, commercial risk
  EXPLORER_API.md        Gradio API — why it matters + validation oracle
  EXPLORER_API_REFERENCE.md  All 34 endpoints, generated from the raw dump
ios/               (empty — Phase 1)
```

## Reproduce

```bash
python3 -m venv .venv && ./.venv/bin/pip install numpy safetensors pandas pyarrow
./tools/fetch_epicure.sh              # download everything (29 MB)
./.venv/bin/python tools/verify_epicure.py         # 24/24 integrity checks
./.venv/bin/python tools/reproducibility_audit.py  # what reproduces vs what doesn't
./.venv/bin/python tools/build_ios_bundle.py       # emit the iOS bundle
./.venv/bin/python tools/replicate.py              # independent replication (needs scikit-learn)
./.venv/bin/python tools/slerp_via_space_recipe.py  # SLERP reproduction via the Space recipe
./.venv/bin/python tools/api_differential.py        # live-API parity + replication (cached)
./.venv/bin/python tools/parse_api_docs.py         # regenerate the API reference
./.venv/bin/python tools/pairing.py                # "can X pair with Y?" demo
./.venv/bin/python tools/calibrate_pairing.py      # pairing calibration (in-sample; AUC 1.000)
./.venv/bin/python tools/validate_holdout.py       # honest held-out score (AUC 0.976)
./.venv/bin/python tools/audit_pairing_consistency.py  # 7 mechanical checks, all must PASS
./.venv/bin/python tools/build_recipe_cooc.py - allrecipes  # 2.1M recipes -> co-occurrence (~12 min)
./.venv/bin/python tools/ground_truth.py           # embeddings vs real recipes
./.venv/bin/python tools/tune_recipe_weight.py     # re-derive W_RECIPE / W_NPMI
./.venv/bin/python tools/recalibrate_corpus.py     # corpus-scale calibration
```

`build_recipe_cooc.py` needs `data/raw/recipenlg/ar_*.parquet` (807 MB, from the
`corbt/all-recipes` dataset) and writes `data/derived/recipe_cooc.npz`. Everything
downstream of it depends on that file; `pairing.py` degrades to embeddings-only without it.

## What the audit found

- ✅ All 24 data-integrity checks pass; isotropy, mode counts and vocab match the paper.
- ✅ Our pipeline is **100% identical to the author's live Space** — 54 differential API
  calls, same ranking every time, max cosine delta 1e-06. Every divergence below is theirs.
- 🚨 **The author's own live service does not reproduce the author's own paper.** Re-querying
  45 published cells live: 94.7% overlap at θ=0, but **0% exact at θ=30 or θ=60**.
- ⚠️ Only **7** cuisine poles ship, not 8 — **and the sets differ per sibling**.
- ⚠️ The paper's SLERP table is **81% reconstructible** using the author's own recipe
  (recovered from the Space source) — but still **0% exact at any θ>0**, with overlap
  decaying 94% → 35% → 9% as the angle grows. The shipped poles are not the published ones.
- ⚠️ Raw neighbours return near-duplicates (`rice → brown_rice`, `lamb → mutton`) — a
  dedup filter is a **product requirement**.
- 🚨 **The headline mode-coherence claim doesn't survive a like-for-like control.**
  The published "coherence" and "baseline" are different statistics on different
  populations. Against size-matched random groups scored identically, the true margin
  is **+0.06 to +0.09**, not ~0.5 — a **6–8× overstatement**. Modes are real but weak:
  tight core, near-random tail. See `docs/REPLICATION.md`.
- 🚨 **Aggregated poles collapse onto the corpus mean.** The 10 reconstructible sensory
  axes have mean pairwise |cos| of **0.94–0.98** — raw, every axis returns the same corpus
  hubs (`sweet`, `citrus` and `bitter` all → garlic, salt, black pepper). Centring fixes
  the collapse but not the labels. **Don't ship labelled sensory sliders.**
- 🚨 **Commercial licensing is unresolved.** RecipeNLG (53.9% of the training corpus) and
  FlavorDB (the whole chemistry layer) are NonCommercial, yet Epicure is released CC BY 4.0.

Details in `docs/FINDINGS.md` and `docs/REPLICATION.md`.

## The flagship feature: "I have X — can it pair with Y?"

`tools/pairing.py`. Pure geometry, no LLM, microseconds. The three siblings are three
independent kinds of evidence — and their **disagreement** is the interesting signal:

```
  tomato  +  basil
  cooc  ################### p 99.5  YES people cook these together
  core  #################... p 88.6  ~   similar culinary role
  chem  ###############..... p 75.9  ~   shared flavour compounds
  VERDICT: STRONG (better than 93.9% of all 1.6M ingredient pairs)
  bridge via: bell_pepper, red_pepper, olive_oil, red_onion, black_olive

  chicken  +  lemon                                    23,940 real recipes
  VERDICT: TRADITIONAL — an established pairing NOT explained by shared
           compounds; it works for cultural reasons.
```

Scores blend three embeddings with **real co-occurrence counts from 2,147,248 recipes**.
That matters more than the embeddings do: on a held-out human-judged set the raw recipe
statistics score AUC 0.984 against the embeddings' 0.873, and the embeddings alone rate
`chicken + lemon` a *clash* despite 23,940 recipes using both. See FINDINGS §14.

| | embeddings only | + real recipes |
|---|---|---|
| held-out AUC | 0.873 | **0.976** |
| good pairs called "clash" | 31% | **14%** |
| bad pairs sold as plausible | 0/13 | **0/13** |

Only 18.6% of possible pairs appear in any recipe, so the embeddings still carry the
other 81%; among common ingredients corpus coverage is 81.3%. In-sample AUC is 1.000 —
use `tools/validate_holdout.py`, not `calibrate_pairing.py`, to judge quality.

## The three siblings, same query (`miso`)

| Sibling | Top-5 | Reads as |
|---|---|---|
| cooc | sake, snow_pea, mirin, shichimi_togarashi, fried_tofu_puff | what people cook it with |
| core | mirin, tofu, bonito_flakes, dashi, sesame_oil | the Japanese pantry |
| chem | octopus, kombu, natto, tofu, wakame | shared umami compounds |

## Attribution

Ingredient embeddings from **Epicure** by Jakub Radzikowski and Josef Chen (KAIKAKU.AI),
licensed under **CC BY 4.0**. Paper: *Epicure: Navigating the Emergent Geometry of Food
Ingredient Embeddings*, [arXiv:2605.22391](https://arxiv.org/abs/2605.22391).
