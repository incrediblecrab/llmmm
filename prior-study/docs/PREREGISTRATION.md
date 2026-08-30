# Pre-registration — Cookbook Regression

**Frozen: 2026-08-05, before any Azure training job was run.**

This document fixes hypotheses, metrics and falsification criteria *in advance*.
Its purpose is to make the study falsifiable rather than decorative: without a
prediction recorded beforehand, any result can be narrated as a success after
the fact (HARKing — Hypothesising After Results are Known).

Amendments must be appended with a date and reason, never edited in place.

---

## 0. Standing rules

**R1. Pre-registration.** Every hypothesis below states a direction and a
threshold. Results that contradict them are reported as contradictions.

**R2. Independent ground truth.** Model quality is judged against labels that
were never used in training:

| Source | Size | Used in training? |
|---|---|---|
| Substitution pairs (`29-ingredient-substitutions`) | 210,612 | No |
| Held-out co-occurrence edges (10%) | ~20,350 | No — removed before training |

Cosine-similarity eyeballing ("does this look tasty") is not evidence and is
not reported as such.

**R3. Negative controls.** Every metric must be run on random Gaussian vectors
before it is trusted. A metric that rates noise as good is broken, and any
result computed with it is void. This gate is Phase 0 and blocks everything.

**R4. No local contamination.** Locally trained models in `models/` are
quarantined. They are not read, compared, or used to set thresholds until all
Azure results are final. Thresholds below come from the published paper and
from chance level, not from local runs.

**R5. Everything runs on Azure ML** in the `cookbook-regression` workspace, on
the subscription credit, so results are reproducible from a job ID.

**R6. Product relevance.** The end goal is a shippable consumer app. Every
hypothesis names the product decision it informs. A hypothesis that changes no
decision is removed.

---

## 1. Metrics (defined before results exist)

Let `W` be the `n x d` embedding matrix (n = 1,790, d = 300), rows L2-normalised
unless stated.

**M1 — Participation ratio.** With `λ` the eigenvalues of the covariance of `W`:

    PR = (Σ λ_i)^2 / Σ (λ_i)^2            range [1, d]

Number of effectively used dimensions. Published Epicure values: cooc 217.8,
chem 186.5. `PR < 10` is collapse. Note PR is *not* to be maximised: strict
isotropy conflicts with clustering (Mickus et al., ACL 2024). Healthy is ~100–220.

**M2 — Substitution triplet accuracy.** For a substitution pair `(a, b)` with
`votes >= 100` and a uniformly random `c`:

    correct if cos(a, b) > cos(a, c)

Reported over 20,000 sampled triplets with a fixed seed. **Chance = 0.50.**

**M3 — Substitution recall@10.** Fraction of an ingredient's known substitutes
that appear in its 10 nearest neighbours, averaged over ingredients with >= 3
known substitutes.

**M4 — Held-out link prediction AUC.** 10% of `ii_graph` edges are removed
before training. AUC of ranking held-out edges above degree-matched non-edges.
**Chance = 0.50.**

**M5 — Popularity coupling.** Pearson `r` between each of the top 3 principal
components of `W` and `log(unigram count)`. Also the Jaccard overlap between a
model's top-20 "what to buy" ranking and the top-20 by raw frequency.
**High overlap = the feature is a popularity chart in disguise.**

---

## 2. Hypotheses

### H1 — The `chem` collapse is structural, not an implementation bug

`walks_chem` builds walks from ingredient→compound→ingredient edges only, with
no ingredient–ingredient edges, and allocates `walks_per_node // n_categories`
= 6 walks per node per compound category. `walks_core` injects I–I edges at
`ii_repeat=10`. If collapse follows from the schema, it must vary smoothly with
that one parameter.

- **Manipulation:** `chem` with `ii_repeat ∈ {0, 0.1, 1, 10, 100}` — 5 models.
- **Prediction:** M1 increases monotonically in `ii_repeat`; `PR(0) < 10`.
- **Falsified if:** M1 is flat, non-monotonic, or `PR(0) > 50`. That would mean
  the collapse is an implementation bug, not the schema, and H1 is wrong.
- **Contribution:** the published work reports one operating point; this yields
  the collapse *curve* for bipartite metapath schemas.
- **Product decision:** whether a chemistry-driven "surprising pairing" feature
  is buildable at all.

### H2 — Matrix factorisation matches random-walk SGNS

Levy & Goldberg showed SGNS implicitly factorises a shifted PMI matrix. At
n=1,790 the full matrix is trivial to decompose exactly.

- **Manipulation:** SVD-PPMI on the co-occurrence matrix; IDF-weighted
  shared-compound SVD for chemistry; GloVe-style weighted factorisation — 3 models.
- **Prediction:** M2 within 0.02 absolute of the best SGNS model, M4 within
  0.03, with `PR > 50` in every case (no collapse possible by construction).
- **Falsified if:** SGNS beats factorisation by more than those margins.
- **Product decision:** if factorisation matches, retraining is seconds rather
  than hours, so the app can rebuild embeddings on new data continuously.

### H3 — "What should I buy" degenerates into a popularity chart for geometric reasons

The app simulation returned black_pepper, tomato, bell_pepper — i.e. global
frequency. Suspected cause: the leading principal components encode frequency,
so any score aggregating cosine similarity re-ranks by popularity.

- **Manipulation:** no new models; each model evaluated raw vs after
  all-but-the-top whitening (remove top 1–3 PCs, renormalise).
- **Prediction:** M5 shows `|r| > 0.5` for at least one of the top 3 PCs on the
  raw model; whitening reduces top-20 overlap with the frequency ranking to
  `< 0.3` while M2 degrades by no more than 0.02.
- **Falsified if:** `|r| < 0.3` everywhere (the cause is elsewhere), or
  whitening damages M2 by more than 0.02 (the cure is worse than the disease).
- **Product decision:** whether the flagship "buy one thing, unlock the most"
  feature can ship.

### H4 — ⭐ Does the Ahn et al. (2011) food-pairing asymmetry survive at scale?

Ahn, Ahnert, Bagrow & Barabási (*Sci. Rep.* 1:196, 2011) reported that Western
cuisines favour ingredient pairs **sharing** flavour compounds while East Asian
cuisines **avoid** them. That analysis used ~57,000 recipes from three
English-language Western sites. This corpus has ~4.3M recipes across ~13
languages from native-language sources.

- **Manipulation:** stratify the corpus by source language/origin (auditable
  metadata, *not* inferred per-recipe cuisine, which would introduce a modelling
  choice into the independent variable). One co-occurrence graph and one model
  per stratum — ~10 models.
- **Statistic:** for stratum `s`, mean shared-compound count over co-occurring
  ingredient pairs, minus the same quantity under a **degree-preserving null**
  (configuration model, 1,000 rewirings): `Δ_s`, reported with a 95% CI.
- **Prediction (Ahn):** `Δ_s > 0` for Spanish, German, Greek, Romanian, Russian;
  `Δ_s < 0` for Chinese, Japanese, Thai, Vietnamese, Filipino, Indonesian.
- **Falsified if:** signs disagree with Ahn for a majority of strata, or all CIs
  span zero (no cuisine signal at all).
- **Either outcome is a result.** Confirmation extends a landmark finding by
  ~75x in scale and beyond English-language sources; refutation suggests the
  original effect was an artefact of Western recipe-site data.
- **Product decision:** whether pairing recommendations should be
  cuisine-conditioned rather than global.

### H5 — Does flavour chemistry add anything at 4.3M recipes?

With a corpus this large, co-occurrence may already encode whatever chemistry
would have told us.

- **Manipulation:** `cooc`, `chem` (best `ii_repeat` from H1), `core` — 3 models
  at matched hyperparameters.
- **Prediction:** `core` exceeds `cooc` on M4 by `> 0.02`.
- **Falsified if:** `|M4(core) - M4(cooc)| <= 0.02` — chemistry is redundant.
- **Product decision:** this one has teeth. FlavorDB carries licensing risk
  (see `docs/LICENSE_AUDIT.md`) and is the sole source of the chemistry signal.
  If H5 is falsified, the app can **drop FlavorDB entirely**, removing a legal
  risk at no quality cost.

### H6 — Do general LLM embeddings already encode culinary pairing?

- **Manipulation:** embed all 1,790 ingredient names with a Foundry text
  embedding model; evaluate alone and blended with the corpus model as
  `α·LLM + (1-α)·corpus`, sweeping α — 1 extraction plus a sweep.
- **Prediction:** LLM alone beats corpus models on M3 for ingredients with
  `recipe_count < 50` (rare items); loses on M4 (co-occurrence); the blend beats
  both endpoints on M2.
- **Falsified if:** the blend never beats both endpoints at any α.
- **Product decision:** **cold start.** The simulation showed the resolver
  silently mapping `"Trader Joe's orange chicken"` → `orange`, producing false
  positives with no unknown path. Real users type arbitrary text; an LLM
  embedding space is what makes out-of-vocabulary input safe.
- **Note:** Foundry model deployments consume token-per-minute quota, not GPU
  vCPU quota, so this is viable on this subscription.

---

## 3. Phases and gates

| Phase | Work | Gate to proceed |
|---|---|---|
| 0 | Eval harness, held-out split, negative controls | **M2 ≈ 0.50 and M4 ≈ 0.50 on random vectors.** Blocks all training |
| 1 | H1 (5) + H2 (3) | H1 resolved either way |
| 2 | H3 (whitening) + H5 (3) | Popularity fix validated |
| 3 | H4 per-cuisine corpus build + ~10 models | Headline result with CIs |
| 4 | H6 + synthesis | — |

Phase 0 is not a formality. If a metric rates random vectors highly, every
number produced afterwards is meaningless.

## 4. Compute budget

`Standard_F2s_v2` (2 vCPU) x 3 nodes = 6 vCPU, the regional ceiling. ~25 models.
At ~$0.085/node-hour, even 40 node-hours is **~$4 of the $150 monthly credit**.
Wall-clock, not cost, is the binding constraint.

## 5. Known limitations, stated up front

- **GPU is unavailable** on this subscription; every method chosen is CPU-viable.
  This constrains H6 to API-based embeddings rather than a fine-tuned encoder.
- **Vocabulary is fixed at 1,790 ingredients** by the published artefact. H4's
  cuisine strata inherit that vocabulary, which was derived from a
  English-dominant corpus and may under-represent region-specific ingredients.
  This biases H4 *against* finding cuisine differences, so a positive finding is
  conservative; a null finding is weaker evidence and must be reported as such.
- **Corpus licensing.** RecipeNLG (~54% of the corpus) is non-commercial. H5
  determines whether FlavorDB can be dropped; the RecipeNLG constraint is
  separate and unresolved for commercial release.

---

## 6. Amendments

### A1 — 2026-08-05, before any model was trained

**Substitution vote threshold replaced by two pre-registered tiers.**

§3 M2/M3 originally specified `votes >= 100`. On building the harness, that cut
left only 919 unique pairs and 172 anchors of the 210,612 available rows — the
vote distribution has median 1, so a single high cut discards ~93% of usable
labels and leaves too little power to separate models.

Rather than pick a new single threshold (which would invite choosing it after
seeing results), both tiers are locked here and **both must be reported for
every model**:

| tier | rule | pairs | anchors (>=3 subs) |
|---|---|---|---|
| `broad` | `votes >= 1` | 13,793 | 1,139 |
| `strict` | `votes >= 10` | 3,322 | 487 |

Agreement between tiers is evidence of robustness; disagreement is a reportable
finding about vote noise, not a licence to select the flattering tier.

**Also locked in this amendment:**
- M2 and M4 report a 95% CI half-width (±0.0069 at current sample sizes).
  Differences smaller than the CI must be reported as indistinguishable, which
  supersedes the bare `0.02` thresholds used in H2/H3/H5 where the two conflict.
- M4 negatives are **degree-matched** (sampled within a ±10%-of-vocab band of
  the positive's degree rank) so link prediction cannot be won by popularity
  alone. Popularity is measured separately by M5.
- Ties in M2/M4 score 0.5 rather than 0, so a degenerate constant-similarity
  model scores exactly chance instead of being spuriously penalised.

**Reason for the change:** statistical power and control validity — both
decided from label counts and control behaviour only. **No model output of any
kind was consulted.**

### Phase 0 gate result — 2026-08-05, PASSED

Run before any training job. Controls behave as required:

| control | M1 | M2 broad | M2 strict | M4 | M5 |
|---|---|---|---|---|---|
| random gaussian | 256.7 | 0.5017 | 0.5010 | 0.4932 | 0.026 |
| collapsed rank-1 | **1.0** | 0.5000 | 0.5081 | 0.5002 | 0.050 |

All within ±0.0069 of chance on M2/M4, and M1 correctly reports 1.0 for a
collapsed model. The metrics do not reward noise, so downstream numbers are
interpretable.

**Incidental finding (pre-training, recorded now to prevent it being presented
later as a result):** 242 of 1,790 ingredients (13.5%) have **no co-occurrence
edge at all** in `ii_graph`. They were already isolated in the full graph — the
holdout split created zero new isolated nodes. These ingredients cannot be
learned by `cooc` or `core` and are reachable only through the chemistry graph.
This is the empirical cold-start population for H6 and a direct product
constraint. Only 2.5% of substitution anchors are affected, so M2/M3 are
essentially unaffected by it.

### Pipeline validation — 2026-08-05, Azure end-to-end proven

First successful cluster run (`brave_arm_s0z9qp7xrh`), a deliberately
undertrained 1-epoch `cooc` model whose only purpose was to prove the path:

```
graph=ii_graph_train.npz            <- trained on the split, no leakage
torch 2.10.0+cu126  cuda False      <- CUDA image falls back to CPU cleanly
saved epicure_cooc.npy (1790, 300)  <- written back to blob
240s / epoch on 2 vCPU
```

| metric | 1-epoch cooc | random control |
|---|---|---|
| M1 participation ratio | 6.3 | 256.7 |
| M2 triplet acc (broad) | 0.7805 | 0.5017 |
| M2 triplet acc (strict) | 0.8232 | 0.5010 |
| M3 recall@10 (broad) | 0.0398 | 0.0061 |
| M4 held-out link AUC | 0.7523 | 0.4932 |
| M5 max PC-freq corr | **0.6771** | 0.0256 |

This is **not** a result, and is recorded only as evidence the harness resolves
real signal from noise. It does license one early observation to be tested
properly under H3: M5 is already 0.677, i.e. the leading direction of the space
is largely "how common is this ingredient". A cosine-ranked app built on this
would recommend salt, butter and sugar to everyone.

**Infrastructure constraint discovered:** 240s/epoch x 20 epochs = ~80 min per
model, 3 models in parallel. The ~25-model program is therefore ~11 hours of
cluster time, comfortably inside the credit budget but not something to run
speculatively -- each submitted job should answer a pre-registered question.

---

### Amendment A2 — 2026-08-06, H7 added (registered before any H7 job ran)

**Provenance.** H1 is complete and every model reports M1, the participation
ratio (PR) — the number of embedding directions carrying non-negligible
variance. `cooc` trains at d=300 but uses PR ~115. H7 asks whether that number
is *real*: does PR measure usable capacity, or is it a descriptive statistic
with no operational meaning?

**H7 — the participation ratio predicts the minimum sufficient width.**

If PR measures usable capacity, then training at a width at or above PR should
cost nothing, and training below it should cost accuracy. Sweeping d over
{32, 64, 128, 300, 600} on `cooc`, holding all else fixed:

| width | prediction |
|---|---|
| 600 | no better than d=300 (M2 within +/-0.0069) |
| 300 | reference, M2 0.8677 |
| 128 | within CI of d=300 — above PR, so nothing is lost |
| 64  | measurably below d=300 |
| 32  | clearly below d=300 |

**Supported if** M2(128) is within the +/-0.0069 CI of M2(300) **and** M2(64) is
below it by more than the CI. That is a two-sided test: PR must be neither an
over- nor an under-estimate of the needed width.

**Falsified if** d=64 matches d=300 (PR overestimates — the space is far more
compressible than PR claims), or if d=128 is clearly worse (PR underestimates).

Falsification is genuinely possible and informative either way, which is why it
is worth the compute. This is registered now, with the prediction fixed, before
any H7 job is submitted.

**Why it matters beyond the paper.** The width chosen here is the width the iOS
app ships in its embedding table, so this converts a modelling statistic into a
product decision backed by a measurement rather than a guess.

**Note on the H1 recovery.** `core-ii1` completed on the cluster at 05:33 but
its result was lost by an orchestrator bug: a transient blob download failure
was indistinguishable from "the job produced no output", and the retry path sat
inside a branch that only ran on a status *transition*, so the job was never
re-scored. Fixed, and the stored artefact was re-scored rather than retrained —
no new training run, so no additional researcher degrees of freedom.

---

### Amendment A3 — 2026-08-06, H6 re-scoped and registered (before any H6 run)

**Why re-scoped.** H6 was originally "LLM *embeddings* for cold start". This
subscription has a token quota of **0** for every embedding model, so that form
is not runnable. gpt-oss-120b has 5,000 TPM, so H6 is re-scoped from embedding
geometry to *elicited pairing judgements*. This is a change of instrument, not
of question, and is recorded before any H6 job ran.

**The problem it addresses.** 242 of 1,790 ingredients (13.5%) have no
co-occurrence edge at all. For those, every model in this study is a random
vector — the app can say nothing. This is the cold-start case.

**H6 — an LLM's pairing judgements carry real co-occurrence signal.**

Procedure: sample *warm* ingredients (those with held-out edges), ask the model
for pairings constrained to our 1,790-token vocabulary, and score the returned
ranking against held-out co-occurrence with M3 recall@10.

**Supported if** LLM recall@10 >= 0.071 — half the trained `cooc` model's
0.1427 — **and** it beats the popularity control below by more than the CI.

**The control is the whole test.** An LLM asked for pairings will tend to name
salt, butter, onion and garlic, which score well by frequency alone. So the
pre-registered comparison is not against chance but against a
**frequency baseline** that ignores the query ingredient entirely and returns
the globally most common ingredients. If the LLM does not beat that baseline it
has contributed nothing beyond popularity — the same degeneracy M5 measures and
H3 failed to remove. Reported alongside is the LLM's own M5-style frequency
coupling.

**Falsified if** the LLM merely matches the frequency baseline, in which case
cold-start ingredients cannot be bootstrapped this way and the app must instead
mark them as unsupported rather than emit a confident-looking wrong answer.

Scoring uses the same held-out split and the same harness as every other
hypothesis, so H6 is comparable to the trained models rather than a side quest.

---

### Amendment A4 — 2026-08-06, H8 added (registered before any H8 job ran)

**Provenance.** Auditing corpus coverage to answer "did every recipe actually
get used?" exposed an inherited constraint we had not examined. Every recipe
*is* used: all 4,647,847 contribute, and every within-recipe ingredient pair is
counted exactly. But the graph is then **pruned**, and the pruning rule was
chosen for replication, not for modelling:

| | pairs | of possible |
|---|---|---|
| possible pairs (1790 choose 2) | 1,601,155 | 100% |
| co-occur in >=1 real recipe | 284,919 | 17.8% |
| co-occur in >=2 real recipes | 224,429 | 14.0% |
| **kept in the graph we trained on** | **203,504** | **12.7%** |

`build_ii_graph.py` sweeps NPMI thresholds and selects whichever one best
reproduces the paper's 203,508 edges. That discards **81,415 pairs that
genuinely co-occur** — 28.6% of all observed pairs — purely to hit a target
number from someone else's table.

**It also manufactures the cold-start problem.** 242 ingredients have no edge
under the threshold. On the unpruned observed graph that figure is **195**, so
**47 ingredients were severed by the threshold, not by missing data.** Their
median degree in the unpruned graph is 7.

**H8 — the inherited NPMI threshold discards usable signal.**

Train `cooc` identically (d=300, 20 epochs, seed 0) on the unpruned graph:
264,569 train edges vs 183,154, after removing the identical 20,350 held-out
edges. The evaluation set does not change, so M2/M4 stay directly comparable to
every model already run.

**Supported if** held-out M4 improves by more than the CI (+0.0069) over
`cooc`'s 0.8432. **Falsified if** M4 is unchanged or worse — which would
vindicate the threshold as a noise filter and is a genuinely useful answer.

**Why it is the highest-value experiment left.** Every other hypothesis varies
something we chose. This one varies something we *inherited without checking*,
and it is the only pending experiment that can improve the product directly:
if supported, 47 ingredients stop being unanswerable for free, and the 81,415
discarded pairs are negative evidence ("these are used together less than
chance") that the paper's graph cannot express at all.

**Leakage check performed before submission:** all 20,350 held-out edges were
located in the unpruned graph and removed (20350/20350).

---

### H8 result — 2026-08-06, FALSIFIED

| model | train graph | edges | M2 broad | M3@10 | M4 AUC | M5 freq |
|---|---|---|---|---|---|---|
| cooc | pruned | 183,154 | 0.8677 | 0.1427 | **0.8432** | 0.443 |
| cooc-full | unpruned | 264,569 | 0.8607 | 0.0971 | **0.7618** | 0.693 |
| core-ii10 | pruned | 183,154 | 0.8675 | 0.1449 | **0.8408** | 0.402 |
| core-ii10-full | unpruned | 264,569 | 0.8636 | 0.1038 | **0.7635** | 0.732 |

Pre-registered threshold was M4 improving by more than +0.0069. M4 **fell by
0.081**, so H8 is falsified and the inherited NPMI threshold is vindicated as a
noise filter rather than convicted as an arbitrary replication artefact.

**Mechanism, from M5.** Popularity coupling rises 0.443 -> 0.693 when the
discarded edges are restored. The 81,415 low-NPMI pairs are overwhelmingly
"this co-occurs with salt" — real co-occurrences that carry frequency
information rather than pairing information. NPMI thresholding removes exactly
that, which is what it was designed to do. The paper's constraint was load
bearing and we were wrong to suspect it.

**Consequence for cold start.** The 47 ingredients severed by the threshold
cannot be rescued by dropping it: doing so costs 0.081 M4 across all 1,790.
Cold start needs an external source, which is H6, and the honest population is
242 ingredients rather than 195.

**Bonus finding — M1 is not a quality proxy.** PR *rises* 115.1 -> 146.1 while
every accuracy metric falls. Participation ratio measures how many directions
carry variance, not whether that variance is useful. Combined with the H7
observation that PR is ~40% of whatever width is allotted, M1 should be read as
a descriptive geometry statistic and never as a scoreboard.

### Seed variance — measured, and larger than the sampling CI

`chem` vs `chem-s1`: M2 0.5784 vs 0.5906, M4 0.5198 vs 0.5042. Seed spread is
~0.012 M2 / ~0.016 M4, roughly **2x the +/-0.0069 sampling CI**. All
comparisons in this study must therefore clear ~0.02, not 0.007. This does not
change any verdict reached so far — H1's effects are ~0.3, H8's is 0.081 — but
it does confirm that `core-ii10` vs `cooc` (0.0025 apart on M4) is a genuine
null rather than a small real effect.

### Amendment A5 — 2026-08-06, recipe-level backtest (H9)

**Declared exploratory, not confirmatory.** The scorers below were chosen after
seeing aggregate results, so this section is written as a discovery plus a
pre-committed rule for the confirmatory replication that follows it. It is
reported as exploratory no matter how the confirmatory run turns out.

**Why.** M2-M5 all score the *graph*: held-out edges and substitution triplets.
That is the correct way to evaluate an embedding, but it is not the task the
product performs. The app is handed a partial recipe and asked what else
belongs. Nothing registered so far measures that, so H1-H8 could all resolve
cleanly while the app still felt useless.

**Protocol.** 100,000 recipes with >=3 ingredients held out whole (seed
20260806). The II graph was rebuilt from the remaining 4,547,847 recipes at the
*same* fixed threshold (NPMI >= -0.15, min_count 2) the main graph uses, giving
202,140 edges vs 203,504 -- the only difference between the two graphs is which
recipes were counted. One ingredient is hidden per test recipe; all 1,790
candidates are ranked from the ingredients that remain, excluding those already
present. Metric: recall@10 and MRR against a popularity control.

**H9.** Ranking by cosine to the mean of the retained ingredients beats a
popularity baseline by more than the ~0.02 seed-variance floor.

**Result (exploratory; cooc, aggregate over all 100k):**

| scorer | recall@10 | MRR | median rank | recall@50 |
|---|---|---|---|---|
| embed_mean | 0.3921 | 0.1906 | 18 | 0.7169 |
| popularity | 0.3690 | 0.1823 | 21 | 0.6661 |
| graph_npmi | 0.3632 | 0.1787 | 26 | 0.5968 |
| embed_max | 0.3455 | 0.1502 | 24 | 0.6736 |

H9 is **technically supported but substantively hollow**: +0.023 over a control
that ignores the query entirely, barely clearing the noise floor.

**The finding that matters is the slice.** The top-50 ingredients appear in
almost every recipe, so predicting them inflates every scorer including the
control. Restricting to the 35,461 test recipes (35.5%) whose hidden ingredient
is *not* a top-50 staple -- the cases where a suggestion is worth making:

| scorer | recall@10 | MRR | median rank |
|---|---|---|---|
| **graph_npmi** | **0.3097** | **0.1437** | **30** |
| embed_mean | 0.1186 | 0.0536 | 52 |
| popularity | 0.0000 | 0.0099 | 110 |

**A direct NPMI lookup beats the embedding by 2.6x on the cases that matter.**
The embedding's aggregate lead was entirely staples. (Popularity is structurally
0.0 here -- it can only ever propose top-50 items, which are excluded by
construction -- so the meaningful contrast is graph vs embedding.)

This comparison is *conservative in the embedding's favour*: `cooc` was trained
on the edge-level split, so it saw these test recipes, while `graph_npmi` used
the clean recipe-holdout graph. The leaky model lost to the clean baseline.

**Mechanism.** Compressing 1,790 ingredients into 300 dimensions smooths away
precisely the sharp, low-frequency associations that make a pairing
interesting, while preserving the dense staple structure that dominates the
loss. This is the same popularity concentration M5 measures and H3's whitening
failed to remove, now visible as a product-level failure rather than a
geometric curiosity.

**Popularity de-biasing at ranking time does not fix it.** Subtracting
lambda * z(log freq) from the score:

| lambda | recall@10 | top-10 mean log freq |
|---|---|---|
| 0 | 0.3921 | 12.406 |
| 0.02 | 0.3640 | 12.211 |
| 0.05 | 0.3089 | 11.824 |
| 0.1 | 0.1791 | 10.380 |
| 0.2 | 0.0008 | 1.548 |

2.8 recall points buy 0.2 nats of novelty; by lambda=0.2 the ranking has
degenerated into near-hapax ingredients. Novelty must come from candidate
*selection*, not from a score penalty.

**Confirmatory replication, rule fixed in advance.** `cooc-recipeholdout` is
training on `ii_graph_recipe_train.npz`, which never saw the test recipes. On
that clean model the pre-committed claim is: **graph_npmi retains a non-staple
recall@10 advantage over embed_mean exceeding 0.02.** If it does not, the
finding above is withdrawn and treated as a leakage artefact.

**Product consequence if confirmed.** Ship the NPMI graph as the primary
retrieval path and use the embedding only where the graph is silent -- unseen
pairs and the 242 cold-start ingredients H8 showed cannot be rescued from data.
An embedding-only app would be measurably worse at exactly the suggestions
users would find worth having.

### A5 addendum — hybrid ranker, the shippable algorithm (exploratory)

The app runs neither the graph nor the embedding alone, so the combination was
backtested directly: `score = sum_context NPMI(ctx, cand) + beta * cos(mean_ctx, cand)`.
The graph is authoritative where it has support; beta controls how far the
embedding may override real evidence.

| scorer | all r@10 | non-staple r@10 |
|---|---|---|
| hybrid b=2.0 | 0.4269 | 0.2998 |
| hybrid b=0.5 | 0.3909 | 0.3143 |
| hybrid b=0.1 | 0.3695 | 0.3118 |
| graph_npmi | 0.3632 | 0.3097 |
| embed_mean | 0.3921 | 0.1186 |

Two readings, and the second is the honest one:

1. Combining beats either component alone in aggregate: +0.035 over the
   embedding and +0.064 over the graph, both clearing the 0.02 noise floor.
2. On the non-staple slice the hybrid's margin over the pure graph is 0.0046 --
   **inside the noise floor**. The embedding is not adding retrieval signal
   where it matters; it is contributing staple ordering and coverage of pairs
   the graph never saw. The A5 conclusion stands.

beta trades the two slices against each other: 2.0 wins aggregate but is
*worse* than the bare graph on non-staples (0.2998 vs 0.3097), because letting
the embedding dominate reintroduces the popularity smoothing A5 identified.
**beta=0.5 is the pre-committed shipping default** -- the only setting in the
top tier on both slices. Still measured against the leaky embedding; the
confirmatory `cooc-recipeholdout` run may move the optimum, and beta will be
re-fit on that model before any bundle is frozen.

---

## Amendment A6 — LLM head-to-head (product claim test)

**Date:** 2026-02 · **Status: EXPLORATORY.** Not pre-registered before data were
seen; the non-staple stratification was chosen after observing the aggregate loss.
Recorded as exploratory and reported as such regardless of how favourable it is.

**Motivation.** The product's marketing claim is "better than Google/Claude." A
claim that can be checked by a user should be checked by us first.

**Method.** `tools/llm_benchmark.py`. Recipe completion with one ingredient hidden,
`SEED=20260806` shared with `backtest.py` so both systems are scored on identical
problems. Opponent `gpt-oss-120b` (Foundry, temp 0), prompt deliberately generous
(vocabulary format, worked example, explicit ranking instruction).

**Results.**

| slice | LLM r@10 | ours r@10 | LLM MRR | ours MRR |
|---|---|---|---|---|
| all (n=400) | **0.4225** ±0.048 | 0.3575 ±0.047 | **0.2627** | 0.1983 |
| non-staple (n=600) | 0.1350 ±0.027 | **0.2833** ±0.036 | 0.0520 | **0.1196** |

**Interpretation.** The claim as originally stated is **falsified for aggregate
recipe completion** — the LLM wins outside CI, and our ranker had the advantage of
a graph built including the test recipes. The claim **holds, decisively, on
non-staple completions**: 2.1× recall@10, 2.3× MRR, disjoint CIs. The mechanism is
legible — the LLM is reproducing what recipes usually contain, which is why it
degrades 3× once the answer is not a staple, while co-occurrence retrieval degrades
1.3×.

**Consequence.** The product claim is narrowed to what was measured: *"on
non-obvious ingredients, 2.1× more accurate than a frontier LLM, with evidence."*
Recorded in `fable-5/VALUE_PROP.md`, including the aggregate loss.

**Threats to validity.** Single opponent; 13% of LLM outputs unmappable to our
vocabulary and scored as misses (understates the LLM); "staple" = top-50 by
frequency, a chosen boundary. Falsification rule for any future confirmatory run:
if the non-staple recall@10 advantage does not exceed the 0.02 noise floor against
a second, larger model, the narrowed claim is withdrawn too.

**Instrumentation error, logged.** The first run scored the LLM at 0.000 because
`gpt-oss-120b` emits reasoning to `reasoning_content` and the answer to `content`;
`max_tokens=200` exhausted the budget on reasoning, returning empty `content` with
`finish_reason: "length"`. This would have been published as an infinite advantage.
Fixed by `max_tokens=4000` and a hard truncation check. Logged here because a
result favouring the experimenter warrants more scrutiny than one that does not.

---

## Amendment A7 — rank sweep: why the graph beats the embedding on the tail

**Date:** 2026-08-06 · **Status: EXPLORATORY** in outcome, but the three
predictions below were written down in `tools/rank_sweep.py` **before** the sweep
was run, so P1–P3 are genuine falsification tests rather than post-hoc narrative.

**Motivation.** A5 established that the exact NPMI graph beats the embedding 2.6×
on non-staple recipe completion. It did not explain *why*. Levy & Goldberg (2014)
show SGNS implicitly factorises the shifted PMI matrix, so the graph and the
embedding estimate the same quantity — one exactly and sparsely, one at rank d and
densely. Eckart–Young then suggests truncation preserves high-frequency (staple)
structure and discards the tail.

**Predictions (pre-stated).**
- P1: non-staple recall@10 rises monotonically with rank r.
- P2: aggregate recall@10 saturates at lower r than non-staple.
- P3: where the graph has no edge, the embedding rescues the case.

**Method.** `tools/rank_sweep.py`. SVD-truncate the *shipped* d=128 embedding to
r ∈ {2,…,128}, renormalise rows, re-score the identical 100k held-out recipes at
`SEED=20260806`. Truncating one trained model isolates rank while holding training
fixed; it cannot model how a lower-d model would reallocate capacity in training,
so this speaks to representational capacity, not training dynamics.

| r | var. expl. | all r@10 | non-staple r@10 |
|---|---|---|---|
| 2 | 0.379 | 0.0138 | 0.0135 |
| 8 | 0.483 | 0.1646 | 0.0699 |
| 16 | 0.555 | 0.2737 | 0.1290 |
| 32 | 0.656 | 0.3295 | **0.1386** |
| 64 | 0.804 | 0.3468 | 0.1384 |
| 128 | 1.000 | **0.3690** | 0.1270 |
| exact graph | — | 0.3632 | **0.3097** |

**Results: all three predictions falsified.**

- **P1 false.** Non-staple peaks at r=32 and *declines* to r=128 (−0.0116; 95%
  half-width 0.0050, paired test set, so outside the interval). Capacity beyond
  r≈32 buys aggregate accuracy by trading away the non-staple cases.
- **P2 false.** Aggregate rises throughout; non-staple is the metric that saturates.
- **P3 false.** The graph is silent in **26 of 100,000** cases (0.026%), and the
  embedding scores **0.000** on exactly those cases.

**Consequence — a correction to a claim already in our code.** `backtest.py`
justified β as ordering "candidates the graph is silent about (unseen pairs,
cold-start ingredients)." That justification is now measured and false. The
comment has been corrected. **β=0.5 is retained on narrower grounds:** it improves
aggregate recall 0.3632 → 0.3909 and costs nothing measurable on non-staples
(+0.0046, inside the 0.02 floor). Anyone proposing to raise β must read this first.

**What survives.** The mechanism hypothesis is dead but the phenomenon is
stronger without it: **no rank closes the gap.** The best embedding at any rank
reaches 0.1386 against the graph's 0.3097 — still 2.2×. The graph's advantage on
the tail is not a truncation artifact recoverable with more dimensions.

**Threats to validity.** Truncation ≠ training at lower d (stated above). Single
embedding, single corpus. The r=32 peak is modest (~3σ) and should not be used to
argue for shipping d=32 without a confirmatory run — d=128 remains the shipped
choice under H7 because it wins aggregate and the non-staple difference is small.

---

## Amendment A8 — β re-fit on a leakage-free embedding (2026-08-06, pre-stated)

**Why.** A7 kept β=0.5 on the narrow ground that it lifts aggregate recall
0.3632 → 0.3909 at no measurable non-staple cost. That measurement used the
`cooc-full` embedding, which was **trained on every recipe including the 100,000
held out by `backtest.py build`**. The graph side of that backtest is clean —
`build` rebuilds co-occurrence from the training recipes only — but the
embedding side was not. Every number attributed to β therefore has an unknown
amount of memorisation in it, and β is the one parameter whose entire
justification rests on those numbers.

`epicure-cooc-recipeholdout` (Azure, completed) trained the same architecture on
the same corpus **minus the held-out recipes**, d=300. Re-running the identical
backtest against it isolates leakage: same test set, same graph, same scorers,
only the embedding's training data differs.

**Predictions, stated before running.**

- **P1.** `embed_mean` aggregate recall@10 falls by **more than 0.02** from
  0.3921. The embedding-alone result is the most leakage-exposed number we have.
- **P2.** `hybrid_b2.0`'s aggregate advantage over `hybrid_b0.5`
  (0.4269 vs 0.3909) **shrinks by at least half**, because β=2.0 weights the
  leaked signal most heavily.
- **P3.** Non-staple `hybrid_b0.5` stays within **0.01** of graph-only
  (0.3143 vs 0.3097). β's non-staple neutrality was near-zero to begin with and
  should not be leakage-dependent.

**Decision rule, stated before running.** If with the clean embedding β=0.5 no
longer beats graph-only on aggregate recall@10 by more than **0.01**, we drop β
to 0 and ship graph-only: the embedding would then be paying for a dependency,
a shipped 917 KB matrix, and a per-query GEMV it does not earn. Otherwise β=0.5
stands and this amendment records its first honest measurement.

**Threat we cannot remove.** `cooc-recipeholdout` is d=300 and the shipped
bundle is d=128, so a difference could be dimension rather than leakage. The
comparison is therefore run against `cooc-full` at its native d=300, which is
the matched control; the d=128 shipped cost is already quantified in A7.

**Results — P1 and P2 falsified, P3 confirmed. β=0.5 stands.**

Same 100,000 test recipes, same graph (`ii_graph_recipe_train.npz`), same
scorers. Only the embedding's training data differs.

| scorer | leaky (cooc-full) | clean (cooc-recipeholdout) | Δ | non-staple Δ |
|---|---|---|---|---|
| embed_mean | 0.3921 | **0.3985** | **+0.0064** | +0.0110 |
| graph_npmi | 0.3632 | 0.3632 | 0.0000 | 0.0000 |
| popularity | 0.3690 | 0.3690 | 0.0000 | 0.0000 |
| hybrid β=0.1 | 0.3695 | 0.3698 | +0.0003 | 0.0000 |
| **hybrid β=0.5** | 0.3909 | **0.3911** | **+0.0002** | 0.0000 |
| hybrid β=2.0 | 0.4269 | 0.4249 | −0.0020 | −0.0025 |

- **P1 false, and false in the direction that matters.** We predicted
  `embed_mean` would fall by more than 0.02 once the test recipes were removed
  from training. It **rose** by 0.0064 aggregate and 0.0110 non-staple. There is
  **no detectable recipe-level leakage** in the embedding.
- **P2 false.** β=2.0's aggregate advantage over β=0.5 shrank from 0.0360 to
  0.0338 — a 6% change, not the predicted 50%. β=2.0's edge is not memorisation.
- **P3 confirmed.** Non-staple β=0.5 is 0.3143 against graph-only 0.3097
  (+0.0046), unchanged to four decimal places by removing the leakage.

**Decision rule outcome.** β=0.5 beats graph-only on aggregate by **0.0279**,
well above the pre-stated 0.01 threshold. **β=0.5 is retained**, and for the
first time on a measurement with no embedding-side leakage in it.

**Why leakage was negligible, and why that matters more than the β result.**
The held-out 100,000 recipes are 2.15% of a 4,647,847-recipe corpus. An
embedding trained on the remaining 97.85% learns essentially the same aggregate
co-occurrence statistics, because that is the only thing it can learn at this
scale — it is not memorising individual recipes. This **retroactively clears
every embedding number in the study**, including A7's rank sweep, of the
leakage objection. The strongest version of that objection is now measured and
answered rather than argued.

**A caveat this run creates for A7.** Two independently trained d=300 embeddings
on the same corpus differ by **0.0110** on the non-staple slice. A7's r=32 peak
was −0.0116 relative to r=128 — the same magnitude. A7's measurement is still
sound *as stated*, because it truncated a single embedding and is therefore
paired with no seed variance (half-width 0.0050). But the conclusion "r≈32 is
the optimum" must be read as a property of that embedding's spectrum, not an
architectural constant: separating those would need several seeds per rank. The
existing instruction not to ship d=32 on the strength of A7 is unchanged and now
better justified.

**On shipping a model trained on everything.** The bundle ships `cooc-full`,
which saw all 4.65M recipes including the test set. That is the correct
production choice — a shipped model should use all available data — and it is
not what the holdout run is for. The holdout run exists to establish that the
metrics we *quote* are not inflated by that decision. This amendment establishes
it: the difference is +0.0002 on the shipped configuration. No re-export needed.
