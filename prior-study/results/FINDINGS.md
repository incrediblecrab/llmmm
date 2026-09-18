# Findings

_26 jobs completed, 1 failed, 27 submitted._

Verdicts are mechanical against thresholds fixed in `docs/PREREGISTRATION.md` before any model ran.
Random-vector control: M2 0.5017, M4 0.4932. 95% CI on both is ±0.0069, so smaller gaps are not differences.

## H1 — the chem collapse is structural, not a training bug

One knob (`ii_repeat`) interpolates from Chem's pure ingredient→compound
schema (0) to Cooc's pure ingredient–ingredient schema (∞). If collapse were
an optimiser or hardware artefact it would not track this knob.

| ii_repeat | M1 PR | M2 broad | M4 AUC |
|---|---|---|---|
| 0 | 41.6 | 0.6585 | 0.5157 |
| 0.1 | 84.7 | 0.8531 | 0.7768 |
| 1 | 109.2 | 0.8708 | 0.8278 |
| 10 | 117.2 | 0.8675 | 0.8408 |
| 100 | 118.8 | 0.8690 | 0.8374 |
| _chem_ | 2.7 | 0.5784 | 0.5198 |
| _cooc_ | 115.1 | 0.8677 | 0.8432 |

**SUPPORTED** — PR and held-out AUC both rise with I-I mixing, so the collapse is a property of the walk schema — chemistry-only walks cannot express ingredient–ingredient structure.

Chem alone sits at PR 2.7 with held-out AUC 0.5198 against a 0.50 chance and a 0.4932 random control — it is close to uninformative about which ingredients actually co-occur.

## H2 — a closed-form factorisation matches SGNS at this scale

| model | M1 PR | M2 broad | M4 AUC | vs SGNS (M2) |
|---|---|---|---|---|
| cooc | 115.1 | 0.8677 | 0.8432 | +0.0000 |
| svd-ppmi | 222.7 | 0.7649 | 0.7410 | -0.1028 |
| glove | 12.9 | 0.8332 | 0.7795 | -0.0345 |
| chem-svd | 140.4 | 0.5827 | 0.5132 | -0.2850 |

**FALSIFIED** — factorisation does not reach SGNS; the random-walk sampling contributes something the co-occurrence matrix alone does not capture.

## H3 — popularity degeneration is low-rank and removable

Pre-registered test: removing the top 3 principal directions should push
M5 below 0.3 while costing M2 no more than 0.02.

| model | M5 before | M5 after | M2 before | M2 after | M2 cost |
|---|---|---|---|---|---|
| chem-s1 | 0.397 | 0.074 | 0.5906 | 0.5420 | +0.0486 |
| chem-s2 | 0.394 | 0.079 | 0.5827 | 0.5533 | +0.0294 |
| chem-svd | 0.386 | 0.102 | 0.5827 | 0.4892 | +0.0935 |
| chem | 0.394 | 0.090 | 0.5784 | 0.5399 | +0.0385 |
| cooc-d128 | 0.440 | 0.415 | 0.8672 | 0.7593 | +0.1078 |
| cooc-d300-b16k | 0.334 | 0.667 | 0.8636 | 0.7621 | +0.1015 |
| cooc-d32 | 0.592 | 0.436 | 0.8521 | 0.7580 | +0.0941 |
| cooc-d600-b16k | 0.323 | 0.670 | 0.8586 | 0.7526 | +0.1059 |
| cooc-d64 | 0.493 | 0.447 | 0.8610 | 0.7642 | +0.0968 |
| cooc-full | 0.693 | 0.195 | 0.8607 | 0.7235 | +0.1372 |
| cooc-recipeholdout | 0.434 | 0.595 | 0.8692 | 0.7743 | +0.0949 |
| cooc-s1 | 0.448 | 0.557 | 0.8677 | 0.7604 | +0.1072 |
| cooc-s2 | 0.435 | 0.581 | 0.8726 | 0.7656 | +0.1070 |
| cooc | 0.443 | 0.578 | 0.8677 | 0.7620 | +0.1057 |
| core-ii0.1 | 0.400 | 0.522 | 0.8531 | 0.7542 | +0.0989 |
| core-ii0 | 0.455 | 0.081 | 0.6585 | 0.5740 | +0.0846 |
| core-ii1 | 0.409 | 0.560 | 0.8708 | 0.7743 | +0.0965 |
| core-ii10-full | 0.732 | 0.158 | 0.8636 | 0.7278 | +0.1358 |
| core-ii10-s1 | 0.390 | 0.598 | 0.8662 | 0.7718 | +0.0944 |
| core-ii10-s2 | 0.390 | 0.605 | 0.8713 | 0.7714 | +0.0999 |
| core-ii10 | 0.402 | 0.608 | 0.8675 | 0.7697 | +0.0978 |
| core-ii100 | 0.412 | 0.593 | 0.8690 | 0.7676 | +0.1014 |
| glove | 0.933 | 0.099 | 0.8332 | 0.7185 | +0.1147 |
| svd-ppmi | 0.280 | 0.265 | 0.7649 | 0.7048 | +0.0601 |

**FALSIFIED** — whitening either fails to remove popularity or costs more accuracy than the pre-registered budget. Where M5 stays high after removing three directions, popularity is spread across many dimensions rather than concentrated in a few — it is not a low-rank artefact and cannot be projected away for free.

## H4 — the food-pairing asymmetry across cuisines (headline)

18 cuisines, `cuisine_pairing_v2` (no cuisine capped, 12 null replicates, 4,485,919 recipes analysed). Δ > 0 means a cuisine pairs ingredients that share flavour compounds more than its own ingredient frequencies would predict.

The interval is a recipe-level 95% CI on Δ. The earlier z column divided Δ by the scatter of the *null* replicates, which ignores sampling error in the observed statistic and therefore grows without bound as the corpus grows.

| cuisine | region | recipes | Δ | relative | 95% CI | sign |
|---|---|---|---:|---:|---|---|
| thai | Southeast Asia | 1,685 | +5.065 | +23.07% | [+3.693, +6.437] | resolved |
| moroccan | North Africa | 2,442 | +2.996 | +7.44% | [+2.293, +3.699] | resolved |
| filipino | Southeast Asia | 1,982 | +1.603 | +6.23% | [+0.888, +2.319] | resolved |
| persian | Middle East | 5,737 | +2.767 | +6.11% | [+2.035, +3.499] | resolved |
| indonesian | Southeast Asia | 14,929 | +1.221 | +5.58% | [+0.955, +1.488] | resolved |
| north_american | North America | 2,655,755 | +1.202 | +4.19% | [+1.174, +1.230] | resolved |
| spanish | Southern Europe | 45,803 | +1.293 | +3.62% | [+1.071, +1.515] | resolved |
| indian | South Asia | 22,160 | +1.184 | +3.36% | [+0.905, +1.464] | resolved |
| chinese | East Asia | 1,393,014 | +0.236 | +1.54% | [+0.205, +0.268] | resolved |
| romanian | Eastern Europe | 777 | +0.366 | +1.10% | [-1.449, +2.181] | **unresolved** |
| greek | Southern Europe | 4,832 | +0.244 | +0.63% | [-0.405, +0.892] | **unresolved** |
| russian | Eastern Europe | 162,302 | -0.006 | -0.02% | [-0.105, +0.093] | **unresolved** |
| israeli | Middle East | 8,586 | -0.020 | -0.05% | [-0.562, +0.522] | **unresolved** |
| german | Western Europe | 4,285 | -0.208 | -0.57% | [-0.808, +0.392] | **unresolved** |
| taiwanese | East Asia | 1,574 | -0.210 | -1.61% | [-1.308, +0.888] | **unresolved** |
| vietnamese | Southeast Asia | 31,241 | -0.479 | -2.25% | [-0.666, -0.292] | resolved |
| japanese | East Asia | 3,492 | -0.568 | -2.73% | [-1.225, +0.089] | **unresolved** |
| turkish | Middle East | 125,323 | -0.647 | -2.85% | [-0.748, -0.545] | resolved |

| region | mean relative Δ | cuisines |
|---|---|---|
| Southeast Asia | +8.16% | 4 |
| North Africa | +7.44% | 1 |
| North America | +4.19% | 1 |
| South Asia | +3.36% | 1 |
| Southern Europe | +2.13% | 2 |
| Middle East | +1.07% | 3 |
| Eastern Europe | +0.54% | 2 |
| Western Europe | -0.57% | 1 |
| East Asia | -0.93% | 3 |

Pre-registered per-stratum signs (the registered test):

| cuisine | Ahn predicts | observed | |
|---|---|---:|---|
| spanish | + | +3.62% | agree |
| german | + | -0.57% | **disagree** |
| greek | + | +0.63% | agree |
| romanian | + | +1.10% | agree |
| russian | + | -0.02% | **disagree** |
| chinese | − | +1.54% | **disagree** |
| japanese | − | -2.73% | agree |
| thai | − | +23.07% | **disagree** |
| vietnamese | − | -2.25% | agree |
| filipino | − | +6.23% | **disagree** |
| indonesian | − | +5.58% | **disagree** |

Agree 5, disagree 6 of 11.
Signs not resolved by their own CI: german, greek, japanese, romanian, russian.

Scored instead against Ahn's own five regions (which place Southern European with East Asian on the avoiding side): agree 3, disagree 4.

### Robustness

**Duplicate recipes.** The scraped corpora overlap (RecipeNLG re-publishes food.com), so recipes repeat and pseudo-replicate the CI. Collapsing recipes that share an ingredient set drops 1,247,691 of 4,541,640 (27.5%) and re-runs the whole test (`cuisine_pairing_v2_dedup`): agree 5, disagree 6. The registered verdict is unchanged.

Signs that flip under deduplication: greek, russian (of which resolved in the main run: none).

The secondary scoring against Ahn's own regions is **not robust**: it reads agree 3/disagree 4 on the full corpus but agree 4/disagree 3 after deduplication, so its majority is decided by signs their own CIs cannot resolve. Only the pre-registered scoring above should be relied on.

**Flavour-database coverage.** Only 489 of 1,790 vocabulary ingredients carry any compound, and a pair contributes nothing unless both ends are covered, so per-cuisine pair coverage spans 41%–81%. If Δ merely tracked how well FlavorDB covers a cuisine's larder the contrast would be a property of the database. It does not: Spearman r = +0.079 between relative Δ and pair coverage across 18 cuisines (permutation p = 0.75).

**Trivial ingredient couplings.** Palermo et al. 2024 (arXiv:2406.15533) report that food pairing is "mostly due to trivial couplings of very similar ingredients". Masking the 6,045 pairs that are the same food twice (one name contains the other, or compound Jaccard ≥ 0.9) and re-scoring against byte-identical null draws (`trivial_pairs_audit`) moves relative Δ by at most 0.38%, flipping only israeli, whose Δ is within noise of zero. The effect is not an artefact of vocabulary granularity.


**FALSIFIED** — signs disagree with Ahn for a majority of strata (6 of 11), which is the pre-registered falsification condition. The clearest single result is Chinese: Ahn et al.'s East Asian avoidance claim rests on 2,512 recipes in total (their Table S2, Korean + Chinese + Japanese), while Chinese alone here is 1.39M recipes — roughly 550× the sample — and sits significantly on the *pairing* side. The strongest effect in the corpus is Thai, a Southeast Asian cuisine predicted to avoid shared compounds. Two caveats bound this: the corpus has no Korean data at all, though Korean is inside Ahn's East Asian group, and Japanese and Taiwanese are too small here to resolve, so 'East Asia' is not adjudicated as a bloc — only Chinese is. Note also that 41,525 of Ahn's 56,498 recipes are North American, the one regional claim that does reproduce here (+4.2%, stable under deduplication); every other regional claim of theirs rested on ≤4,180 recipes.

## H5 — does the chemistry graph add value?

Pre-registered: a chemistry-informed model should beat pure co-occurrence on held-out link AUC by more than 0.02.

| model | M4 AUC | vs cooc |
|---|---|---|
| cooc | 0.8432 | +0.0000 |
| chem | 0.5198 | -0.3234 |
| chem-svd | 0.5132 | -0.3300 |
| core-ii1 | 0.8278 | -0.0154 |
| core-ii10 | 0.8408 | -0.0025 |
| core-ii100 | 0.8374 | -0.0058 |

**FALSIFIED** — no chemistry-informed model beats pure co-occurrence by the pre-registered margin. FlavorDB can be dropped, which removes a licensing risk from the product without costing measured quality.

## Seed variance

A difference smaller than this spread is not a result.

| model | seeds | M2 spread |
|---|---|---|
| chem | 4 | 0.0123 |
| cooc | 3 | 0.0050 |
| core-ii10 | 3 | 0.0051 |

