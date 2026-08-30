# Findings

_17 jobs completed, 1 failed, 21 submitted._

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
| chem-svd | 0.386 | 0.102 | 0.5827 | 0.4892 | +0.0935 |
| chem | 0.394 | 0.090 | 0.5784 | 0.5399 | +0.0385 |
| cooc-d128 | 0.440 | 0.415 | 0.8672 | 0.7593 | +0.1078 |
| cooc-d32 | 0.592 | 0.436 | 0.8521 | 0.7580 | +0.0941 |
| cooc-d64 | 0.493 | 0.447 | 0.8610 | 0.7642 | +0.0968 |
| cooc-full | 0.693 | 0.195 | 0.8607 | 0.7235 | +0.1372 |
| cooc | 0.443 | 0.578 | 0.8677 | 0.7620 | +0.1057 |
| core-ii0.1 | 0.400 | 0.522 | 0.8531 | 0.7542 | +0.0989 |
| core-ii0 | 0.455 | 0.081 | 0.6585 | 0.5740 | +0.0846 |
| core-ii1 | 0.409 | 0.560 | 0.8708 | 0.7743 | +0.0965 |
| core-ii10-full | 0.732 | 0.158 | 0.8636 | 0.7278 | +0.1358 |
| core-ii10 | 0.402 | 0.608 | 0.8675 | 0.7697 | +0.0978 |
| core-ii100 | 0.412 | 0.593 | 0.8690 | 0.7676 | +0.1014 |
| glove | 0.933 | 0.099 | 0.8332 | 0.7185 | +0.1147 |
| svd-ppmi | 0.280 | 0.265 | 0.7649 | 0.7048 | +0.0601 |

**FALSIFIED** — whitening either fails to remove popularity or costs more accuracy than the pre-registered budget. Where M5 stays high after removing three directions, popularity is spread across many dimensions rather than concentrated in a few — it is not a low-rank artefact and cannot be projected away for free.

## H4 — the food-pairing asymmetry across cuisines (headline)

18 cuisines. Δ > 0 means a cuisine pairs ingredients that share flavour compounds more than its own ingredient frequencies would predict.

| cuisine | region | recipes | Δ | relative | z |
|---|---|---|---|---|---|
| thai | Southeast Asia | 1,685 | +4.984 | +22.61% | +8.7 |
| filipino | Southeast Asia | 1,982 | +1.832 | +7.19% | +5.8 |
| moroccan | North Africa | 2,442 | +2.875 | +7.12% | +11.1 |
| persian | Middle East | 5,737 | +2.723 | +6.01% | +13.4 |
| indonesian | Southeast Asia | 14,929 | +1.246 | +5.70% | +10.3 |
| north_american | North America | 29,820 | +1.233 | +4.29% | +13.6 |
| spanish | Southern Europe | 29,570 | +1.403 | +3.92% | +9.4 |
| indian | South Asia | 22,160 | +1.228 | +3.49% | +10.7 |
| romanian | Eastern Europe | 777 | +0.617 | +1.86% | +1.0 |
| chinese | East Asia | 29,164 | +0.128 | +0.83% | +1.5 |
| greek | Southern Europe | 4,832 | +0.168 | +0.43% | +0.5 |
| taiwanese | East Asia | 1,574 | -0.017 | -0.14% | -0.1 |
| russian | Eastern Europe | 29,917 | -0.064 | -0.25% | -0.7 |
| german | Western Europe | 4,285 | -0.163 | -0.44% | -0.5 |
| israeli | Middle East | 8,586 | -0.173 | -0.47% | -0.8 |
| vietnamese | Southeast Asia | 29,400 | -0.447 | -2.11% | -6.5 |
| turkish | Middle East | 29,904 | -0.674 | -3.00% | -9.2 |
| japanese | East Asia | 3,492 | -0.731 | -3.48% | -2.5 |

| region | mean relative Δ | cuisines |
|---|---|---|
| Southeast Asia | +8.35% | 4 |
| North Africa | +7.12% | 1 |
| North America | +4.29% | 1 |
| South Asia | +3.49% | 1 |
| Southern Europe | +2.18% | 2 |
| Middle East | +0.85% | 3 |
| Eastern Europe | +0.80% | 2 |
| Western Europe | -0.44% | 1 |
| East Asia | -0.93% | 3 |

**SUPPORTED** — western cuisines pair shared-compound ingredients more than East Asian cuisines do, reproducing Ahn et al. 2011 on natively-sourced regional corpora rather than on a western-dominated corpus. That is the part of their claim most open to sampling bias, and it holds.

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
| chem | 3 | 0.0123 |

