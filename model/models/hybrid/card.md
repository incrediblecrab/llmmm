# Model card — `hybrid`

## What it is

Post-hoc combinations of spaces that were trained separately. Inputs are run
IDs, so any two runs from any two families can be combined without either
knowing the other exists.

| model | method |
|---|---|
| `concat` | length-normalise each space, weight it, stack |
| `residual` | regress the correction onto the base, append only what is left over |

```bash
im train concat --set runs=svd-ppmi-recipe-holdout-s0,ease-recipe-holdout-s0
im train residual --set base=sgns-cooc-recipe-holdout-s0 \
                  --set correction=ease-recipe-holdout-s0
```

## Why this family exists

Two facts from the prior study sit awkwardly together: raw co-occurrence
statistics rank held-out pairs better than any learned embedding (AUC 0.984 vs
0.873), yet embeddings say something about pairs that have never co-occurred,
where counting is silent. Neither dominates, and choosing one discards what the
other knows.

## Reading the output

`residual` prints **how much of the correction space the base already
explains**. This is the number to look at, not the metrics. If the base explains
95% of the correction, the two models have learned the same thing and blending
them is not going to produce a third result — that is a finding about the
models, and a cheaper one than the leaderboard row.

`strength=0` reduces `residual` to exactly the base space. That is the control:
any improvement must be measured against it, not against chance.

## Caveats

- **Normalising before stacking is load-bearing.** A PPMI SVD carries singular
  values in the hundreds; a trained SGNS table sits near unit norm. Stacking
  them raw is not a blend, it is whichever space had larger numbers.
- Dimensionality reduction happens *after* stacking, so the projection is chosen
  with both spaces in view. Reducing each first discards directions that only
  look redundant in isolation.
- These models are fitted on the full vocabulary with no held-out data of their
  own, so they cannot overfit in the usual sense — but a blend tuned by watching
  the leaderboard is fitted to the test set by hand. Choose weights from the
  residual diagnostic, not by sweeping until M4 peaks.
