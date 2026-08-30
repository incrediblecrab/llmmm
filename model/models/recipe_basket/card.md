# Model card — `recipe_basket`

## What it is

Models that read the 4.6M-recipe corpus directly, as implicit feedback, instead
of consuming the pairwise ingredient–ingredient graph.

| model | method | cost |
|---|---|---|
| `ease` | closed-form item-item ridge autoencoder (Steck 2019) | seconds |
| `ials` | implicit alternating least squares (Hu/Koren/Volinsky 2008) | ~20–40 min |
| `item2vec` | skip-gram over baskets, no graph intermediary | ~30–60 min |

## Why this family exists

Every other family consumes the ingredient–ingredient graph, which is a
*pairwise summary*. A recipe is a set someone chose as a whole; reducing it to
pairs discards the higher-order structure that made it a set. These models are
the test of whether that discarded structure was worth anything.

`item2vec` is the tightest comparison in the workspace: it has the identical
objective to `sgns-cooc` with the graph removed, so any difference between them
is attributable to the NPMI re-weighting and to nothing else.

`ease` is the one most likely to break the popularity degeneration that affects
cosine-on-counts, because inverting the Gram matrix makes its scores
*conditional* — it divides out ingredients that co-occur only because both
co-occur with salt.

## Key parameters

| name | default | note |
|---|---|---|
| `reg` | 250.0 | `ease` ridge; the only knob, and it matters — sweep it |
| `d_model` | 300 / 128 | `ials` defaults lower; recipe factors dominate memory |
| `alpha` | 40.0 | `ials` confidence on observed entries |
| `max_recipes` | 0 / 400k | 0 means all; `ials` subsamples because recipe factors are held in memory |
| `subsample` | 1e-4 | `item2vec` frequent-ingredient downsampling |

## Caveats

- **The embedding is not the model — measured.** `ease` returns an asymmetric
  score matrix `B`. The harness needs vectors, so `B` is symmetrised and
  truncated to its positive eigenvalues, retaining only ~47% of the spectrum.
  The cost is large and specific:

  | M6 recall@10 | |
  |---|---|
  | through the exported embedding | 0.166 — **loses** to the popularity baseline |
  | through `B`, the real prediction rule | **0.593** — beats it by +0.22 |

  Judged on its embedding alone `ease` looks worse than recommending onion and
  salt to everyone. It supplies a native `scorer`, so M6 reports both; use the
  saved `item_scores.npy` for anything that consumes the model directly.
- **Regularisation barely matters.** A sweep over `reg` from 1.0 to 5000 moves
  M6 only between 0.5849 and 0.5932, peaking around 50. The Gram matrix built
  from 3.25M recipes is well enough conditioned that the ridge term has little
  to do. The untuned default cost nothing — but that is now measured rather than
  assumed (`experiments/ease-regularisation.yaml`).
- `ials` subsamples recipes by default. Ingredient factors converge well before
  the recipe count is exhausted, but a sweep over `max_recipes` is the way to
  confirm that rather than assume it.
