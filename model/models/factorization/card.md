# Model card — `factorization`

## What it is

Closed-form and full-batch factorisations of the co-occurrence matrix. At
n = 1,790 the matrix is 1790×1790, so an exact decomposition takes seconds.

| model | factors | cost |
|---|---|---|
| `svd-ppmi` | shifted positive PMI of the co-occurrence matrix | seconds |
| `glove` | weighted log-count objective, full-batch AdaGrad | minutes |
| `chem-svd` | IDF-weighted ingredient × compound incidence | seconds |

## When to use it

Two purposes:

1. **Control.** These cannot collapse by construction, so if a random-walk model
   has a low participation ratio and its factorisation counterpart does not, the
   collapse belongs to the walk schema, not to the data.
2. **Iteration speed.** If a factorisation matches SGNS, retraining is seconds
   rather than hours, and embeddings can be rebuilt continuously on new data.

## Known behaviour

Prior work found factorisation does *not* reach SGNS on this data — the
random-walk sampling contributes something the co-occurrence matrix alone does
not capture. `svd-ppmi` has by far the highest participation ratio of any model
here while scoring below `sgns-cooc` on held-out link prediction, which is the
clearest single demonstration that participation ratio is a health check and not
a quality score.

## Cost

`svd-ppmi` and `chem-svd` are cheap enough to run on every code change.
`glove` is moderate: 200 full-batch epochs over ~366k non-zeros.
