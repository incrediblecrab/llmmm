# Model card — `sgns_walk`

## What it is

Skip-gram with negative sampling trained on random walks over ingredient graphs.
Three registered models share one architecture and differ only in which edges
the walk may traverse.

| model | walks over | reads as |
|---|---|---|
| `sgns-cooc` | ingredient–ingredient NPMI graph | what people cook together |
| `sgns-chem` | typed ingredient→compound→ingredient metapaths | shared flavour compounds |
| `sgns-core` | the union, ingredient–ingredient up-weighted by `ii_repeat` | a controlled blend |

## When to use it

`sgns-cooc` is the reference baseline. Every new family should be compared
against it before it is taken seriously, because it is the strongest model that
prior work found on held-out link prediction.

## Key parameters

| name | default | note |
|---|---|---|
| `d_model` | 300 | width of the shipped table — a product decision as much as a modelling one |
| `epochs` | 20 | from the published configuration; not tuned |
| `walks_per_node` | 100 | |
| `walk_length` | 50 | |
| `context_size` | 7 | sliding window; first node is the centre |
| `negative_samples` | 5 | drawn per centre from unigram^0.75 |
| `ii_repeat` | 10.0 | `sgns-core` only; 0 ≈ chem, large ≈ cooc |

## Known behaviour

- `sgns-chem` collapses: effective dimensionality near 3 and held-out link AUC
  at chance. This is a property of the walk schema, not a bug — chemistry-only
  walks cannot express ingredient–ingredient structure. `ii_repeat` traces the
  curve between the two regimes smoothly, which is how the collapse was
  attributed to the schema rather than to the optimiser.
- Nearest neighbours contain near-duplicates (`rice → brown_rice`,
  `lamb → mutton`). Any product surface needs a dedup filter; that is a
  presentation concern, not a modelling defect.

## Cost

Moderate — roughly 10–25 minutes on 2 CPU vCPUs at default settings.
