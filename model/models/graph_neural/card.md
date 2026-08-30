# Model card — `graph_neural`

## What it is

Models that learn a *propagation rule* rather than a free embedding table. An
ingredient's representation is assembled from its neighbours', so graph
structure enters at inference as well as during training.

| model | method | cost |
|---|---|---|
| `sgc` | k-step propagation of a spectral base, no parameters | seconds |
| `lightgcn` | layer-averaged linear propagation fitted with BPR | ~5–15 min |

## Why this family exists

`sgc` is the control that keeps `lightgcn` honest. SGC's claim is that the depth
of a graph network buys almost nothing beyond the smoothing its propagation
performs. If `lightgcn` does not beat `sgc` here, its parameters are not earning
their cost and the family reduces to a matrix power.

## Key parameters

| name | default | note |
|---|---|---|
| `layers` | 3 / 2 | propagation depth; more means a wider neighbourhood and more smoothing |
| `weight` | `npmi` | `npmi` or `count`; `count` makes the operator a popularity diffusion |
| `d_model` | 300 | |
| `epochs` | 60 | `lightgcn` only |

## Caveats

- Symmetric normalisation is not optional. With row normalisation or none at
  all, three layers of propagation converge on the graph's dominant
  eigenvector — which is to say, they return salt.
- `lightgcn` exports the **propagated** representation, not the free parameter
  table, because that is what the model scores with. The base table is saved
  separately as `base_table.npy` for inspection.
- Both are transductive: an ingredient with no training edge gets only its
  self-loop, so the 261 isolated vocabulary terms stay near their
  initialisation. That is a property of the graph, not of the method.
