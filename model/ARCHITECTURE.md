# Architecture

## The problem this shape solves

The prior study produced 26 experiments across a `tools/` directory of 60 flat
scripts. That works while one person holds the whole thing in their head, and
stops working the moment a model type needs to be added, an evaluation needs to
be re-run under a corrected protocol, or someone asks which of two numbers is
comparable.

Three decisions follow from that, and everything else is detail.

### 1. A model type is a folder, not an entry in a list

`ingredient_model/registry.py` walks `models/` with `pkgutil` at import time.
A new family needs a new directory and a `@register` decorator; there is no
central list to update, so there is no central list to drift.

The cost is that discovery imports every compartment, so a broken one breaks
`im list` rather than only itself. That is the intended trade: a compartment
that does not import is broken, and finding out at `im list` is better than
finding out three hours into a sweep. This actually happened during development
and the loud failure was correct.

### 2. There is one contract, and it is small

```python
TrainContext  -> params, seed, graph, corpus, split, out_dir, device
TrainResult   -> embedding, metadata, extra_arrays, scorer
```

The embedding is the common currency: a random walk model, a matrix
factorisation and a transformer all reduce to one `(n_vocab, d)` array, so one
harness scores all of them on identical terms.

`scorer` is the escape hatch, and it is not cosmetic. For families whose model
is not literally a lookup table, the embedding is a lossy export — EASE scores
0.166 through its embedding and 0.593 through its real prediction rule. Both are
reported: the embedding number is what makes families comparable, the native
number is what would actually be served. Neither replaces the other.

`extra_arrays` carries whatever else a family produces — item biases, context
tables, attention weights — saved beside the embedding without the harness
needing to know it exists.

### 3. Soundness is enforced in code, not in a document

`data/splits.py` is the scientific core. It encodes which evaluation protocol is
valid for which model, and `check_leakage()` **raises**.

The rationale is in the docstring and worth repeating: *an optimistic number
that is merely flagged still ends up quoted.* A warning is read once, by the
person who already knows; the number outlives the warning.

The same reasoning drives the default. `DEFAULT_SPLIT` is `recipe-holdout` —
the protocol valid for every family — not the one that reproduces the most prior
numbers. A default that is unsound for some models produces uncomparable
leaderboards whenever someone omits the flag, and the inflated row is the one
that gets screenshotted. `tests/test_splits.py` asserts this, and caught it when
the default was still `edge-holdout`.

## Data flow

```
recipes (4.6M)  ──split by recipe──►  train corpus  ──►  recipe models
      │                                     │
      │                                rebuild graph
      ▼                                     ▼
  held-out recipes ──► M6            train graph ──►  graph models
                                            │
                                     held-out edges ──► M4
```

Both arms derive from a single recipe-level split, which is what makes the two
families comparable at all. The graph is *rebuilt* from the training recipes
rather than being edge-filtered, because an edge-filtered graph still encodes
statistics computed from held-out recipes.

### A limit that cannot be engineered away

Held-out edges under the recipe protocol are **rare pairs by construction**. A
pair appearing in millions of recipes cannot be hidden without deleting most of
the corpus. Honest recipe-level link prediction is therefore necessarily a
rare-pair task, and M4 values on `recipe-holdout` are not comparable to M4 on
`edge-holdout`. This is a property of the data, not a defect in the split.

The holdout fraction is **0.30**, chosen empirically rather than by convention.
Held-out edges grow far more slowly than recipes removed:

| fraction | held-out edges |
|---|---|
| 0.10 | 6,637 — CI roughly 2× too wide |
| 0.20 | 13,648 |
| **0.30** | **20,980** — matches the edge protocol's 20,350 within 3% |
| 0.40 | 29,206 |

0.30 is the smallest fraction with enough statistical power to compare against
the prior study's numbers.

## Evaluation

`eval/harness.py` builds one `EvalContext` per split, LRU-cached, so every model
scored against a given split sees **identical** negative samples. Without that,
two runs differ by their draw and small differences are noise dressed as
findings.

`control_gate()` scores pure noise and blocks everything. A metric that rates
random vectors above chance is measuring an artefact, and every number computed
with it afterwards is void. It is a gate, not a formality — `make check` runs it
before the tests.

## Reasoning

`reasoning/` is deliberately not a model. It blends three witnesses —
co-occurrence, embedding cosine, chemistry — weighted by measured reliability,
and reports where they **disagree** instead of averaging the disagreement away.

Co-occurrence carries the largest weight because it is the strongest signal
(AUC 0.984 vs the best embedding's 0.873). Chemistry carries **zero**, because
H5 was falsified. Keeping a rejected signal at zero weight records that it was
tested; deleting it would not.

`reasoning/dedup.py` is a product rule, not a model concern, which is why it
lives here. `rice → brown_rice` is the model working correctly and the product
failing. Detection is lexical on purpose: a learned duplicate detector would be
fitted on the same co-occurrence statistics that create the problem and would
confidently agree the duplicates are distinct.

The rule is head-matching plus a vocabulary check — `brown_rice` is a rice
(`brown` is a form word), `rice_vinegar` is a vinegar (different head), and
`peanut_butter` is not a butter (`peanut` is itself an ingredient). All three
cases are asserted in `tests/test_dedup.py`.

## Experiments

Sweeps are files, not shell commands. A sweep typed into a terminal leaves no
record; six weeks later the results directory holds forty runs whose only
provenance is shell history on one laptop.

Every sweep is validated against the leakage rule **before anything runs**.
Discovering that half a grid was invalid after paying for it is the specific
failure this prevents — which matters when compute is a fixed $150 rather than a
tap.

A failing trial does not abort the batch. Aborting discards the trials that
already succeeded and the compute they cost.

## Cloud

`cloud/` submits the *same experiment file* that runs locally. A job that can
only be described in Azure cannot be reproduced on a laptop, and remote
debugging of a distributed configuration is the fastest way to spend a fixed
budget on nothing.

The directory is `cloud/`, not `azure/`, because `azure` is a namespace package
owned by Microsoft's SDK and shadowing it breaks `from azure.ai.ml import ...`.

Hard subscription limits and the reasoning behind each are in
[`cloud/README.md`](cloud/README.md).

## What is deliberately absent

- **No model registry file.** See decision 1.
- **No abstract base class for models.** A function plus a decorator is the
  whole contract; an ABC would add a hierarchy without adding a constraint.
- **No config framework.** `TrainContext.params` is a dict with per-model
  defaults declared next to the model. Hydra-style composition solves a problem
  this workspace does not have.
- **No experiment tracker.** Runs are directories with a manifest and a metrics
  JSON. They diff, they survive, and they need no server.
