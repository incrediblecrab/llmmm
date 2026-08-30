# Model card — `set_transformer`

## What it is

`masked-set` — a permutation-invariant transformer that predicts a hidden
ingredient from the rest of a recipe.

## Why this family exists

Every other family produces a static vector per ingredient and answers "how
similar are these two". This one learns a *conditional* model and answers "given
what is already in the bowl, what belongs next" — which is the product question
verbatim, and a strictly harder object. "Tomato goes with basil" is a fact about
two ingredients; "given tomato, mozzarella and olive oil, basil belongs" is a
fact about a context.

## Architecture notes

- **No positional encoding.** A recipe is a set. The order it was scraped in is
  a property of the scraper, and giving the model positions lets it learn that
  artefact. Omitting them makes the encoder permutation-equivariant by
  construction rather than by hoping.
- **`[MASK]` lives in the ingredient table** as one extra row, so its learned
  position is interpretable as "the average missing ingredient".
- **Weight tying** is done by multiplying against the live token table, not by
  assigning a slice to an `nn.Linear` — the latter copies, which silently unties
  the two and lets the exported embedding drift from the one used for scoring.
- **Linear warmup is not optional.** Without it the first few hundred steps of
  an untrained attention stack produce gradients large enough to push the token
  table into a degenerate configuration it does not recover from.

## Key parameters

| name | default | note |
|---|---|---|
| `d_model` | 256 | |
| `n_layers` | 2 | |
| `max_recipes` | 600,000 | subsample; the full 3.25M is CPU-hours |
| `max_len` | 32 | recipes above this are dropped, not truncated |
| `epochs` | 3 | |

## Caveats

- **The embedding is not the model.** M6 is reported twice: through the exported
  token table, which keeps this family comparable with the others, and through
  the native scorer, which is what would actually be served. Expect a large gap
  — for EASE the same gap is 0.17 vs 0.59.
- Heavy by this workspace's standards. Everything else here finishes in seconds
  to minutes on CPU; this is the one family where the Azure cluster earns its
  cost.
- `max_len=32` **drops** longer recipes rather than truncating them, because a
  truncated recipe is a false negative: the model is asked to predict from an
  incomplete set and penalised for missing what was cut off.
