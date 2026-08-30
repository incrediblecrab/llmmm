# Adding a model

A model type is a folder. Drop it in `models/`, and `im list` finds it — there
is no central registry to edit and therefore no central registry to forget to
edit.

```
models/your_family/
    __init__.py     imports your train functions so discovery can see them
    train.py        one @register-decorated function per model
    card.md         what it is, what it needs, what it cannot do
```

## The contract

```python
from ingredient_model.registry import register
from ingredient_model.spec import TrainContext, TrainResult

DEFAULTS = dict(d_model=300, epochs=10)

@register(name="your-model", family="your_family", cost_hint="moderate",
          defaults=DEFAULTS, requires=("ii_graph_train",),
          description="one line, shown by `im list`")
def train_your_model(ctx: TrainContext) -> TrainResult:
    p = {**DEFAULTS, **dict(ctx.params)}
    ...
    return TrainResult(embedding=W, metadata=dict(p))
```

`requires` is not documentation. It is what the leakage guard reads to decide
which splits your model may be scored on: declare `"recipes"` and the guard will
refuse to run you against edge-level labels. Getting this wrong is how a model
ends up with an inflated number on the leaderboard, so declare what you actually
read.

## Returning more than an embedding

The embedding is the common currency — it is what makes a random-walk model and
a transformer comparable. But for some families it is a lossy export rather than
the model itself. If yours scores candidates conditionally, return a `scorer`:

```python
return TrainResult(embedding=W, scorer=lambda ctx_ids: ...)   # (m,k) -> (m,n_vocab)
```

M6 is then reported twice, through the embedding and natively. This is not a
formality. EASE scores **0.185** through its embedding and **0.625** natively —
the embedding export loses more than half the model, and without the native path
EASE looks worse than recommending onion and salt to everyone.

Anything else — item biases, context tables, attention weights — goes in
`extra_arrays` and is saved beside the embedding without the harness needing to
know about it.

## Before you commit

1. `im train your-model --split recipe-holdout` — it must beat the control gate.
2. Check M6 against the popularity baseline. A model that loses to popularity
   has not learned to cook, it has learned what is common.
3. Write `card.md`, including what the model **cannot** do. The caveats are the
   part that saves someone a week.
