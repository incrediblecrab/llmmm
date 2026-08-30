# Model card — `text_embedding`

## What it is

Ingredient embeddings derived from the ingredient *name* using a hosted text
model, plus a ridge map from that space into a recipe-trained one.

| model | what it does | cost |
|---|---|---|
| `text-embed` | embeds the 1,790 names directly | ~1 API call, cached |
| `text-aligned` | fits `text → learned` and reports the reconstruction | seconds |

## Why this family exists

Every other family here is transductive: it has exactly one row per known
ingredient and nothing whatsoever to say about a word it has not seen. That is
fine for evaluation, where the vocabulary is frozen at 1,790, and fatal in a
product, where someone types "gochujang".

`text-aligned` is the usable artefact: the projection maps any new name into the
recipe-trained geometry, so an unknown ingredient gets a position without
retraining anything.

## What the numbers mean here

`text-embed` is *expected* to score poorly on M2 and M4. It is not competing —
it measures how much of the ingredient space is recoverable from language alone,
and the gap between it and the learned models is the measured value of having a
corpus.

`text-aligned` reports metrics for the **reconstructed** vectors, not the
originals. Read them as "this is how much of the good space survives the trip
from a name", which is precisely the cold-start user's experience.

## Requirements

```bash
pip install 'ingredient-model[foundry]'
export AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
export AZURE_OPENAI_API_KEY=...
```

This is the only family that calls a hosted service. Every other model trains
locally from files.

## Caveats

- **A text model knows words, not cooking.** It knows `lemon` and `lime` are
  similar strings; it does not know they behave differently in a custard. The
  learned table stays authoritative for known ingredients — this family is for
  the ones that are missing.
- Vectors are cached under `data/cache/text_embeddings/`, keyed by model, prompt
  and the exact name list. Re-running a configuration is free; changing any of
  the three correctly misses.
- Foundry embeddings bill against **token quota, not vCPU quota**, so the
  subscription's 6-core / zero-GPU limit does not constrain this family. It is
  the one place a large model is affordable here.
- `text-embed` truncates rather than PCAs when `d_model` is set, because OpenAI
  embedding models are Matryoshka-trained and a prefix is already a valid
  smaller embedding.
