"""Text embeddings for ingredients that are not in the vocabulary.

Every other family in this workspace is *transductive*: it learns one row per
known ingredient and has nothing at all to say about a word it has never seen.
That is fine for evaluation, where the vocabulary is fixed at 1,790, and fatal
in a product, where someone types "gochujang" or "nduja" and expects an answer.

This family closes that gap by embedding the ingredient *name* with a text
model, so a new ingredient gets a position from its name alone. The learned
table remains authoritative for the 1,790 known ingredients — a text model knows
that "lemon" and "lime" are similar words, not that they behave alike in a
recipe — and the text space is used to *place* unknown ingredients relative to
known ones.

Two practical notes:

* Azure Foundry embeddings bill against token quota, not vCPU quota. The
  subscription's hard limit of 6 dedicated cores and zero GPU does not apply
  here, which makes this the one family that can use a large model.
* 1,790 short strings is a single batched request costing a fraction of a cent.
  Results are cached to disk because re-embedding a fixed vocabulary on every
  run is spending money to get the same answer.
"""
from .train import train_text_embed, train_text_aligned

__all__ = ["train_text_embed", "train_text_aligned"]
