"""Combining spaces that disagree.

The prior study established two facts that sit awkwardly together: raw
co-occurrence statistics rank held-out pairs better than any learned embedding
(AUC 0.984 vs 0.873), yet embeddings generalise to pairs that have never
co-occurred, where counting has nothing to say at all. Neither is dominant, and
picking one throws away what the other knows.

``concat``    length-normalise each space and concatenate, with a weight per
              source. The honest baseline for any blend: no parameters, nothing
              fitted, and it captures most of what naive combination can offer.
``residual``  fit the second space to explain what the first gets *wrong*, then
              add only the part that is orthogonal to it. Combination stops
              being a mixture and becomes a correction.

Both take their inputs as run IDs, so any two runs from any two families can be
combined without either knowing the other exists.
"""
from .train import train_concat, train_residual

__all__ = ["train_concat", "train_residual"]
