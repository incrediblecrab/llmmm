"""Turning a trained model into an answer a person can check.

An embedding answers "how similar" with a number between -1 and 1. That is not
an explanation, and on its own it is not even the best available answer: the
prior study found that raw recipe co-occurrence statistics score AUC 0.984 on
held-out pairs while the best embedding scores 0.873. A reasoning layer that
quietly replaced counting with a learned vector would be a downgrade.

So this layer treats the model as *one witness among several* and weights it
accordingly:

* **Co-occurrence** — how often the pair actually appears together, as NPMI.
  Direct evidence, and the strongest single signal.
* **Embedding cosine** — the model's learned similarity. Generalises to pairs
  that have never co-occurred, which is exactly where counting has nothing to
  say.
* **Chemistry** — shared flavour compounds. Included because it is *falsified*
  evidence: the study found chemistry adds nothing (H5), so it is reported for
  transparency and given zero weight by default rather than silently dropped.

Where the witnesses disagree is the interesting part, and disagreement is
surfaced rather than averaged into a single score that hides it. A pair the
model loves but that never co-occurs is either a genuine discovery or an
artefact of the geometry, and the user is told which question they are looking
at.
"""
from .reasoner import Evidence, Reasoner

__all__ = ["Reasoner", "Evidence"]
