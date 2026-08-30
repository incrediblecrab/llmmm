"""Closed-form factorisation of the co-occurrence matrix.

At 1,790 ingredients the co-occurrence matrix is 1790x1790 — small enough to
decompose exactly in seconds. SGNS is an implicit factorisation of shifted PMI
(Levy & Goldberg, 2014), so if what matters is the corpus signal rather than the
optimiser, these should land in the same place.

They also **cannot collapse** by construction, which makes them the control that
separates "this schema carries no information" from "the optimiser wasted it".
"""
from .train import train_chem_svd, train_glove, train_svd_ppmi

__all__ = ["train_svd_ppmi", "train_glove", "train_chem_svd"]
