"""Graph neural propagation over the ingredient graph.

Every other family learns an embedding table. These learn a *propagation rule*:
an ingredient's representation is built from its neighbours', so the graph
structure enters at inference as well as at training.

``lightgcn``  linear neighbourhood propagation with layer averaging (He et al.,
              SIGIR 2020). No feature transform and no non-linearity — both were
              shown to hurt on collaborative-filtering graphs, and their absence
              is what makes it closed-form-adjacent and CPU-viable here.
``sgc``       simple graph convolution (Wu et al., ICML 2019): the same
              propagation applied to a spectral base, with the classifier
              dropped.

Both are cheap at n = 1,790 — a dense propagation matrix is 25 MB and each layer
is one matrix multiply.
"""
from .train import train_lightgcn, train_sgc

__all__ = ["train_lightgcn", "train_sgc"]
