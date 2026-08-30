"""Random-walk SGNS over the ingredient graphs.

Three walk schemas share one architecture and one set of hyperparameters and
differ *only* in which edges the walk may traverse, so any difference in the
resulting geometry is attributable to the schema and nothing else:

``cooc``  the ingredient-ingredient NPMI graph only — what people cook together.
``chem``  typed ingredient -> compound -> ingredient metapaths only — shared
          flavour compounds.
``core``  the union, with ingredient-ingredient edges up-weighted by
          ``ii_repeat``, which interpolates continuously between the two.

``ii_repeat`` is the useful knob: at 0 ``core`` degenerates to ``chem``'s pure
bipartite schema and at large values it approaches ``cooc``, so one parameter
traces the whole curve between the two extremes rather than sampling it at a
single arbitrary point.
"""
from .train import train_chem, train_cooc, train_core

__all__ = ["train_cooc", "train_chem", "train_core"]
