"""Dataset access.

Two rules hold everywhere in this package:

1. **Loaders are cached and read-only.** The corpus is 36M tokens; reloading it
   per model in a sweep dominates runtime. Arrays handed out are marked
   non-writeable so one model cannot corrupt the input of the next.

2. **Training data and evaluation labels are separate namespaces.** Anything
   under ``catalog`` is a label and is never reachable from a training path.
"""
from .graphs import (CHEM_GRAPH, GRAPH_FULL, GRAPH_HELDOUT, GRAPH_TRAIN,
                     ChemGraph, IIGraph, load_chem_graph, load_ii_graph,
                     load_heldout, vocab)
from .recipes import RecipeCorpus, load_recipes
from .labels import Substitutions, load_substitutions
from .registry import DATASETS, check_available, describe
from .splits import (DEFAULT_SPLIT, SPLITS, LeakageError, Split, check_leakage,
                     get_split)

__all__ = [
    "CHEM_GRAPH", "GRAPH_FULL", "GRAPH_HELDOUT", "GRAPH_TRAIN",
    "ChemGraph", "IIGraph", "RecipeCorpus", "Substitutions",
    "load_chem_graph", "load_ii_graph", "load_heldout", "load_recipes",
    "load_substitutions", "vocab",
    "DATASETS", "check_available", "describe",
    "DEFAULT_SPLIT", "SPLITS", "Split", "LeakageError", "check_leakage",
    "get_split",
]
