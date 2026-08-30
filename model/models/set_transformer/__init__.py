"""Masked ingredient modelling over recipes.

Every other family produces a static vector per ingredient. This one learns a
*conditional* model: given the ingredients a recipe already has, what belongs
next? That is the product question verbatim, and it is a strictly harder object
than a similarity table — "tomato goes with basil" is a fact about two
ingredients, while "given tomato, mozzarella and olive oil, basil belongs" is a
fact about a context.

The architecture is a set encoder: recipes are unordered, so there is no
positional encoding and attention is fully permutation-equivariant. Adding
positions would let the model learn artefacts of ingredient listing order, which
is a property of the scraper, not of cooking.

The exported embedding is the input token table, so the harness can score this
family on the same terms as every other. The full conditional model is saved
alongside and is what the reasoning layer uses.
"""
from .train import train_masked_set

__all__ = ["train_masked_set"]
