"""Models trained on recipe baskets directly, without a graph intermediary.

Every other family here consumes the ingredient-ingredient graph, which is a
*pairwise summary* of the corpus. A recipe is not a bag of pairs — it is a set
that someone chose as a whole, and reducing it to pairs discards the
higher-order structure that made it a set.

These models read the 4.6M recipes as implicit feedback: recipes are "users",
ingredients are "items", and an appearance is an unrated positive.

``ease``      closed-form item-item ridge autoencoder (Steck 2019)
``ials``      implicit alternating least squares (Hu, Koren & Volinsky 2008)
``item2vec``  skip-gram over baskets, with no graph in between
"""
from .ease import train_ease
from .ials import train_ials
from .item2vec import train_item2vec

__all__ = ["train_ease", "train_ials", "train_item2vec"]
