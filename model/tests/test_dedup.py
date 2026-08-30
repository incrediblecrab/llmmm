"""Near-duplicate detection is a product rule, so it gets asserted, not eyeballed.

Every case here is one a similarity model gets *right* and the product gets
wrong: `brown_rice` really is the nearest thing to `rice`, and showing it is
still a bad answer. The head-matching rule is what separates that from
`rice_vinegar`, which merely mentions rice.
"""
import pytest

from ingredient_model.reasoning.dedup import canonical, dedup_ranking, is_near_duplicate

DUPLICATES = [
    ("rice", "brown_rice"), ("lamb", "mutton"), ("beef", "ground_beef"),
    ("onion", "green_onion"), ("scallion", "green_onion"),
    ("sugar", "powdered_sugar"), ("milk", "whole_milk"),
    ("cheese", "cheddar_cheese"), ("flour", "all_purpose_flour"),
    ("shrimp", "prawn"), ("cilantro", "coriander"),
]

# The head noun differs, so these are different ingredients even though one name
# contains the other.
DISTINCT = [
    ("rice", "rice_vinegar"), ("beef", "chicken"), ("salt", "pepper"),
    ("olive_oil", "vegetable_oil"), ("tomato", "tomato_paste"),
    ("coconut", "coconut_milk"), ("chicken", "chicken_broth"),
    ("apple", "apple_juice"), ("butter", "peanut_butter"),
]


@pytest.mark.parametrize("a,b", DUPLICATES)
def test_duplicates(a, b):
    assert is_near_duplicate(a, b), f"{a}/{b} should collapse"
    assert is_near_duplicate(b, a), "the relation must be symmetric"


@pytest.mark.parametrize("a,b", DISTINCT)
def test_distinct(a, b):
    assert not is_near_duplicate(a, b), f"{a}/{b} are different ingredients"
    assert not is_near_duplicate(b, a), "the relation must be symmetric"


def test_canonical_never_empties_a_name():
    """A name made entirely of modifiers keeps them — otherwise "extra virgin"
    and "all purpose" both canonicalise to "" and collapse together."""
    for name in ("extra_virgin", "all_purpose", "fresh", "low_fat"):
        assert canonical(name), f"{name} canonicalised to nothing"
    assert not is_near_duplicate("extra_virgin", "all_purpose")


def test_dedup_ranking_keeps_the_best_of_each_group():
    """`chicken_breast` survives beside `chicken`: the head noun differs, and a
    cut of meat is a different purchase from the whole bird. `brown_rice` and
    `white_rice` do not, because they are the same purchase described twice."""
    names = ["rice", "brown_rice", "white_rice", "chicken", "chicken_breast",
             "soy_sauce"]
    got = dedup_ranking(names, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4], k=4)
    assert [n for n, _ in got] == ["rice", "chicken", "chicken_breast",
                                   "soy_sauce"]


def test_dedup_ranking_respects_k():
    names = [f"thing_{i}" for i in range(50)]
    assert len(dedup_ranking(names, list(range(50, 0, -1)), k=7)) == 7
