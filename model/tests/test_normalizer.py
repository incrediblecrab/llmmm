"""The corrected normaliser.

The Chinese quantity stripper in llmmm's normalize.py matched measure words
anywhere in a string, including inside an ingredient name, which made twelve
lexicon entries — chocolate, bread, noodle, bacon among them — impossible to
match however many recipes contained them. These tests pin the fix and, just
as importantly, pin the two ways the obvious fix goes wrong:

  * anchoring alone eats 包菜 (cabbage), which *starts* with a measure word;
  * requiring a numeral prefix eats 八角 (star anise) and 三文鱼 (salmon),
    which start with the numerals 8 and 3.

A regression in either direction is silent — it removes ingredients from a
corpus of 4.6M recipes and nothing raises.
"""
from __future__ import annotations

import pytest

from ingredient_model.data.normalizer import ZH_QTY, get_normalizer


@pytest.fixture(scope="module")
def upstream():
    return get_normalizer(fix_zh_qty=False, extra=False)


@pytest.fixture(scope="module")
def fixed():
    return get_normalizer(fix_zh_qty=True, extra=False)


def names(nz, lang, term):
    return sorted(nz.itos[i] for i in nz.normalize(lang, [term]))


@pytest.mark.parametrize("term,concept", [
    ("培根", "bacon"), ("巧克力", "chocolate"), ("面条", "noodle"),
    ("面包", "bread"), ("包菜", "cabbage"), ("抹茶", "matcha_powder"),
    ("高汤", "broth"), ("鸡汤", "chicken_broth"), ("面包糠", "bread_crumbs"),
])
def test_entries_upstream_could_never_match(upstream, fixed, term, concept):
    assert not upstream.normalize("zh", [term]), (
        f"{term} was expected to be broken upstream; if this fails the "
        f"upstream bug was fixed and this module may be redundant")
    assert names(fixed, "zh", term) == [concept]


@pytest.mark.parametrize("term,concept", [
    ("八角", "star_anise"), ("三文鱼", "salmon"),
    ("五香粉", "five_spice_powder"), ("四季豆", "green_bean"),
    ("五花肉", "pork"), ("十三香", "five_spice_powder"),
])
def test_numeral_initial_names_are_not_read_as_quantities(fixed, term, concept):
    """八 is 'eight'. If the quantity rule strips a bare numeral, star anise
    becomes '角' and disappears. These worked upstream and must keep working."""
    assert concept in names(fixed, "zh", term)


@pytest.mark.parametrize("line,concept", [
    ("1片培根", "bacon"), ("适量面包糠", "bread_crumbs"), ("两瓣蒜", "garlic"),
    ("半个洋葱", "onion"), ("猪肉500克", "pork"), ("70%黑巧克力", "chocolate"),
    ("1个鸡蛋", "egg"), ("一根胡萝卜", "carrot"), ("2汤匙酱油", "soy_sauce"),
])
def test_real_quantities_are_still_stripped(fixed, line, concept):
    """Quantity stripping must keep working; the fix narrows it, and a
    narrowing that goes too far leaves the amount glued to the ingredient."""
    assert concept in names(fixed, "zh", line)


def test_no_lexicon_entry_is_damaged(fixed):
    """The strongest form: no Chinese lexicon key may be altered by quantity
    stripping. Upstream this was true of 15 keys and fatal for 12."""
    damaged = [k for k in fixed.maps["zh"] if ZH_QTY.sub("", k) != k]
    assert damaged == []


def test_every_lexicon_entry_matches_itself(fixed):
    dead = [k for k in fixed.maps["zh"] if not fixed.normalize("zh", [k])]
    assert dead == []


def test_alias_targets_all_exist_in_vocab():
    """A concept name that is not in the vocabulary is rejected rather than
    dropped, so a typo shows up here instead of quietly deleting an
    ingredient from the fix."""
    nz = get_normalizer(fix_zh_qty=True, extra=True)
    assert nz.rejected == {}, f"aliases naming unknown concepts: {nz.rejected}"


def test_aliases_do_not_break_existing_matches():
    """Adding aliases must be monotone: anything the base normaliser resolved
    must still resolve. Chinese matching is substring-based, so a careless
    short alias could shadow a longer entry."""
    base = get_normalizer(fix_zh_qty=True, extra=False)
    full = get_normalizer(fix_zh_qty=True, extra=True)
    for lang in ("zh", "ru"):
        for term in list(base.maps.get(lang, {}))[:400]:
            before = base.normalize(lang, [term])
            after = full.normalize(lang, [term])
            assert before <= after, f"{lang}:{term} lost {before - after}"


def test_english_is_untouched_by_the_zh_fix(upstream, fixed):
    for line in ("1 c. firmly packed brown sugar", "2 Tbsp. butter",
                 "1/2 tsp. vanilla"):
        assert (upstream.normalize("en", [line])
                == fixed.normalize("en", [line]))
