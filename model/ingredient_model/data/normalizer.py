"""A corrected, extensible normaliser layered over the prior study's.

Why this exists
---------------
`prior-study/tools/normalize.py` built the corpus every model here is
trained on. It is upstream and this package does not own it, so nothing in this
file mutates it. Instead the base normaliser is instantiated and then *patched
in memory*: the Chinese quantity regex is replaced and extra aliases are merged
into the per-language maps. The public API is identical — `normalize(lang,
items) -> set[int]` — so this is a drop-in wherever the base class was used, and
the vocabulary is untouched, which keeps every embedding row-comparable with
runs trained before the fix.

The two defects it corrects
---------------------------
**1. The Chinese quantity stripper ate ingredient names.** The upstream pattern
is

    [0-9]+ | [适少半一二三四五六七八九十两]?[大小]?[勺匙杯个只根片块条把颗粒…]+ | (…)

Both prefix groups are optional, so the measure-word class matches *anywhere*,
including inside a word. Every Chinese ingredient containing one of those
characters lost it before matching was attempted:

    培根 bacon        -> 培      面包 bread     -> 面
    巧克力 chocolate  -> 巧力    面条 noodle    -> 面
    抹茶 matcha       -> 抹      包菜 cabbage   -> 菜

Twelve of the 346 Chinese lexicon entries were *unreachable by construction* —
present in the table, impossible to match. Not obscure ones: chocolate, bread,
noodles, bacon, cabbage, broth, chicken broth. The fix anchors quantity
stripping to the ends of the string and requires a Chinese numeral to be
followed by a measure word, which stops `八角` (star anise), `三文鱼` (salmon)
and `五香粉` (five-spice) from being read as "8 …", "3 …" and "5 …".

**2. The lexicons are thin outside English.** 5,710 English aliases against 346
Chinese aliases, for the same fixed concept vocabulary, while Chinese is 31% of
the corpus. `EXTRA` carries the additions; see `data/aliases/`.

Safety property that matters for Chinese
----------------------------------------
Chinese matching is substring-based (no word boundaries exist to use), with
alternatives sorted longest-first so the longest match at each position wins.
That makes adding a *short* alias dangerous: mapping 鸡 -> chicken is only safe
because 鸡蛋 (egg) is already mapped and, being longer, is tried first. Add a
short term whose compounds are absent and those compounds silently become the
wrong ingredient. `validate()` enforces this: every added term is checked
against the frequent unmapped terms that contain it.
"""
from __future__ import annotations

import functools
import json
import re
import sys
from pathlib import Path

from ..config import PATHS

LLMMM = PATHS.prior_tools

# --------------------------------------------------------------------- Chinese
_NUM = r"[0-9０-９.·/]+"
_Q = r"[适少半一二三四五六七八九十两]"
_M = (r"[勺匙杯个只根片块条把颗粒盒袋包碗斤克千毫升汤茶量许瓶罐支张朵头瓣段节撮"
      r"滴份]")
_UNIT = r"(?:[kK][gG]|[gG]|[mM][lL]|[lL]|[cC][cC]|ｇ)"
_WORD = r"(?:适量|少许|适当|少量|一些|若干|随意)"

#: Leading quantity: a bare numeral may stand alone (``100g低粉``) but a Chinese
#: numeral must be followed by a measure word, or ``八角`` reads as "8 corners".
_LEAD = (rf"^(?:{_WORD}|{_NUM}{_UNIT}?[大小]?{_M}*|{_Q}[大小]?{_M}+"
         rf"|\(.*?\)|（.*?）|[\s、,，:：\-%％])+")
_TAIL = (rf"(?:{_NUM}(?:{_UNIT}|[大小]?{_M}+)?|{_WORD}|\(.*?\)|（.*?）)+$")

ZH_QTY = re.compile(rf"{_LEAD}|{_TAIL}")

ALIAS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "aliases"


def _load_extra() -> dict[str, dict[str, str]]:
    """Per-language ``{surface form: vocabulary concept}`` from data/aliases/."""
    out: dict[str, dict[str, str]] = {}
    if not ALIAS_DIR.exists():
        return out
    for path in sorted(ALIAS_DIR.glob("*.json")):
        lang = path.stem
        data = json.loads(path.read_text(encoding="utf-8"))
        out[lang] = {str(k): str(v) for k, v in data.items() if k and v}
    return out


class ExtendedNormalizer:
    """The base normaliser with the Chinese quantity fix and extra aliases.

    ``strict=False`` reproduces upstream behaviour exactly, which is what the
    A/B measurement in ``scripts/measure_alias_gain.py`` compares against.
    """

    def __init__(self, *, fix_zh_qty: bool = True, extra: bool = True) -> None:
        if str(LLMMM) not in sys.path:
            sys.path.insert(0, str(LLMMM))
        import normalize as nm  # type: ignore

        self._base = nm.Normalizer()
        self.vocab = self._base.vocab
        self.itos = self._base.itos
        self.fix_zh_qty = fix_zh_qty
        self.added: dict[str, int] = {}
        self.rejected: dict[str, str] = {}

        if extra:
            self._merge(_load_extra())

    # ------------------------------------------------------------------ setup
    def _merge(self, extra: dict[str, dict[str, str]]) -> None:
        """Add aliases, then recompile the affected patterns.

        A term whose target is not in the vocabulary is *rejected*, not
        silently dropped: a typo in a concept name would otherwise quietly
        remove an ingredient from the fix and nobody would notice.
        """
        import lexicons  # type: ignore

        for lang, table in extra.items():
            if lang == "en":
                for surface, concept in table.items():
                    if concept not in self.vocab:
                        self.rejected[f"en:{surface}"] = concept
                        continue
                    self._base.alias[surface.lower()] = self.vocab[concept]
                    self.added["en"] = self.added.get("en", 0) + 1
                continue

            target = self._base.maps.setdefault(lang, {})
            for surface, concept in table.items():
                if concept not in self.vocab:
                    self.rejected[f"{lang}:{surface}"] = concept
                    continue
                # Non-CJK matching lowercases the recipe text before looking
                # anything up, so a capitalised key can never be hit. This
                # cost a whole pass of Russian aliases that measured +0.0%.
                if lang not in lexicons.NO_BOUNDARY:
                    surface = surface.lower()
                target[surface] = self.vocab[concept]
                self.added[lang] = self.added.get(lang, 0) + 1

        if self.added.get("en"):
            import build_recipe_cooc as brc  # type: ignore
            self._base.en_pat = brc.build_matcher(self._base.alias)

        for lang in extra:
            if lang == "en" or lang not in self._base.maps:
                continue
            keys = sorted(self._base.maps[lang], key=len, reverse=True)
            body = "|".join(re.escape(k) for k in keys)
            self._base.pats[lang] = re.compile(
                body if lang in lexicons.NO_BOUNDARY else rf"\b(?:{body})\b")

    # -------------------------------------------------------------- normalise
    def normalize(self, lang: str, items) -> set[int]:
        if not items:
            return set()
        if lang == "zh" and self.fix_zh_qty:
            text = " ".join(ZH_QTY.sub("", str(i)) for i in items)
            pat, table = self._base.pats["zh"], self._base.maps["zh"]
            return {table[h] for h in pat.findall(text)}
        return self._base.normalize(lang, items)

    @property
    def maps(self):
        return self._base.maps

    @property
    def alias(self):
        return self._base.alias


@functools.lru_cache(maxsize=4)
def get_normalizer(fix_zh_qty: bool = True, extra: bool = True):
    return ExtendedNormalizer(fix_zh_qty=fix_zh_qty, extra=extra)
