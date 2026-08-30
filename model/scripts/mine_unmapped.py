#!/usr/bin/env python
"""Mine the ingredient lines that normalisation throws away.

`analyse_coverage.py` established *that* the non-English lexicons are thin.
This finds *which* terms to add, ranked by how many recipes they would recover,
so the work goes where the mass is rather than where intuition points.

Run: ./.venv/bin/python scripts/mine_unmapped.py [--per-lang N]

Output is `data/recipes/unmapped_terms.json`: per language, the dropped surface
forms with their document frequencies. A term's value is the number of recipes
it appears in, not the number of occurrences — recipes are ingredient *sets*,
so a term mentioned twice in one recipe still recovers only one edge.

Quantity prefixes are stripped before counting, because "1个紫薯" and "适量紫薯"
are the same missing word and splitting them across two rows would bury it.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingredient_model.config import PATHS  # noqa: E402
from ingredient_model.data.recipes import load_recipes  # noqa: E402
from ingredient_model.data.text import TEXT_FILE  # noqa: E402

LLMMM = PATHS.prior_tools
OUT = "unmapped_terms.json"

# Lines that are structure, not food. Recovering these would be a regression:
# they are correctly discarded and only clutter the ranking.
NOISE = re.compile(
    r"^(?:for\s+the\s+)?(?:filling|topping|garnish|sauce|cake|crust|glaze|"
    r"salad|dough|batter|marinade|dressing|frosting|icing|base|assembly|"
    r"optional|other|extras?|ingredients?|method|note|serve|serving|"
    r"主料|辅料|馅料|主面团|配料|调料|腌料|做法|材料)\s*[:：]?\s*$",
    re.I)


_RU_NAME = re.compile(r"['\"]name['\"]\s*:\s*['\"](.+?)['\"]")
#: Povarenok also emits a flat "Ингредиент: 1 шт" form. Split only when what
#: follows the colon really is an amount, so a name containing a colon is safe.
_RU_QTY = re.compile(r"^(.{2,}?)\s*:\s*(?:None|\d.{0,18}|"
                     r"по вкусу|щепотка)\s*$", re.I)
_PCT = re.compile(r"^\s*\d+\s*%\s*|\s*\d+\s*%\s*$")
_UNIT_HEAD = re.compile(r"^(?:[0-9０-９.·/]*\s*(?:kg|g|ml|mL|L|cc|oz|lb)\s*)+",
                        re.I)


def clean(term: str, lang: str, zh_qty) -> str:
    t = term.strip()
    # Povarenok rows are dicts; text.py preserves them as their repr, so the
    # ingredient is buried in a 'name' field alongside its quantity. Without
    # this every alias key would carry its own amount and match nothing else.
    m = _RU_NAME.search(t)
    if m:
        t = m.group(1)
    m = _RU_QTY.match(t)
    if m:
        t = m.group(1)
    if lang in ("zh", "ja", "th"):
        t = zh_qty.sub("", t)
        t = _PCT.sub("", t)
        t = _UNIT_HEAD.sub("", t)
    t = t.strip(" \t:：.,-–—*#•·_()[]")
    return t.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lang", type=int, default=60000,
                    help="recipes sampled per language (0 = all)")
    ap.add_argument("--min-df", type=int, default=3)
    ap.add_argument("--raw", action="store_true",
                    help="mine against upstream behaviour, for A/B comparison")
    a = ap.parse_args()

    path = PATHS.recipes / TEXT_FILE
    if not path.exists():
        print("no text index — run `make text` first", file=sys.stderr)
        return 1
    from ingredient_model.data.normalizer import ZH_QTY, get_normalizer

    # Mine against the *corrected* normaliser, so the ranking reflects what is
    # still missing after the quantity fix. Mining with the buggy stripper
    # invents phantom terms like "g低粉" that then block the real one.
    nz = get_normalizer(fix_zh_qty=not a.raw, extra=not a.raw)
    zh_qty = ZH_QTY if not a.raw else __import__("build_recipe_cooc")._ZH_QTY
    corpus = load_recipes()
    lang = np.asarray(corpus.lang)
    text = pd.read_parquet(path, columns=["raw_ingredients"])
    raw = text["raw_ingredients"].to_numpy()
    delimited = np.array([x.count("\x1f") > 0 for x in raw])
    rng = np.random.default_rng(0)

    out: dict[str, dict] = {}
    print(f"{'lang':>5}{'recipes':>10}{'lines':>10}{'dropped':>9}"
          f"{'terms':>8}{'top-500 df':>12}")
    for lg in sorted(set(lang.tolist())):
        cand = np.flatnonzero((lang == lg) & delimited)
        if len(cand) < 300:
            continue
        if a.per_lang and len(cand) > a.per_lang:
            cand = rng.choice(cand, size=a.per_lang, replace=False)
        miss: Counter = Counter()
        lines = dropped = 0
        for i in cand:
            seen = set()
            for part in raw[i].split("\x1f"):
                part = part.strip()
                if not part:
                    continue
                lines += 1
                if nz.normalize(lg, [part]):
                    continue
                dropped += 1
                t = clean(part, lg, zh_qty)
                # document frequency: once per recipe, however often it recurs
                if t and len(t) < 40 and not NOISE.match(t) and t not in seen:
                    seen.add(t)
                    miss[t] += 1
        keep = {t: c for t, c in miss.most_common() if c >= a.min_df}
        head = sum(sorted(keep.values(), reverse=True)[:500])
        out[lg] = {"recipes": len(cand), "lines": lines, "dropped": dropped,
                   "terms": keep}
        print(f"{lg:>5}{len(cand):>10,}{lines:>10,}{dropped / max(lines,1):>8.1%}"
              f"{len(keep):>8,}{head:>12,}")

    dest = PATHS.recipes / OUT
    dest.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    tot = sum(len(v["terms"]) for v in out.values())
    print(f"\n{tot:,} candidate terms across {len(out)} languages -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
