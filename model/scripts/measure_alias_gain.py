#!/usr/bin/env python
"""A/B the normaliser fixes against upstream, on identical recipes.

    ./.venv/bin/python scripts/measure_alias_gain.py [--per-lang 40000]

Reports, per language, what each layer recovers:

    upstream     llmmm's normalize.py exactly as the corpus was built
    +regex       the Chinese quantity fix alone
    +aliases     the fix plus data/aliases/*.json

The same recipes and the same lines go through every variant, so a difference
is attributable and not a sampling artefact. Two numbers are reported because
they answer different questions:

    line recall     share of ingredient lines that resolve to something.
                    Measures the lexicon.
    tokens/recipe   ingredients recovered per recipe. Measures what the models
                    actually get to learn from, which is the point.

A recovered line is only worth something if it adds an ingredient the recipe
did not already have, so tokens/recipe counts the *set*, not the lines.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingredient_model.config import PATHS  # noqa: E402
from ingredient_model.data.normalizer import get_normalizer  # noqa: E402
from ingredient_model.data.recipes import load_recipes  # noqa: E402
from ingredient_model.data.text import TEXT_FILE  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lang", type=int, default=40000)
    a = ap.parse_args()

    path = PATHS.recipes / TEXT_FILE
    if not path.exists():
        print("no text index — run `make text` first", file=sys.stderr)
        return 1

    variants = {
        "upstream": get_normalizer(fix_zh_qty=False, extra=False),
        "+regex": get_normalizer(fix_zh_qty=True, extra=False),
        "+aliases": get_normalizer(fix_zh_qty=True, extra=True),
    }
    added = variants["+aliases"].added
    rejected = variants["+aliases"].rejected
    print(f"aliases loaded: {added}"
          + (f"   REJECTED {len(rejected)} (target not in vocab)"
             if rejected else ""))
    if rejected:
        for k, v in list(rejected.items())[:5]:
            print(f"    {k} -> {v}")

    corpus = load_recipes()
    lang = np.asarray(corpus.lang)
    raw = pd.read_parquet(path, columns=["raw_ingredients"])[
        "raw_ingredients"].to_numpy()
    delimited = np.array([x.count("\x1f") > 0 for x in raw])
    rng = np.random.default_rng(0)

    print(f"\n{'lang':>5}{'recipes':>9}  {'variant':<10}"
          f"{'line recall':>12}{'tokens/recipe':>15}{'gain':>8}")
    grand = {}
    for lg in sorted(set(lang.tolist())):
        cand = np.flatnonzero((lang == lg) & delimited)
        if len(cand) < 300:
            continue
        if len(cand) > a.per_lang:
            cand = rng.choice(cand, size=a.per_lang, replace=False)
        parts = [[p.strip() for p in raw[i].split("\x1f") if p.strip()]
                 for i in cand]
        lines = sum(len(p) for p in parts)
        base_tok = None
        for name, nz in variants.items():
            hit = 0
            toks = 0
            for lst in parts:
                got: set[int] = set()
                for line in lst:
                    ids = nz.normalize(lg, [line])
                    if ids:
                        hit += 1
                        got |= ids
                toks += len(got)
            per = toks / len(cand)
            if base_tok is None:
                base_tok = per
            gain = (per / base_tok - 1) if base_tok else 0.0
            grand.setdefault(name, []).append((len(cand), toks, hit, lines))
            print(f"{lg:>5}{len(cand):>9,}  {name:<10}{hit / lines:>11.1%}"
                  f"{per:>15.2f}{gain:>+8.1%}")
        print()

    print(f"{'':>5}{'':>9}  {'variant':<10}{'line recall':>12}"
          f"{'tokens/recipe':>15}{'gain':>8}")
    base = None
    for name, rows in grand.items():
        n = sum(r[0] for r in rows)
        toks = sum(r[1] for r in rows)
        hit = sum(r[2] for r in rows)
        lines = sum(r[3] for r in rows)
        per = toks / n
        base = per if base is None else base
        print(f"{'ALL':>5}{n:>9,}  {name:<10}{hit / lines:>11.1%}"
              f"{per:>15.2f}{per / base - 1:>+8.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
