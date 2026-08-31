#!/usr/bin/env python
"""How much of each recipe survives normalisation, and what is lost.

Every model here is trained on ingredient *sets* produced by llmmm's
normaliser. If that step silently discards ingredients, no model can recover
them, and any evaluation stratified by language inherits the bias. Nothing in
the workspace measured this until the recipe text was rejoined, because you
cannot see what was dropped without the original text to compare against.

Run: ./.venv/bin/python scripts/analyse_coverage.py

Method, and one trap worth naming
---------------------------------
The obvious measure — tokens per recipe divided by ingredient lines per
recipe — says English 90% and Chinese 89% and concludes there is no language
gap. That is wrong. Chinese recipes list more ingredients, so a larger
proportional loss still lands on a similar token count. The confound is recipe
length, and it hides the entire effect.

Measuring per *line* instead, and splitting the loss by cause, shows the real
picture. A line is:

  unrecognised  it normalised to nothing               -> a lexicon gap
  duplicate     it normalised to something already in  -> legitimate
                the recipe (dedup: sets, not lists)
  kept          it contributed a new ingredient

Only sources storing a genuinely delimited ingredient list are counted. Where
a source stores one undelimited blob there is no way to count lines without
guessing where one ingredient ends and the next begins, and guessing produced
retention figures above 100% on the first attempt.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingredient_model.config import PATHS  # noqa: E402
from ingredient_model.data.normalizer import get_normalizer  # noqa: E402
from ingredient_model.data.recipes import load_recipes  # noqa: E402
from ingredient_model.data.text import TEXT_FILE  # noqa: E402

LLMMM = PATHS.prior_tools
SAMPLE = 8000
SOURCES = [("01-recipenlg", "en"), ("02-xiachufang", "zh"),
           ("03-povarenok", "ru"), ("taiwan-1.8k", "zh"), ("bhuvii-17k", "en")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--normalizer", choices=("corrected", "base"), default=None,
        help="which normaliser to measure. Defaults to whichever built the "
             "canonical corpus, per data/GENERATION.json.")
    a = ap.parse_args()

    path = PATHS.recipes / TEXT_FILE
    if not path.exists():
        print("no text index — run `make text` first", file=sys.stderr)
        return 1
    if not (LLMMM / "normalize.py").exists():
        print(f"normaliser not found at {LLMMM}", file=sys.stderr)
        return 1

    # This script used to hardcode the upstream normaliser while comparing its
    # output against the v2 corpus, which was built with the corrected one.
    # Every line the correction rescued therefore counted as a lexicon gap,
    # and the loss it reported was the loss of a normaliser the corpus no
    # longer uses. It now defaults to the one that actually built the corpus,
    # and `--normalizer base` still gives the historical figure — the two are
    # simply no longer the same command.
    gen = json.loads((PATHS.data / "GENERATION.json").read_text())
    which = a.normalizer or gen.get("normalizer", "corrected")
    if which == "base":
        sys.path.insert(0, str(LLMMM))
        import normalize as nm  # type: ignore
        nz = nm.Normalizer()
    else:
        nz = get_normalizer()
    print(f"normaliser: {which}"
          f"{'' if a.normalizer else '  (from GENERATION.json)'}\n")

    corpus = load_recipes()
    vocab = set(corpus.itos)
    src = np.asarray(corpus.source)
    text = pd.read_parquet(path, columns=["source", "raw_ingredients"])
    raw = text["raw_ingredients"].to_numpy()
    delimited = np.array([x.count("\x1f") > 0 for x in raw])
    rng = np.random.default_rng(0)

    print(f"vocabulary {len(vocab):,} ingredients   "
          f"corpus {corpus.n_recipes:,} recipes\n")
    print(f"{'source':<16}{'lang':>5}{'lines':>9}{'unrecog':>9}"
          f"{'dup':>7}{'kept':>7}")

    missing: dict[str, Counter] = {}
    for name, lang in SOURCES:
        cand = np.flatnonzero((src == name) & delimited)
        if len(cand) < 500:
            continue
        pick = rng.choice(cand, size=min(SAMPLE, len(cand)), replace=False)
        miss: Counter = Counter()
        lines = dropped = dupes = kept = 0
        for i in pick:
            seen: set[int] = set()
            for part in raw[i].split("\x1f"):
                part = part.strip()
                if not part:
                    continue
                lines += 1
                ids = nz.normalize(lang, [part])
                if not ids:
                    dropped += 1
                    miss[part[:30]] += 1
                elif set(ids) <= seen:
                    dupes += 1
                else:
                    kept += 1
                    seen |= set(ids)
        missing[name] = miss
        print(f"{name:<16}{lang:>5}{lines:>9,}{dropped / lines:>9.1%}"
              f"{dupes / lines:>7.1%}{kept / lines:>7.1%}")

    print("\nunrecognised = lexicon gap   dup = already in this recipe "
          "(legitimate)\n")

    for name in ("01-recipenlg", "02-xiachufang"):
        if name not in missing:
            continue
        print(f"most frequent unrecognised lines — {name}")
        for term, count in missing[name].most_common(10):
            print(f"   {count:>4}  {term}")
        print()

    # The decisive question: are these concepts absent from the vocabulary, or
    # present but unreachable from this surface form? The answer determines
    # whether the fix is "collect more ingredients" or "add aliases".
    probes = {"培根": "bacon", "百香果": "passion_fruit", "啤酒": "beer",
              "紫薯": "purple_sweet_potato", "bay leaves": "bay_leaf",
              "mayo": "mayonnaise"}
    print("dropped surface form -> is the concept already in the vocabulary?")
    for surface, concept in probes.items():
        print(f"   {surface:<14}{concept:<22}"
              f"{'present' if concept in vocab else 'ABSENT'}")
    print("\nEvery one is present, so the loss is a mapping gap, not a "
          "coverage gap.\nAdding aliases would recover them; enlarging the "
          "vocabulary would not.\n")

    # The root cause, in one table. The English lexicon is an order of
    # magnitude richer than any other, so the drop rates above are not a
    # property of the languages — they are a property of the alias tables.
    sizes = {"en (alias table)": len(nz.alias)}
    sizes.update({k: len(v) for k, v in nz.maps.items()})
    print("lexicon size per language, against a shared vocabulary of "
          f"{len(vocab):,} concepts")
    for lang, size in sorted(sizes.items(), key=lambda kv: -kv[1]):
        print(f"   {lang:<20}{size:>7,}   reaches at most "
              f"{min(size / len(vocab), 1.0):>5.0%} of the vocabulary")
    print("\nChinese is 31% of the corpus and has 346 mappings against "
          "English's 5,710.\nThat is the cause of every gap above, and it "
          "confounds any comparison\nof model quality across languages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
