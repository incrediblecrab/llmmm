#!/usr/bin/env python3
"""Full accounting of the recipe corpus. No sampling, no estimates.

Answers three questions that were previously asserted rather than proven:

  1. Did we read every recipe? Every one of the 2,147,248 parquet rows is
     placed in exactly one bucket, and the buckets must sum to the total.
  2. Can we match all the ingredients? Reports line-level match rate and dumps
     the most common UNMATCHED lines, which is the only honest way to see what
     the matcher is blind to.
  3. Which vocabulary terms never appear, and is that our failure or the
     corpus's?

Run: ./.venv/bin/python tools/audit_corpus_coverage.py   (~12 min)
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from build_recipe_cooc import (  # noqa: E402
    RAW, _QTY, build_alias, build_matcher, load_vocab,
)


def main() -> None:
    vocab, itos = load_vocab()
    alias = build_alias(vocab)
    pat = build_matcher(alias)

    import pyarrow.parquet as pq

    rows = empty = no_marker = no_lines = parsed = 0
    used = too_few = 0
    lines_total = lines_hit = 0
    unmatched: Counter = Counter()
    seen = np.zeros(len(vocab), np.int64)
    per_recipe: Counter = Counter()

    src = ROOT / "recipe" / "_superseded" / "corbt-all-recipes-NOT-table-a1"
    for f in sorted(src.glob("ar_*.parquet")):
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            col = pf.read_row_group(g, columns=["input"]).column(0).to_pylist()
            for txt in col:
                rows += 1
                if not txt:
                    empty += 1
                    continue
                body = txt.split("Ingredients:", 1)
                if len(body) < 2:
                    no_marker += 1
                    continue
                body = body[1].split("Directions:", 1)[0]
                lines = [_QTY.sub("", ln).strip().lower()
                         for ln in body.splitlines() if ln.strip()]
                if not lines:
                    no_lines += 1
                    continue
                parsed += 1

                found = set()
                for ln in lines:
                    lines_total += 1
                    # identical to the extractor's inner loop
                    hits = {alias[m.group(1)] for m in pat.finditer(ln)}
                    if hits:
                        lines_hit += 1
                        found |= hits
                    elif len(unmatched) < 400_000:
                        unmatched[ln[:60]] += 1
                per_recipe[len(found)] += 1
                if len(found) >= 2:
                    used += 1
                    for idx in found:
                        seen[idx] += 1
                else:
                    too_few += 1

    print("=" * 68)
    print("  ROW ACCOUNTING  (every parquet row lands in exactly one bucket)")
    print("=" * 68)
    buckets = [("empty / null text", empty),
               ("no 'Ingredients:' marker", no_marker),
               ("marker but zero lines", no_lines),
               ("parsed -> <2 vocab ingredients", too_few),
               ("parsed -> USED for co-occurrence", used)]
    for label, v in buckets:
        print(f"    {label:38s} {v:>10,}  {100*v/rows:6.2f}%")
    tot = sum(v for _, v in buckets)
    print(f"    {'-'*38} {'-'*10}")
    print(f"    {'SUM':38s} {tot:>10,}")
    print(f"    {'parquet rows':38s} {rows:>10,}")
    print(f"    reconciles: {tot == rows}")

    print()
    print("=" * 68)
    print("  INGREDIENT-LINE MATCH RATE")
    print("=" * 68)
    print(f"    ingredient lines seen              {lines_total:>10,}")
    print(f"    lines with >=1 vocab match         {lines_hit:>10,}  "
          f"{100*lines_hit/lines_total:.2f}%")
    print(f"    lines with no match                "
          f"{lines_total-lines_hit:>10,}  "
          f"{100*(lines_total-lines_hit)/lines_total:.2f}%")

    print()
    print("    top 25 UNMATCHED lines (what the matcher is blind to):")
    for ln, c in unmatched.most_common(25):
        print(f"      {c:>7,}  {ln}")

    print()
    print("=" * 68)
    print("  VOCABULARY COVERAGE")
    print("=" * 68)
    miss = [itos[i] for i in range(len(itos)) if seen[i] == 0]
    print(f"    vocab terms                        {len(itos):>10,}")
    print(f"    appear in >=1 used recipe          "
          f"{len(itos)-len(miss):>10,}  "
          f"{100*(len(itos)-len(miss))/len(itos):.1f}%")
    print(f"    never appear                       {len(miss):>10,}")
    print(f"    appear in >=100 recipes            "
          f"{int((seen >= 100).sum()):>10,}")
    print()
    print("    a sample of the never-seen terms:")
    for i in range(0, min(len(miss), 60), 6):
        print("      " + "  ".join(f"{m:<22s}" for m in miss[i:i + 6]))

    print()
    print("    ingredients matched per recipe:")
    for k in sorted(per_recipe):
        if k <= 12:
            print(f"      {k:>3d} ingredients  {per_recipe[k]:>9,}  "
                  f"{100*per_recipe[k]/parsed:5.2f}%")
    big = sum(v for k, v in per_recipe.items() if k > 12)
    print(f"      >12 ingredients  {big:>9,}  {100*big/parsed:5.2f}%")
    tot_ing = sum(k * v for k, v in per_recipe.items())
    print(f"      mean {tot_ing/parsed:.2f} matched ingredients per parsed recipe")

    out = ROOT / "data" / "derived" / "corpus_coverage.json"
    out.write_text(json.dumps({
        "parquet_rows": rows, "empty": empty, "no_marker": no_marker,
        "no_lines": no_lines, "too_few_ingredients": too_few, "used": used,
        "reconciles": tot == rows,
        "ingredient_lines": lines_total, "lines_matched": lines_hit,
        "line_match_rate": lines_hit / lines_total,
        "vocab_total": len(itos), "vocab_seen": len(itos) - len(miss),
        "vocab_never_seen": miss,
        "mean_ingredients_per_recipe": tot_ing / parsed,
        "top_unmatched": unmatched.most_common(200),
    }, indent=2))
    print(f"\n  -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
