#!/usr/bin/env python3
"""Per-source yield audit: does every corpus actually hand us ingredients?

A source that parses without error but returns empty lists is invisible to the
smoke test and silently drops its recipes from training. This reports, per
source, the share of records that yield at least one ingredient string and the
mean ingredients per record. Anything with low yield or a mean far from the
5-12 range typical of real recipes is a parser bug, not a data property.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus

ROOT = Path(__file__).resolve().parents[1]


def main(limit: int | None) -> None:
    rec = defaultdict(lambda: [0, 0, 0])  # recipes, with_ing, total_ing
    lang_of = {}
    for key, lang, items in corpus.iter_all(limit):
        r = rec[key]
        lang_of[key] = lang
        r[0] += 1
        if items:
            r[1] += 1
            r[2] += len(items)

    print(f"{'source':26}{'lang':5}{'recipes':>12}{'yield':>9}{'ing/recipe':>12}  flag")
    bad = []
    for key, (n, w, tot) in sorted(rec.items(), key=lambda x: -x[1][0]):
        y = w / max(n, 1)
        per = tot / max(w, 1)
        flag = ""
        if y < 0.90:
            flag = "LOW YIELD"
        elif per < 2.5:
            flag = "TOO FEW ING"
        elif per > 40:
            flag = "TOO MANY ING"
        if flag:
            bad.append(key)
        print(f"{key:26}{lang_of[key]:5}{n:>12,}{y:>8.1%}{per:>12.1f}  {flag}")
    print(f"\n{len(rec)} sources, {sum(v[0] for v in rec.values()):,} records")
    if bad:
        print("needs attention:", ", ".join(bad))
    else:
        print("all sources healthy")


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "-"
    main(None if a == "-" else int(a))
