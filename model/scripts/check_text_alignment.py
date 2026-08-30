"""Pre-flight the text index: does every metadata stream track its reader?

`build_index` verifies alignment for real, but only by replaying the whole
corpus, which takes tens of minutes. A metadata stream that desynchronises does
not corrupt one row — it shifts every later row in that source, so the failure is
worth catching in seconds rather than at minute twenty.

Run: `python scripts/check_text_alignment.py`  (exit 1 if any source drifts)
"""
from __future__ import annotations

import itertools
import sys

from ingredient_model.data.text import _load_llmmm, _meta_stream

CAP = 12_000


def main() -> int:
    raw, _nm = _load_llmmm()
    print(f"{'source':<24}{'records':>10}{'titled':>9}{'urls':>8}{'steps':>8}  status")
    bad = []
    covered = blank = 0
    for key in raw.ALL_KEYS:
        stream = _meta_stream(raw, key)
        if stream is None:
            print(f"{key:<24}{'-':>10}{'-':>9}{'-':>8}{'-':>8}  no text reader")
            blank += 1
            continue
        n = t = u = s = 0
        status = "ok"
        try:
            for _k, _l, _items in itertools.islice(
                    raw.iter_all(None, only={key}), CAP):
                m = next(stream, None)
                if m is None:
                    status = "DESYNC — metadata ran out before the reader"
                    break
                n += 1
                t += bool(m["title"])
                u += bool(m["url"])
                s += bool(m["steps"])
            else:
                if n < CAP and next(stream, None) is not None:
                    status = "DESYNC — metadata outlasted the reader"
        except Exception as e:
            status = f"ERROR {type(e).__name__}: {str(e)[:40]}"
        if status != "ok":
            bad.append(key)
        else:
            covered += 1
        print(f"{key:<24}{n:>10,}{t / max(n, 1):>8.0%}{u / max(n, 1):>8.0%}"
              f"{s / max(n, 1):>8.0%}  {status}")

    print(f"\n{covered} sources mirrored, {blank} without a text reader "
          f"(they stay aligned and carry blank text)")
    for k in bad:
        print(f"  BROKEN: {k}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
