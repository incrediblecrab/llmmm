#!/usr/bin/env python3
"""Adversarial check on the H4 verdict logic in summarize.py.

A verdict function that always returns FALSIFIED would "correctly" mark the real
data and still be worthless. This feeds it three payloads with known answers:

  real            -> the measured corpus, expected FALSIFIED
  ahn_conforming  -> signs forced to match the pre-registered list, expect SUPPORTED
  inverted        -> every sign flipped, expected FALSIFIED

Passing requires the middle case to flip. Run:
    python tools/test_h4_verdict.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_cuisines_v2 import PREREG  # noqa: E402
from summarize import h4  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def load_real() -> dict:
    for name in ("cuisine_pairing_v2", "cuisines-full", "cuisines"):
        p = RESULTS / f"{name}.json"
        if p.exists():
            return {name: json.loads(p.read_text())}
    raise SystemExit("no cuisine results found")


def force(payload: dict, sign_of) -> dict:
    """Rewrite each pre-registered cuisine's delta to a chosen sign."""
    out = copy.deepcopy(payload)
    key = next(iter(out))
    for c, r in out[key]["cuisines"].items():
        if c not in PREREG:
            continue
        s = sign_of(c)
        mag = max(abs(r["delta"]), 0.5)
        r["delta"] = s * mag
        r["rel_delta"] = s * abs(r["rel_delta"] or 0.01)
        if "delta_ci_lo" in r:
            r["delta_ci_lo"], r["delta_ci_hi"] = sorted(
                (s * mag * 0.5, s * mag * 1.5))
            r["sign_resolved"] = True
    return out


def result_of(payload: dict) -> str:
    text = "\n".join(h4(payload))
    if "**SUPPORTED**" in text:
        return "SUPPORTED"
    if "**FALSIFIED**" in text:
        return "FALSIFIED"
    return "PENDING"


def main() -> None:
    real = load_real()
    cases = [
        ("real", real, "FALSIFIED"),
        ("ahn_conforming", force(real, lambda c: PREREG[c]), "SUPPORTED"),
        ("inverted", force(real, lambda c: -PREREG[c]), "FALSIFIED"),
    ]
    bad = 0
    for name, payload, want in cases:
        got = result_of(payload)
        ok = got == want
        bad += not ok
        print(f"{name:16} want {want:10} got {got:10} {'ok' if ok else 'FAIL'}")
    if bad:
        raise SystemExit(f"{bad} case(s) failed: verdict logic does not discriminate")
    print("\nverdict logic discriminates correctly")


if __name__ == "__main__":
    main()
