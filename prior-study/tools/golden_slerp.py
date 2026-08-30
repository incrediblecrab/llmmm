#!/usr/bin/env python3
"""Reproduce paper_slerp_results.csv exactly from the shipped weights.

If this passes, the operator math is pinned and can be ported to Swift with
these rows as the acceptance test.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SIBLINGS = ["cooc", "core", "chem"]


def unit(v, axis=-1, eps=1e-9):
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), eps)


def load(sib):
    d = RAW / f"epicure-{sib}"
    E = unit(load_file(d / "embeddings.safetensors")["embeddings"].astype(np.float32))
    vocab = json.loads((d / "vocab.json").read_text())
    poles = {k: np.array(v, np.float32) for k, v in
             json.loads((d / "supervised_poles.json").read_text()).items()}
    return E, vocab, {i: n for n, i in vocab.items()}, poles


def slerp_topk(E, vocab, itos, seed, d, theta_deg, k=5):
    si = vocab[seed]
    v = E[si]
    d = unit(np.asarray(d, np.float32))
    dp = d - (d @ v) * v
    n = np.linalg.norm(dp)
    q = v if n < 1e-9 else unit(np.cos(np.deg2rad(theta_deg)) * v
                                + np.sin(np.deg2rad(theta_deg)) * (dp / n))
    sims = E @ q
    sims[si] = -np.inf
    order = np.argsort(-sims)[:k]
    return [(itos[int(i)], float(sims[i])) for i in order]


def resolve_direction(test_case: str, poles: dict):
    """'chicken + Japanese' -> the pole key whose suffix matches 'Japanese'."""
    target = test_case.split("+", 1)[1].strip()
    key = "cuisine:" + target.replace(" ", "_")
    if key in poles:
        return key
    for k in poles:
        if k.split(":", 1)[-1].replace("_", " ").lower() == target.lower():
            return k
    return None


def main():
    grand_ok = grand_tot = 0
    unresolved: dict[str, set] = {}

    for sib in SIBLINGS:
        E, vocab, itos, poles = load(sib)
        df = pd.read_csv(RAW / f"epicure-{sib}" / "paper_slerp_results.csv")
        df = df[df.model == sib]

        ok = tot = 0
        miss = set()
        name_ok = sim_ok = 0

        for (tc, seed, ang), g in df.groupby(["test_case", "seed", "angle_deg"], sort=False):
            key = resolve_direction(tc, poles)
            if key is None:
                miss.add(tc.split("+", 1)[1].strip())
                continue
            g = g.sort_values("hit_rank")
            got = slerp_topk(E, vocab, itos, seed, poles[key], float(ang), k=len(g))
            exp_names = list(g.hit_name)
            exp_sims = list(g.hit_sim)
            got_names = [n for n, _ in got]
            got_sims = [s for _, s in got]
            tot += 1
            if got_names == exp_names:
                name_ok += 1
                if np.allclose(got_sims, exp_sims, atol=5e-4):
                    sim_ok += 1
                    ok += 1

        grand_ok += ok
        grand_tot += tot
        unresolved[sib] = miss
        pct = 100 * ok / tot if tot else 0
        print(f"epicure-{sib:5s}  exact(name+sim): {ok}/{tot} ({pct:5.1f}%)   "
              f"names-only: {name_ok}/{tot}")
        if miss:
            print(f"               unresolved directions: {sorted(miss)}")

    print(f"\nTOTAL exact reproduction: {grand_ok}/{grand_tot} "
          f"({100*grand_ok/max(grand_tot,1):.1f}%)")

    # Show one concrete worked example for the Swift port acceptance test
    print("\n=== Golden example for Swift acceptance test ===")
    E, vocab, itos, poles = load("core")
    for ang in (0, 30, 60, 90):
        got = slerp_topk(E, vocab, itos, "rice", poles["cuisine:South_Asian"], ang, k=5)
        print(f"  core  rice +{ang:>2}deg -> South_Asian: "
              + ", ".join(f"{n}({s:.4f})" for n, s in got))


if __name__ == "__main__":
    main()
