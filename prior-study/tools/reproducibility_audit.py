#!/usr/bin/env python3
"""Establish exactly WHICH published rows the shipped weights can reproduce.

Conclusion drives what can be used as a Swift acceptance test.
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
    return [(itos[int(i)], float(sims[i])) for i in np.argsort(-sims)[:k]]


rows = []
for sib in SIBLINGS:
    E, vocab, itos, poles = load(sib)
    df = pd.read_csv(RAW / f"epicure-{sib}" / "paper_slerp_results.csv")
    df = df[df.model == sib]
    for (tc, seed, ang), g in df.groupby(["test_case", "seed", "angle_deg"], sort=False):
        target = tc.split("+", 1)[1].strip()
        key = "cuisine:" + target.replace(" ", "_")
        kind = "cuisine" if key in poles else "continuous/compound"
        if key not in poles:
            rows.append(dict(sib=sib, case=tc, angle=ang, kind=kind,
                             status="NO_POLE_SHIPPED", names_match=None))
            continue
        g = g.sort_values("hit_rank")
        got = slerp_topk(E, vocab, itos, seed, poles[key], float(ang), k=len(g))
        nm = [n for n, _ in got] == list(g.hit_name)
        sm = np.allclose([s for _, s in got], list(g.hit_sim), atol=5e-4)
        rows.append(dict(sib=sib, case=tc, angle=ang, kind=kind,
                         status="EXACT" if (nm and sm) else "MISMATCH", names_match=nm))

r = pd.DataFrame(rows)

print("=== Reproducibility of paper_slerp_results.csv from shipped weights ===\n")
print(r.groupby(["kind", "status"]).size().to_string())

print("\n=== Cuisine cases broken down BY ANGLE (the decisive test) ===")
cu = r[r.kind == "cuisine"]
print(pd.crosstab(cu.angle, cu.status).to_string())

print("\n=== Per-sibling, cuisine cases only ===")
print(pd.crosstab([cu.sib, cu.angle], cu.status).to_string())

n_nopole = (r.status == "NO_POLE_SHIPPED").sum()
print(f"\nRows whose direction vector is NOT in the release: {n_nopole}/{len(r)}"
      f"  ({100*n_nopole/len(r):.0f}%)")

# ---- Emit the fixture that IS trustworthy: theta=0 == pure neighbours ----
print("\n=== Building Swift acceptance fixture from verifiable rows ===")
fixture = {"note": "theta=0 rows equal pure top-K neighbours and reproduce exactly; "
                   "theta>0 cuisine rows do NOT reproduce because shipped cuisine poles "
                   "are heuristic reconstructions, not the paper's originals.",
           "neighbors": {}, "slerp_reference": {}}
for sib in SIBLINGS:
    E, vocab, itos, poles = load(sib)
    fixture["neighbors"][sib] = {
        s: [[n, round(v, 6)] for n, v in slerp_topk(E, vocab, itos, s, E[vocab[s]], 0.0, k=10)]
        for s in ["chicken", "rice", "miso", "tomato", "chocolate", "olive_oil"]
    }
    fixture["slerp_reference"][sib] = {
        f"rice|cuisine:South_Asian|{a}":
            [[n, round(v, 6)] for n, v in
             slerp_topk(E, vocab, itos, "rice", poles["cuisine:South_Asian"], a, k=5)]
        for a in (0, 30, 60, 90)
    }
out = ROOT / "data" / "derived" / "golden_fixture.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(fixture, indent=2))
print(f"  wrote {out.relative_to(ROOT)}  ({out.stat().st_size/1024:.1f} KB)")
