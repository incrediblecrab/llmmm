#!/usr/bin/env python3
"""Re-attempt the paper's SLERP table using the AUTHOR'S OWN direction
reconstruction, recovered from the epicure-explorer Space source
(`app.py::_aggregate_pole`).

Earlier (`reproducibility_audit.py`) we concluded 90% of `direction_arithmetic_full`
was irreproducible because no `sweet` / `nutty` / `high protein` vectors ship.
The Space reveals the author builds them on the fly as the unit-mean of every
supervised pole whose key starts with a given property prefix. Modes are the HIGH
end of each property (98% have prop_z_mean > 0), so that mean is a valid
"high X" direction.

This script tests whether that recipe reproduces the published table.
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

# Author's mapping (app.py::_SENSORY_SLIDER_KEYS), extended to cover every
# direction that appears in the published table.
DIRECTION_PREFIXES = {
    "sweet":        ["sweet_score/", "cf_sweet/"],
    "sour":         ["sour_score/", "cf_sour/"],
    "bitter":       ["bitter_score/", "cf_bitter/"],
    "umami":        ["umami_score/", "cf_umami/", "cf_meaty/"],
    "fatty":        ["fatty_score/", "cf_fatty/"],
    "pungent":      ["pungent_score/", "cf_pungent/"],
    "savory":       ["cf_savory/"],
    "citrus":       ["cf_citrus/"],
    "woody":        ["cf_woody/"],
    "earthy":       ["cf_earthy/"],
    "meat direction": ["cf_meaty/", "umami_score/"],
    "high protein": ["usda_protein_g/"],
    "high fiber":   ["usda_fiber_g/"],
    "high sugars":  ["usda_sugars_g/"],
    "high fat":     ["fatty_score/", "cf_fatty/", "usda_caloric_density/"],
    # no matching pole family ships for these:
    "floral":       [],
    "nutty":        [],
    "high water":   [],
}


def unit(v, axis=-1, eps=1e-9):
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), eps)


def slerp(a, b, deg):
    """Author's semantics (epicure.py): rotate a toward b by `deg` degrees."""
    a, b = unit(a), unit(b)
    b_perp = b - float(a @ b) * a
    n = np.linalg.norm(b_perp)
    if n < 1e-8:
        return a
    b_perp = b_perp / n
    t = np.deg2rad(deg)
    return unit(np.cos(t) * a + np.sin(t) * b_perp)


def load(sib):
    d = RAW / f"epicure-{sib}"
    E = unit(load_file(d / "embeddings.safetensors")["embeddings"].astype(np.float32))
    vocab = json.loads((d / "vocab.json").read_text())
    poles = {k: np.asarray(v, dtype=np.float32)
             for k, v in json.loads((d / "supervised_poles.json").read_text()).items()}
    return E, vocab, poles


def aggregate_pole(poles, prefixes):
    keys = [k for p in prefixes for k in poles if k.startswith(p)]
    if not keys:
        return None, 0
    return unit(np.stack([unit(poles[k]) for k in keys]).mean(0)), len(keys)


def direction_vector(poles, direction):
    """Cuisine directions are single poles; everything else is an aggregate."""
    key = f"cuisine:{direction}"
    if key in poles:
        return unit(poles[key]), 1
    if direction in ("whole", "processed"):
        return aggregate_pole(poles, ["nova_level/"])
    return aggregate_pole(poles, DIRECTION_PREFIXES.get(direction, []))


def parse_case(tc):
    """'chicken + Japanese' -> ('chicken', 'Japanese');
       'chicken + whole + Mediterranean' -> ('chicken', 'Mediterranean') [2-step]."""
    parts = [p.strip() for p in tc.split("+")]
    return parts[0], parts[1:]


def main():
    df = pd.read_parquet(RAW / "epicure-corpus-resources" / "data" /
                         "direction_arithmetic_full.parquet")
    rows = []
    for sib in SIBLINGS:
        E, vocab, poles = load(sib)
        sub = df[df.model == sib]
        for (tc, ang), g in sub.groupby(["test_case", "angle_deg"], sort=False):
            seed, steps = parse_case(tc)
            if seed not in vocab:
                rows.append((sib, tc, ang, None, "seed not in vocab"))
                continue
            q = E[vocab[seed]].copy()
            missing = []
            for st in steps:
                vec, nk = direction_vector(poles, st)
                if vec is None:
                    missing.append(st)
                    continue
                if ang:
                    q = slerp(q, vec, ang / len(steps) if len(steps) > 1 else ang)
            if missing:
                rows.append((sib, tc, ang, None, f"no pole family for {missing}"))
                continue
            sims = E @ unit(q)
            sims[vocab[seed]] = -np.inf
            inv = {v: k for k, v in vocab.items()}
            got = [inv[i] for i in np.argsort(-sims)[:5]]
            want = g.sort_values("hit_rank").hit_name.tolist()[:5]
            overlap = len(set(got) & set(want)) / 5
            rows.append((sib, tc, ang, overlap,
                         "exact" if got == want else f"{overlap:.0%} overlap"))

    R = pd.DataFrame(rows, columns=["model", "test_case", "angle", "overlap", "note"])
    cov = R.overlap.notna()
    print(f"coverage: {cov.sum()}/{len(R)} case-angle-model cells reconstructible "
          f"({100*cov.mean():.1f}%)   [was 10% before the Space recipe]")
    print("\nmean top-5 overlap with published table, by angle:")
    print(R[cov].groupby("angle").overlap.agg(["mean", "count"]).to_string())
    print("\nby model:")
    print(R[cov].groupby("model").overlap.agg(["mean", "count"]).to_string())
    exact = R[cov].groupby("angle").apply(
        lambda g: (g.note == "exact").mean(), include_groups=False)
    print("\nexact top-5 match rate by angle:")
    print(exact.to_string())
    print("\nstill unreconstructible:")
    print(R[~cov].note.value_counts().to_string())

    out = ROOT / "data" / "derived" / "slerp_reconstruction.csv"
    R.to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
