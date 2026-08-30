#!/usr/bin/env python3
"""Verify the downloaded Epicure artefacts reproduce the paper's published numbers.

Anything that FAILS here is a data-integrity problem worth knowing about BEFORE
any of it gets baked into an iOS bundle.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SIBLINGS = ["cooc", "core", "chem"]

# Published values, model cards + paper Section 3
EXPECTED_PR = {"cooc": 173.6, "core": 94.2, "chem": 183.1}
EXPECTED_MODES = {"cooc": 150, "core": 193, "chem": 200}
EXPECTED_COS = {"cooc": (0.09, 0.13), "core": (0.33, 0.37), "chem": (0.09, 0.13)}

results: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    results.append((label, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{'  — ' + detail if detail else ''}")


def unit(v, axis=-1, eps=1e-9):
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), eps)


def participation_ratio(E_raw: np.ndarray) -> float:
    """PR = (sum eig)^2 / sum(eig^2) of the covariance spectrum."""
    X = E_raw - E_raw.mean(axis=0, keepdims=True)
    eig = np.linalg.svd(X, compute_uv=False) ** 2
    return float(eig.sum() ** 2 / (eig**2).sum())


def load(sib: str):
    d = RAW / f"epicure-{sib}"
    E_raw = load_file(d / "embeddings.safetensors")["embeddings"]
    vocab = json.loads((d / "vocab.json").read_text())
    modes = json.loads((d / "modes.json").read_text())
    poles = json.loads((d / "supervised_poles.json").read_text())
    cfg = json.loads((d / "config.json").read_text())
    return E_raw, vocab, modes, poles, cfg


def slerp_query(E: np.ndarray, v: np.ndarray, d: np.ndarray, theta_deg: float) -> np.ndarray:
    d = unit(np.asarray(d, dtype=np.float32))
    d_perp = d - (d @ v) * v
    n = np.linalg.norm(d_perp)
    if n < 1e-9:
        return v
    d_perp = d_perp / n
    th = np.deg2rad(float(theta_deg))
    return unit(np.cos(th) * v + np.sin(th) * d_perp)


def main() -> int:
    store = {}

    for sib in SIBLINGS:
        print(f"\n=== epicure-{sib} ===")
        E_raw, vocab, modes, poles, cfg = load(sib)
        E = unit(E_raw.astype(np.float32))
        itos = {i: n for n, i in vocab.items()}
        store[sib] = (E, vocab, itos, modes, poles)

        check(f"shape 1790x300", E_raw.shape == (1790, 300), str(E_raw.shape))
        check("vocab size 1790", len(vocab) == 1790, str(len(vocab)))
        check("config d_model/vocab agree",
              cfg["d_model"] == 300 and cfg["vocab_size"] == 1790)
        check("embeddings NOT pre-normalised on disk (card claim)",
              not np.allclose(np.linalg.norm(E_raw, axis=1), 1.0, atol=1e-3),
              f"mean L2 norm {np.linalg.norm(E_raw, axis=1).mean():.3f}")
        check("no NaN/Inf", bool(np.isfinite(E_raw).all()))

        pr = participation_ratio(E_raw)
        exp = EXPECTED_PR[sib]
        check(f"participation ratio ~= {exp}", abs(pr - exp) / exp < 0.05, f"got {pr:.1f}")

        # average pairwise cosine over the normalised matrix, excluding diagonal
        S = E @ E.T
        n = S.shape[0]
        avg_cos = float((S.sum() - np.trace(S)) / (n * (n - 1)))
        lo, hi = EXPECTED_COS[sib]
        check(f"avg pairwise cosine in [{lo},{hi}]", lo <= avg_cos <= hi, f"got {avg_cos:.3f}")

        check(f"mode count == {EXPECTED_MODES[sib]}", len(modes) == EXPECTED_MODES[sib],
              str(len(modes)))
        kinds = sorted({m["kind"] for m in modes})
        props = sorted({m["property"] for m in modes})
        print(f"       mode kinds: {kinds}")
        print(f"       properties: {len(props)}")
        print(f"       supervised pole keys: {len(poles)}")
        cui = [k for k in poles if k.startswith("cuisine:")]
        print(f"       cuisine poles ({len(cui)}): {sorted(cui)}")

    # ---- golden SLERP reproduction against the paper's own table ----
    print("\n=== SLERP golden-value reproduction (paper_slerp_results.csv) ===")
    for sib in SIBLINGS:
        E, vocab, itos, modes, poles = store[sib]
        csv = RAW / f"epicure-{sib}" / "paper_slerp_results.csv"
        df = pd.read_csv(csv)
        print(f"\n  epicure-{sib}: {len(df)} rows, cols={list(df.columns)}")
        print(df.head(3).to_string(index=False))
        break  # structure is identical across siblings; inspect one

    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    code = main()
    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\n{'=' * 60}\n{n_pass}/{len(results)} checks passed")
    sys.exit(code)
