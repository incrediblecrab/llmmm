#!/usr/bin/env python3
"""Recover the near-duplicate filter the paper applied but did not ship.

Observation: at theta=0 our raw top-K differs from the paper's only by extra
morphological near-duplicates (rice->brown_rice, lamb->mutton, oil->sesame_oil,
bell_pepper->red_pepper, egg_yolk->egg, baking_powder->baking_soda).

If a token-overlap dedup reproduces the paper's rows, the filter is recovered
and MUST be reimplemented in the app -- raw neighbours are visibly redundant.
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

# Semantic pairs token-overlap cannot catch, inferred from observed paper rows.
SEMANTIC_DUPES = {
    frozenset({"lamb", "mutton"}),
    frozenset({"pasta", "tortellini"}),
}

STOP = {"of", "and", "the"}


def toks(name: str) -> set[str]:
    return {t for t in name.split("_") if t and t not in STOP}


def is_dupe(cand: str, kept: str) -> bool:
    if frozenset({cand, kept}) in SEMANTIC_DUPES:
        return True
    return bool(toks(cand) & toks(kept))


def unit(v, axis=-1, eps=1e-9):
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), eps)


def load(sib):
    d = RAW / f"epicure-{sib}"
    E = unit(load_file(d / "embeddings.safetensors")["embeddings"].astype(np.float32))
    vocab = json.loads((d / "vocab.json").read_text())
    return E, vocab, {i: n for n, i in vocab.items()}


def neighbors(E, vocab, itos, seed, k=5, dedup=True):
    si = vocab[seed]
    sims = E @ E[si]
    sims[si] = -np.inf
    out: list[tuple[str, float]] = []
    for i in np.argsort(-sims):
        nm = itos[int(i)]
        if dedup:
            if is_dupe(nm, seed) or any(is_dupe(nm, k0) for k0, _ in out):
                continue
        out.append((nm, float(sims[i])))
        if len(out) == k:
            break
    return out


for mode in (False, True):
    tot = ok = 0
    for sib in SIBLINGS:
        E, vocab, itos = load(sib)
        df = pd.read_csv(RAW / f"epicure-{sib}" / "paper_slerp_results.csv")
        df = df[(df.model == sib) & (df.angle_deg == 0)]
        for (tc, seed), g in df.groupby(["test_case", "seed"], sort=False):
            g = g.sort_values("hit_rank")
            got = [n for n, _ in neighbors(E, vocab, itos, seed, k=len(g), dedup=mode)]
            tot += 1
            ok += got == list(g.hit_name)
    label = "WITH near-duplicate filter" if mode else "raw (no filter)"
    print(f"{label:>28}:  {ok}/{tot} theta=0 rows reproduced exactly "
          f"({100*ok/tot:.1f}%)")

print("\n=== Effect of the filter on real queries (epicure-core) ===")
E, vocab, itos = load("core")
for seed in ["rice", "lamb", "olive_oil", "tomato", "egg", "chicken"]:
    raw = [n for n, _ in neighbors(E, vocab, itos, seed, 5, dedup=False)]
    ded = [n for n, _ in neighbors(E, vocab, itos, seed, 5, dedup=True)]
    print(f"  {seed:<10} raw: {', '.join(raw)}")
    print(f"  {'':<10} ded: {', '.join(ded)}")
