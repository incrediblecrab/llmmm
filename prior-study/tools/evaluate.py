#!/usr/bin/env python3
"""Compare our retrained embeddings against the published Epicure models.

Judging embeddings by loss alone is meaningless -- a model can drive SGNS loss
down while collapsing every ingredient onto one direction. The paper reports
specific geometric fingerprints, so those are what we check:

  * participation ratio -- how many of the 300 dimensions are actually used.
    The paper's headline result is that Core is markedly more anisotropic
    (94.2/300) than Cooc or Chem (~180), because blending chemistry with recipe
    context concentrates variance. Reproducing the *ordering* matters more than
    hitting the exact number, since that depends on corpus and threshold.
  * average pairwise cosine (~0.35 for Core) -- the companion statistic; a
    degenerate model pushes this toward 1.0.
  * nearest-neighbour agreement with the published model, per ingredient.
    This is the only metric that checks we learned the same *semantics* rather
    than merely a well-conditioned random projection.

Neighbour overlap is computed on the shared vocabulary in L2-normalised space,
per the published config's note to normalise before any cosine operation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "data" / "raw"
MODELS = ROOT / "models"

PROBES = ["garlic", "tomato", "basil", "chocolate", "cinnamon", "soy_sauce",
          "lemon", "ginger", "beef", "mushroom", "coconut_milk", "saffron"]


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-9, None)


def participation_ratio(X: np.ndarray) -> float:
    """(sum eigenvalue)^2 / sum(eigenvalue^2) of the covariance -- the effective
    number of dimensions carrying variance."""
    Xc = X - X.mean(0, keepdims=True)
    ev = np.linalg.svd(Xc, compute_uv=False) ** 2
    return float(ev.sum() ** 2 / np.clip((ev ** 2).sum(), 1e-12, None))


def avg_pairwise_cos(X: np.ndarray, n: int = 1500, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    U = l2(X[idx])
    G = U @ U.T
    iu = np.triu_indices(len(U), k=1)
    return float(G[iu].mean())


def load_published(variant: str):
    # tolerate sweep tags like "cooc-i4.0" -> published model "epicure-cooc"
    d = PUB / f"epicure-{variant.split(chr(45))[0]}"
    if not d.exists():
        return None, None
    from safetensors.numpy import load_file
    t = load_file(str(d / "embeddings.safetensors"))
    W = next(iter(t.values()))
    itos = json.loads((d / "itos.json").read_text())
    if isinstance(itos, dict):
        # published as {"0": "abalone", "1": ...}; keys are strings, so sorting
        # them lexically would put "10" before "2" and shuffle the vocabulary
        itos = [v for _, v in sorted(itos.items(), key=lambda kv: int(kv[0]))]
    return np.asarray(W, np.float32), list(itos)


def neighbours(W: np.ndarray, i: int, k: int) -> np.ndarray:
    s = W @ W[i]
    s[i] = -2
    return np.argpartition(-s, k)[:k]


def compare(ours: np.ndarray, our_itos, variant: str, k: int = 10):
    pub, pub_itos = load_published(variant)
    print(f"\n=== {variant} ===")
    print(f"  ours  shape {ours.shape}  PR {participation_ratio(ours):7.1f}"
          f"  avg-cos {avg_pairwise_cos(ours):+.3f}")
    if pub is None:
        print("  (no published model to compare against)")
        return
    print(f"  paper shape {pub.shape}   PR {participation_ratio(pub):7.1f}"
          f"  avg-cos {avg_pairwise_cos(pub):+.3f}")

    pidx = {t: i for i, t in enumerate(pub_itos)}
    shared = [(i, pidx[t]) for i, t in enumerate(our_itos) if t in pidx]
    if not shared:
        print("  no shared vocabulary")
        return
    A = l2(ours)
    B = l2(pub)
    oi = np.array([a for a, _ in shared])
    pi = np.array([b for _, b in shared])
    o2p = dict(zip(oi.tolist(), pi.tolist()))

    ov = []
    for a, b in shared:
        na = {o2p.get(x) for x in neighbours(A, a, k)}
        nb = set(neighbours(B, b, k).tolist())
        ov.append(len(na & nb) / k)
    ov = np.array(ov)
    rng = np.random.default_rng(0)
    chance = k / len(shared)
    print(f"  top-{k} neighbour overlap  mean {ov.mean():.3f}"
          f"  median {np.median(ov):.3f}  (chance {chance:.4f})"
          f"  lift x{ov.mean()/max(chance,1e-9):.0f}")

    print(f"\n  {'ingredient':16}{'ours':46}{'paper'}")
    name_o = {t: i for i, t in enumerate(our_itos)}
    for p in PROBES:
        if p not in name_o or p not in pidx:
            continue
        a = ", ".join(our_itos[j] for j in
                      neighbours(A, name_o[p], 5))[:44]
        b = ", ".join(pub_itos[j] for j in neighbours(B, pidx[p], 5))[:44]
        print(f"  {p:16}{a:46}{b}")


def main(variants) -> None:
    itos = list(np.load(ROOT / "data" / "derived" / "ii_graph.npz",
                        allow_pickle=True)["itos"])
    for v in variants:
        f = MODELS / f"epicure_{v}.npy"
        if not f.exists():
            print(f"skip {v}: {f.relative_to(ROOT)} not found")
            continue
        compare(np.load(f), itos, v)


if __name__ == "__main__":
    main(sys.argv[1:] or ["cooc", "chem", "core"])
