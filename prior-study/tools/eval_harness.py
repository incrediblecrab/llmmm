#!/usr/bin/env python3
"""Evaluation harness for ingredient embeddings — metrics M1-M5 of the
pre-registration (docs/PREREGISTRATION.md).

Every metric is scored against labels the models never trained on: the 210,612
substitution pairs, and a 10% held-out slice of co-occurrence edges.

    python tools/eval_harness.py --random            # negative control
    python tools/eval_harness.py --emb path/to.npy   # score a model
    python tools/eval_harness.py --random --json out.json

The `--random` mode is the Phase 0 gate. Gaussian noise must score at chance on
M2 and M4; if it does not, the metric is measuring an artefact and every result
computed with it is void. Run it before trusting any model number.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DERIVED = Path(os.environ.get("EPICURE_DERIVED", ROOT / "data" / "derived"))
CATALOG = Path(os.environ.get("EPICURE_CATALOG", ROOT / "data" / "catalog"))

SEED = 20260805
N_TRIPLETS = 20_000
# Two pre-registered tiers instead of one arbitrary cut. "broad" maximises
# statistical power; "strict" requires community agreement. Reporting both
# means a threshold cannot be chosen after seeing results.
TIERS = {"broad": 1, "strict": 10}
D_DEFAULT = 300


# ------------------------------------------------------------------ data
def load_context() -> dict:
    z = np.load(DERIVED / "ii_graph.npz", allow_pickle=True)
    itos = [str(x) for x in z["itos"]]
    stoi = {s: i for i, s in enumerate(itos)}
    ctx = {"itos": itos, "stoi": stoi, "n": len(itos), "uni": z["uni"].astype(float)}

    hp = DERIVED / "ii_graph_heldout.npz"
    if hp.exists():
        h = np.load(hp, allow_pickle=True)
        ctx["held"] = (h["src"].astype(np.int64), h["dst"].astype(np.int64))
    else:
        ctx["held"] = None

    # every observed edge, so link-prediction negatives are true non-edges
    n = ctx["n"]
    lo = np.minimum(z["src"], z["dst"]).astype(np.int64)
    hi = np.maximum(z["src"], z["dst"]).astype(np.int64)
    ctx["edge_set"] = set((lo * n + hi).tolist())
    # graph is stored one-directional and symmetrised at load, so true degree
    # counts both endpoints
    ctx["degree_sym"] = (np.bincount(z["src"].astype(np.int64), minlength=n)
                         + np.bincount(z["dst"].astype(np.int64), minlength=n)
                         ).astype(float)

    subs = _load_subs(stoi)
    ctx["subs"] = subs

    isolated = ctx["degree_sym"] == 0
    ctx["n_isolated"] = int(isolated.sum())
    anchors = {a for a, _ in subs["broad"][0]}
    ctx["iso_anchor_frac"] = (
        float(np.mean([isolated[a] for a in anchors])) if anchors else 0.0)
    return ctx


def _load_subs(stoi: dict) -> dict:
    """Deduplicated substitution pairs per pre-registered vote tier."""
    import pandas as pd
    df = pd.read_parquet(CATALOG / "substitutions.parquet")
    df = df[df.ingredient_vocab.isin(stoi) & df.alternative_vocab.isin(stoi)]
    out = {}
    for tier, min_votes in TIERS.items():
        s = df[df.votes >= min_votes]
        pairs = {(stoi[a], stoi[b])
                 for a, b in zip(s.ingredient_vocab, s.alternative_vocab)
                 if stoi[a] != stoi[b]}
        m: dict = {}
        for ia, ib in pairs:
            m.setdefault(ia, set()).add(ib)
        out[tier] = (sorted(pairs), m)
    return out


# ------------------------------------------------------------------ metrics
def unit(W: np.ndarray) -> np.ndarray:
    return W / np.clip(np.linalg.norm(W, axis=1, keepdims=True), 1e-12, None)


def all_but_top(W: np.ndarray, k: int = 3) -> np.ndarray:
    """Mu & Viswanath's all-but-the-top: centre, then project out the leading
    k principal directions.

    H3 claims popularity lives in those directions. If it does, removing them
    should cut M5 sharply while leaving M2/M4 roughly intact. If M2 collapses
    too, the popularity signal is load-bearing and the cure is worse than the
    disease -- which is a real finding, not a failure.
    """
    X = W - W.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return X - (X @ Vt[:k].T) @ Vt[:k]


def m1_participation_ratio(W: np.ndarray) -> float:
    """Effective dimensionality. Collapse shows up as PR near 1."""
    X = W - W.mean(0)
    lam = np.linalg.svd(X, compute_uv=False) ** 2
    return float(lam.sum() ** 2 / np.maximum((lam ** 2).sum(), 1e-30))


def m2_triplet_accuracy(U: np.ndarray, ctx: dict, tier: str) -> tuple[float, float]:
    """cos(a, known substitute) > cos(a, random ingredient). Chance = 0.50."""
    pairs = ctx["subs"][tier][0]
    if not pairs:
        return float("nan"), float("nan")
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(pairs), N_TRIPLETS)
    a = np.array([pairs[i][0] for i in idx])
    b = np.array([pairs[i][1] for i in idx])
    c = rng.integers(0, ctx["n"], N_TRIPLETS)
    ok = c != a
    pos = np.einsum("ij,ij->i", U[a[ok]], U[b[ok]])
    neg = np.einsum("ij,ij->i", U[a[ok]], U[c[ok]])
    return _prop_ci((pos > neg).mean() + 0.5 * (pos == neg).mean(), int(ok.sum()))


def _prop_ci(p: float, n: int) -> tuple[float, float]:
    """Point estimate and 95% half-width, so two models can be told apart."""
    return float(p), float(1.96 * np.sqrt(max(p * (1 - p), 1e-12) / max(n, 1)))


def m3_recall_at_10(U: np.ndarray, ctx: dict, tier: str) -> float:
    m = {k: v for k, v in ctx["subs"][tier][1].items() if len(v) >= 3}
    if not m:
        return float("nan")
    keys = np.array(sorted(m))
    S = U[keys] @ U.T
    S[np.arange(len(keys)), keys] = -np.inf
    top = np.argpartition(-S, 10, axis=1)[:, :10]
    return float(np.mean([len(m[k] & set(top[i].tolist())) / len(m[k])
                          for i, k in enumerate(keys)]))


def m4_link_auc(U: np.ndarray, ctx: dict) -> tuple[float, float]:
    """Rank held-out edges above degree-matched non-edges. Chance = 0.50.

    Negatives are degree-matched so a model cannot win by memorising which
    ingredients are popular -- that failure mode is what M5 measures.
    """
    if ctx["held"] is None:
        return float("nan"), float("nan")
    hu, hv = ctx["held"]
    n, deg, edges = ctx["n"], ctx["degree_sym"], ctx["edge_set"]
    rng = np.random.default_rng(SEED)

    order = np.argsort(deg)
    rank = np.empty(n, np.int64)
    rank[order] = np.arange(n)
    band = max(n // 10, 1)

    neg = np.empty(len(hv), np.int64)
    for i, v in enumerate(hv):
        r = rank[v]
        cand = v
        for _ in range(40):
            cand = order[int(np.clip(r + rng.integers(-band, band + 1), 0, n - 1))]
            u = hu[i]
            if cand != u and (min(u, cand) * n + max(u, cand)) not in edges:
                break
        neg[i] = cand
    pos_s = np.einsum("ij,ij->i", U[hu], U[hv])
    neg_s = np.einsum("ij,ij->i", U[hu], U[neg])
    return _prop_ci((pos_s > neg_s).mean() + 0.5 * (pos_s == neg_s).mean(), len(hu))


def m5_popularity(W: np.ndarray, ctx: dict) -> dict:
    """Do the leading directions encode frequency? If so, any cosine-aggregating
    ranker becomes a popularity chart."""
    X = W - W.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    proj = X @ Vt[:3].T
    logf = np.log1p(ctx["uni"])
    r = [float(abs(np.corrcoef(proj[:, k], logf)[0, 1])) for k in range(3)]

    U = unit(W)
    score = (U @ U.T).mean(1)
    top_model = set(np.argsort(-score)[:20].tolist())
    top_freq = set(np.argsort(-ctx["uni"])[:20].tolist())
    inter = len(top_model & top_freq)
    return {"pc_freq_corr": r, "max_pc_freq_corr": max(r),
            "top20_freq_jaccard": inter / (40 - inter)}


# ------------------------------------------------------------------ driver
def evaluate(W: np.ndarray, ctx: dict) -> dict:
    U = unit(W)
    auc, auc_ci = m4_link_auc(U, ctx)
    out = {
        "n": int(W.shape[0]), "d": int(W.shape[1]),
        "M1_participation_ratio": m1_participation_ratio(W),
        "M4_link_auc": auc, "M4_link_auc_ci95": auc_ci,
    }
    for tier in TIERS:
        acc, ci = m2_triplet_accuracy(U, ctx, tier)
        out[f"M2_triplet_accuracy_{tier}"] = acc
        out[f"M2_triplet_accuracy_{tier}_ci95"] = ci
        out[f"M3_recall_at_10_{tier}"] = m3_recall_at_10(U, ctx, tier)
    out.update({f"M5_{k}": v for k, v in m5_popularity(W, ctx).items()})
    return out


def render(name: str, r: dict) -> str:
    lines = [f"  {name}",
             f"    M1 participation ratio   {r['M1_participation_ratio']:8.1f}  / {r['d']}"]
    for tier in TIERS:
        lines.append(
            f"    M2 triplet acc [{tier:<6}]  {r[f'M2_triplet_accuracy_{tier}']:8.4f}"
            f"  +/-{r[f'M2_triplet_accuracy_{tier}_ci95']:.4f}  (chance 0.50)")
    for tier in TIERS:
        lines.append(f"    M3 recall@10   [{tier:<6}]  {r[f'M3_recall_at_10_{tier}']:8.4f}")
    lines += [
        f"    M4 held-out link AUC     {r['M4_link_auc']:8.4f}"
        f"  +/-{r['M4_link_auc_ci95']:.4f}  (chance 0.50)",
        f"    M5 max PC-freq corr      {r['M5_max_pc_freq_corr']:8.4f}",
        f"    M5 top20 freq jaccard    {r['M5_top20_freq_jaccard']:8.4f}",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", help="path to an embedding .npy")
    ap.add_argument("--random", action="store_true",
                    help="Phase 0 negative control on Gaussian noise")
    ap.add_argument("--name", default=None)
    ap.add_argument("--json", help="write results as JSON")
    a = ap.parse_args()
    if not a.emb and not a.random:
        sys.exit("give --emb PATH or --random")

    ctx = load_context()
    print(f"vocab {ctx['n']}  ({ctx['n_isolated']} isolated, no co-occurrence edge)")
    for tier, mv in TIERS.items():
        p, m = ctx["subs"][tier]
        print(f"  subs[{tier:<6}] votes>={mv:<3} {len(p):>6,} pairs  "
              f"{sum(1 for v in m.values() if len(v) >= 3):>5,} anchors")
    print(f"  held-out edges {0 if ctx['held'] is None else len(ctx['held'][0]):,}"
          f"   isolated substitution anchors {ctx['iso_anchor_frac']:.1%}\n")

    results = {}
    if a.random:
        rng = np.random.default_rng(SEED)
        for label, W in [
            ("random gaussian", rng.normal(size=(ctx["n"], D_DEFAULT))),
            ("collapsed (rank 1)", rng.normal(size=(ctx["n"], 1))
                                   @ rng.normal(size=(1, D_DEFAULT))),
        ]:
            r = evaluate(W, ctx)
            results[label] = r
            print(render(label, r))
        g = results["random gaussian"]
        checks = {f"M2_{t}": g[f"M2_triplet_accuracy_{t}"] for t in TIERS}
        checks["M4"] = g["M4_link_auc"]
        bad = {k: v for k, v in checks.items() if abs(v - 0.5) >= 0.02}
        print(f"\n  GATE: {'PASS' if not bad else 'FAIL ' + str(bad)}"
              f" — random must score ~0.50 on M2 and M4")
        if bad:
            sys.exit("Phase 0 gate failed: metrics rate noise as signal")

    if a.emb:
        W = np.load(a.emb)
        name = a.name or Path(a.emb).stem
        r = evaluate(W, ctx)
        results[name] = r
        print(render(name, r))

    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
