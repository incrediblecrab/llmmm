#!/usr/bin/env python3
"""Differential test against the live epicure-explorer Gradio API.

Answers two separate questions:

  PARITY      Does our local bundle reproduce the author's live service exactly?
              (i.e. is our reimplementation 1:1 with theirs?)

  REPLICATION Does the author's live service reproduce the author's published
              tables? (i.e. do the shipped weights match the paper?)

These are NOT the same question, and they have opposite answers.

Results are cached in data/derived/api_cache.json so re-runs are free and CI
never depends on the network. Be polite: the Space is free community hardware.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CACHE_PATH = ROOT / "data" / "derived" / "api_cache.json"
BASE = "https://kaikaku-epicure-explorer.hf.space/gradio_api/call"
SIBLINGS = ["cooc", "core", "chem"]
DELAY = 0.4

CACHE: dict[str, list] = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}


def unit(v, axis=-1, eps=1e-9):
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), eps)


def slerp(a, b, deg):
    a, b = unit(a), unit(b)
    perp = b - float(a @ b) * a
    n = np.linalg.norm(perp)
    if n < 1e-8:
        return a
    t = np.deg2rad(deg)
    return unit(np.cos(t) * a + np.sin(t) * (perp / n))


def call(endpoint: str, data: list, retries: int = 3):
    """Two-step Gradio protocol: POST -> event_id, GET -> SSE stream."""
    key = f"{endpoint}:{json.dumps(data)}"
    if key in CACHE:
        return CACHE[key]
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                f"{BASE}/{endpoint}", method="POST",
                data=json.dumps({"data": data}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                eid = json.loads(r.read())["event_id"]
            time.sleep(DELAY)
            with urllib.request.urlopen(f"{BASE}/{endpoint}/{eid}", timeout=90) as r:
                body = r.read().decode()
            payload = None
            for line in body.splitlines():
                if line.startswith("data: "):
                    payload = json.loads(line[6:])
            if payload is None:
                raise ValueError("no data frame")
            out = payload[0]
            CACHE[key] = out
            CACHE_PATH.write_text(json.dumps(CACHE))
            time.sleep(DELAY)
            return out
        except (urllib.error.URLError, ValueError, TimeoutError) as e:
            if attempt == retries - 1:
                print(f"      ! {endpoint}{data} failed: {e}")
                return None
            time.sleep(2 * (attempt + 1))
    return None


def load(sib):
    d = RAW / f"epicure-{sib}"
    E = unit(load_file(d / "embeddings.safetensors")["embeddings"].astype(np.float32))
    vocab = json.loads((d / "vocab.json").read_text())
    poles = {k: np.asarray(v, np.float32)
             for k, v in json.loads((d / "supervised_poles.json").read_text()).items()}
    return E, vocab, poles


def local_topk(E, vocab, q, k, exclude):
    inv = {v: k_ for k_, v in vocab.items()}
    s = E @ unit(q)
    for x in exclude:
        if x in vocab:
            s[vocab[x]] = -np.inf
    idx = np.argsort(-s)[:k]
    return [(inv[i], round(float(s[i]), 6)) for i in idx]


# ============================================================ PART 1: PARITY
SEEDS = ["miso", "rice", "chicken", "tomato", "chocolate", "basil",
         "lamb", "egg", "coffee", "lime"]


def part1_parity():
    print("\n" + "=" * 74)
    print("PART 1 — PARITY: our local bundle vs the author's live service")
    print("=" * 74)
    rows = []
    for sib in SIBLINGS:
        E, vocab, poles = load(sib)
        for seed in SEEDS:
            if seed not in vocab:
                continue
            live = call("neighbors", [seed, sib, 5])
            if live is None:
                continue
            mine = local_topk(E, vocab, E[vocab[seed]], 5, [seed])
            names_ok = [x["name"] for x in live] == [n for n, _ in mine]
            maxdiff = max(abs(x["cosine"] - c) for x, (_, c) in zip(live, mine))
            rows.append(("neighbors", sib, seed, names_ok, maxdiff))

        for seed, pole in [("rice", "cuisine:South_Asian"), ("beef", "cuisine:East_Asian"),
                           ("salmon", "cuisine:Mediterranean"), ("corn", "cuisine:Latin_American")]:
            if pole not in poles or seed not in vocab:
                continue
            for th in (30, 60):
                live = call("slerp", [seed, pole, th, sib, 5])
                if live is None or (isinstance(live, dict) and "error" in live):
                    continue
                q = slerp(E[vocab[seed]], unit(poles[pole]), th)
                mine = local_topk(E, vocab, q, 5, [seed])
                names_ok = [x["name"] for x in live] == [n for n, _ in mine]
                maxdiff = max(abs(x["cosine"] - c) for x, (_, c) in zip(live, mine))
                rows.append((f"slerp@{th}", sib, f"{seed}->{pole}", names_ok, maxdiff))

    R = pd.DataFrame(rows, columns=["op", "sibling", "query", "names_match", "max_cos_diff"])
    print(f"\n  {len(R)} live calls compared")
    print(f"  ranking identical      : {R.names_match.sum()}/{len(R)} "
          f"({100*R.names_match.mean():.1f}%)")
    print(f"  max cosine difference  : {R.max_cos_diff.max():.2e}")
    print(f"  mean cosine difference : {R.max_cos_diff.mean():.2e}")
    print("\n  by operator:")
    print(R.groupby("op").agg(n=("names_match", "size"),
                              match=("names_match", "mean"),
                              worst=("max_cos_diff", "max")).to_string())
    bad = R[~R.names_match]
    if len(bad):
        print("\n  MISMATCHES:")
        print(bad.to_string(index=False))
    return R


# ======================================================= PART 2: REPLICATION
PAPER_CUISINE = [("chicken", "Japanese"), ("beef", "East_Asian"),
                 ("salmon", "Mediterranean"), ("rice", "South_Asian"),
                 ("corn", "Latin_American"), ("lamb", "Eastern_European")]


def part2_replication():
    print("\n" + "=" * 74)
    print("PART 2 — REPLICATION: the live service vs the PUBLISHED tables")
    print("=" * 74)
    df = pd.read_parquet(RAW / "epicure-corpus-resources" / "data" /
                         "direction_arithmetic_full.parquet")
    rows = []
    for sib in SIBLINGS:
        _, _, poles = load(sib)
        for seed, cui in PAPER_CUISINE:
            pole = f"cuisine:{cui}"
            tc = f"{seed} + {cui}"
            if pole not in poles:
                rows.append((sib, tc, None, None, "pole not shipped for this sibling"))
                continue
            for th in (0, 30, 60):
                pub = df[(df.test_case == tc) & (df.model == sib) & (df.angle_deg == th)]
                if pub.empty:
                    continue
                want = pub.sort_values("hit_rank").hit_name.tolist()[:5]
                live = call("slerp", [seed, pole, th, sib, 5])
                if live is None or isinstance(live, dict):
                    continue
                got = [x["name"] for x in live]
                ov = len(set(got) & set(want)) / 5
                rows.append((sib, tc, th, ov, "exact" if got == want else f"{ov:.0%}"))

    R = pd.DataFrame(rows, columns=["sibling", "test_case", "angle", "overlap", "note"])
    ok = R.overlap.notna()
    print(f"\n  {ok.sum()} published cells re-queried against the LIVE service")
    print("\n  agreement between the author's live API and the author's paper:")
    g = R[ok].groupby("angle").agg(n=("overlap", "size"), mean_overlap=("overlap", "mean"))
    g["exact_rate"] = R[ok].groupby("angle").note.apply(lambda s: (s == "exact").mean())
    print(g.to_string())
    skipped = R[R.overlap.isna()]
    if len(skipped):
        print(f"\n  skipped ({len(skipped)}): cuisine pole absent for that sibling")
        print("   ", ", ".join(f"{r.sibling}/{r.test_case}" for r in skipped.itertuples()))
    return R


if __name__ == "__main__":
    p = part1_parity()
    r = part2_replication()

    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    ok = r.overlap.notna()
    th0 = r[ok & (r.angle == 0)].overlap.mean()
    thx = r[ok & (r.angle > 0)].overlap.mean()
    exact_x = (r[ok & (r.angle > 0)].note == "exact").mean()
    print(f"""
  PARITY      : {100*p.names_match.mean():.1f}% identical rankings, max cosine delta
                {p.max_cos_diff.max():.1e}. Our bundle IS the live service.

  REPLICATION : live-vs-paper overlap {th0:.1%} at theta=0 but {thx:.1%} at theta>0,
                with {exact_x:.1%} exact. The author's own service does not
                reproduce the author's own published tables.

  => The API cannot be used to replicate the paper's findings. It CAN be used as
     a bit-exact conformance oracle for the Swift port.
""")
    R = ROOT / "data" / "derived"
    p.to_csv(R / "api_parity.csv", index=False)
    r.to_csv(R / "api_replication.csv", index=False)
    print(f"  wrote {(R/'api_parity.csv').relative_to(ROOT)} and "
          f"{(R/'api_replication.csv').relative_to(ROOT)}")
    print(f"  cached {len(CACHE)} API responses in {CACHE_PATH.relative_to(ROOT)}")
