#!/usr/bin/env python3
"""Reference implementation of the shipping ranker. Swift must match this.

This file is normative. Where this disagrees with prose in any document,
this file wins. `golden` emits fixtures the Swift port asserts against.

The algorithm is the one validated in docs/PREREGISTRATION.md A5:

    score(c) = SUM_{b in basket, npmi(b,c) >= FLOOR} npmi(b, c)
             + BETA * cos(normalise(mean_{b in basket} emb[b]), emb[c])

Both terms are load-bearing and the constants are not free parameters:

FLOOR (-0.15) is inherited from the graph construction and H8 proved it is
load-bearing -- training on the unpruned graph moved M4 from 0.843 to 0.762 and
pushed the popularity metric from 0.443 to 0.693. The pairs it removes are
popularity noise ("co-occurs with salt"). Do not lower it to gain coverage.

BETA (0.5) was fitted on the 100k-recipe holdout. It is a compromise, chosen
deliberately: beta=2.0 scores better overall (0.4269 vs 0.3909 recall@10) but
*worse* than using no embedding at all on the cases that matter (0.2998 vs
0.3097), because letting the embedding dominate reintroduces the popularity
smoothing the graph avoids. Raising beta will make aggregate benchmarks look
better while making the product worse.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

FLOOR = -0.15
BETA = 0.5
N_STAPLES = 50
# The Adventurous lane's candidate exclusion is deliberately NOT N_STAPLES.
# N_STAPLES=50 is the *research* definition used to slice the non-staple metric;
# reusing it as a UI filter made the lane a near-no-op, because NPMI already
# divides popularity out, so top-50 items rarely reach a top-10 anyway (0.5 of
# 10 on average). Measured over 250 random baskets, excluding only the top 50
# left the two lanes returning identical top-10s for ~20% of baskets. At 300 the
# lanes genuinely diverge while every suggestion still rests on real evidence:
#
#   N excluded |  50 | 200 | 300 | 500 | 700
#   median count |  68 |  44 |  30 |  13 |   7
#   % with evidence | 100 | 100 | 100 | 99.7 | 96.4
#   median NPMI | .362| .368| .372| .369| .364
#
# NPMI is flat across the sweep: adventurousness costs evidence *volume*, not
# association *strength*. 300 is the last point with 100% coverage and a
# double-digit median count.
ADVENTUROUS_EXCLUDE = 300


class Ranker:
    def __init__(self, bundle: Path, emb: Path | None = None):
        b = Path(bundle)
        self.meta = json.loads((b / "graph.meta.json").read_text())
        v = json.loads((b / "vocab.json").read_text())
        self.names: list[str] = v["ingredients"]
        self.freq = np.array(v["freq"], np.int64)
        n = self.meta["n_ingredients"]
        self.indptr = np.frombuffer((b / "graph.indptr.u32").read_bytes(),
                                    "<u4")
        self.indices = np.frombuffer((b / "graph.indices.u16").read_bytes(),
                                     "<u2")
        self.npmi = np.frombuffer((b / "graph.npmi.f16").read_bytes(),
                                  "<f2").astype(np.float32)
        self.count = np.frombuffer((b / "graph.count.u16").read_bytes(), "<u2")
        assert len(self.indptr) == n + 1
        self.n = n
        # Staples are defined by corpus frequency, not by a hand-written list,
        # so the boundary moves with the data instead of rotting.
        # kind="stable" is load-bearing, not tidiness: 755 ingredients share a
        # frequency with another, and there is a genuine tie exactly at the
        # ADVENTUROUS_EXCLUDE boundary (9972 == 9972). numpy's default
        # quicksort is unstable, so without this the Python and Swift lanes
        # could select different ingredients and golden parity would fail for
        # a reason no one would find quickly. Swift matches by breaking ties on
        # ascending index, which is what a stable ascending sort produces.
        order = np.argsort(self.freq, kind="stable")
        self.staples = set(order[-N_STAPLES:].tolist())
        self.adventurous_excluded = set(order[-ADVENTUROUS_EXCLUDE:].tolist())
        self.W = None
        # Load the raw float32 blob the app ships, not a .npy -- the reference
        # must read exactly the bytes Swift reads. Dimension is inferred from
        # file size so the same code works for d=64/128/300.
        blob = b / "cooc.emb.f32" if emb is None else Path(emb)
        if blob.exists():
            raw = np.frombuffer(blob.read_bytes(), "<f4")
            assert raw.size % n == 0, f"{blob} not divisible by {n} rows"
            self.W = raw.reshape(n, raw.size // n).astype(np.float32)
            self.d = self.W.shape[1]
            norms = np.linalg.norm(self.W, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-3), \
                "embedding rows must ship L2-normalised"

    def graph_scores(self, basket: list[int]) -> np.ndarray:
        """Walk each basket item's CSR row, accumulating NPMI above the floor."""
        s = np.zeros(self.n, np.float32)
        for b in basket:
            lo, hi = int(self.indptr[b]), int(self.indptr[b + 1])
            idx, w = self.indices[lo:hi], self.npmi[lo:hi]
            m = w >= FLOOR
            np.add.at(s, idx[m].astype(np.int64), w[m])
        return s

    def pair(self, i: int, j: int) -> dict | None:
        """Single-pair evidence for the pairing screen. None means never seen."""
        lo, hi = int(self.indptr[i]), int(self.indptr[i + 1])
        row = self.indices[lo:hi]
        k = int(np.searchsorted(row, j))
        if k >= len(row) or int(row[k]) != j:
            return None
        return {"npmi": float(self.npmi[lo + k]),
                "count": int(self.count[lo + k]),
                "above_floor": bool(self.npmi[lo + k] >= FLOOR)}

    def rank(self, basket: list[int], k: int = 10, adventurous: bool = False,
             dedup: bool = True, beta: float = BETA) -> list[dict]:
        s = self.graph_scores(basket)
        if self.W is not None and beta:
            q = self.W[basket].mean(0)
            q /= np.linalg.norm(q) + 1e-9
            s = s + beta * (self.W @ q)
        s[list(basket)] = -np.inf
        # Novelty comes from removing staples as *candidates*, never from
        # penalising their score: the lambda sweep in A5 showed a score penalty
        # costs 2.8 recall points for 0.2 nats of novelty and collapses
        # entirely by lambda=0.2.
        if adventurous:
            s[list(self.adventurous_excluded)] = -np.inf

        out: list[dict] = []
        for c in np.argsort(-s):
            if not np.isfinite(s[c]):
                break
            if dedup and self._dup(int(c), basket, [o["id"] for o in out]):
                continue
            out.append({"id": int(c), "name": self.names[c],
                        "score": round(float(s[c]), 6)})
            if len(out) == k:
                break
        return out

    def _dup(self, c: int, basket: list[int], kept: list[int]) -> bool:
        """Suppress near-synonyms: rice -> brown_rice is a substitute, not a pairing."""
        t = self._tok(c)
        return any(t & self._tok(o) for o in list(basket) + kept)

    def _tok(self, i: int) -> set[str]:
        return {w for w in re.split(r"[_\s]+", self.names[i].lower())
                if len(w) > 2}


def golden(args) -> None:
    r = Ranker(Path(args.bundle), Path(args.emb) if args.emb else None)
    look = {n: i for i, n in enumerate(r.names)}
    baskets = [b for b in (["chicken", "lemon", "garlic"], ["tomato", "basil"],
                           ["chocolate", "chili_powder"], ["rice", "soy_sauce"],
                           ["butter", "flour", "sugar", "egg"])
               if all(x in look for x in b)]
    fx = {"floor": FLOOR, "beta": BETA, "n_staples": N_STAPLES,
          "adventurous_exclude": ADVENTUROUS_EXCLUDE,
          "meta": r.meta, "cases": []}
    for b in baskets:
        ids = [look[x] for x in b]
        fx["cases"].append({
            "basket": b, "ids": ids,
            "classic": r.rank(ids, 10),
            "adventurous": r.rank(ids, 10, adventurous=True),
            "no_dedup": r.rank(ids, 10, dedup=False),
            "graph_only": r.rank(ids, 10, beta=0.0)})
    pairs = [("chicken", "lemon"), ("strawberry", "basil"), ("rice", "cinnamon")]
    fx["pairs"] = [{"a": a, "b": bb, "evidence": r.pair(look[a], look[bb])}
                   for a, bb in pairs if a in look and bb in look]
    Path(args.out).write_text(json.dumps(fx, indent=2))
    print(f"wrote {args.out} with {len(fx['cases'])} baskets")
    for c in fx["cases"]:
        print(f"\n{' + '.join(c['basket'])}")
        print("  classic    :", ", ".join(x["name"] for x in c["classic"][:6]))
        print("  adventurous:",
              ", ".join(x["name"] for x in c["adventurous"][:6]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="../data/derived/app_bundle")
    ap.add_argument("--emb", default="")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("golden")
    g.add_argument("--out", default="../data/derived/golden_ranker.json")
    g.set_defaults(fn=golden)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
