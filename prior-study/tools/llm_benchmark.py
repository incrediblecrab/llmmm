#!/usr/bin/env python3
"""Head-to-head: our ranker vs an LLM on recipe completion.

The product claim is "better than Google/Claude for this". That is testable, so
it should be tested before an app is built on top of it -- and tested in a way
that could falsify it.

Same task as tools/backtest.py: a real recipe with one ingredient hidden. Our
ranker sees the remaining ingredients and ranks all 1,790 candidates. The LLM
sees the same ingredients and is asked for its 10 best guesses. Identical
recipes, identical hidden ingredients (same seed), scored identically.

The prompt is deliberately generous -- vocabulary format, worked example, and an
explicit instruction to rank. A strawman prompt would make our numbers look good
and teach us nothing. If the LLM wins we need to know that now, not after
shipping.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_ranker import Ranker  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
SEED = 20260806

SYSTEM = (
    "You are an expert chef with encyclopaedic knowledge of world cuisine. "
    "You will be shown most of the ingredients from a real recipe. Exactly one "
    "ingredient has been removed. Your job is to guess which one."
)

USER = """Here are the ingredients remaining in a real recipe:

{ingredients}

Exactly one ingredient was removed. Guess which one.

Rules:
- Reply with exactly 10 guesses, best first, one per line.
- Use lowercase with underscores instead of spaces, e.g. olive_oil, soy_sauce, black_pepper.
- Use common single ingredient names, not brands, quantities or descriptions.
- Do not repeat any ingredient already listed above.
- Output only the 10 names. No numbering, no commentary, no blank lines.

Example of a valid reply:
garlic
olive_oil
black_pepper
oregano
basil
parmesan_cheese
onion
red_pepper_flake
salt
parsley"""


def call_llm(endpoint: str, key: str, model: str, ingredients: list[str],
             retries: int = 4) -> list[str]:
    # gpt-oss is a reasoning model: chain-of-thought goes to reasoning_content
    # and the answer to content. A tight max_tokens is spent entirely on
    # reasoning and returns an EMPTY answer with finish_reason "length", which
    # silently scores as a total loss. Budget generously and verify.
    body = json.dumps({
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": USER.format(
                         ingredients="\n".join(ingredients))}],
        "model": model, "temperature": 0.0, "max_tokens": 4000,
    }).encode()
    url = endpoint.rstrip("/") + "/openai/v1/chat/completions"
    for a in range(retries):
        try:
            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=180) as r:
                ch = json.load(r)["choices"][0]
            txt = ch["message"].get("content") or ""
            if not txt.strip() and ch.get("finish_reason") == "length":
                raise ValueError("truncated before answering")
            lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            return lines[:10]
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
                KeyError, ValueError, json.JSONDecodeError):
            if a == retries - 1:
                return []
            time.sleep(2 ** a + random.random())
    return []


def build_matcher(names: list[str]):
    """Map free-text LLM output onto vocabulary ids, generously but not loosely."""
    exact = {n.lower(): i for i, n in enumerate(names)}
    loose: dict[str, int] = {}
    for i, n in enumerate(names):
        k = re.sub(r"[^a-z]", "", n.lower())
        loose.setdefault(k, i)
        loose.setdefault(k.rstrip("s"), i)

    def match(s: str) -> int | None:
        s = s.strip().lower().replace(" ", "_").strip("-*.0123456789 ")
        if s in exact:
            return exact[s]
        k = re.sub(r"[^a-z]", "", s)
        return loose.get(k) or loose.get(k.rstrip("s"))
    return match


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--bundle", default=str(DERIVED / "app_bundle"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--non-staple-only", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "llm_benchmark.json"))
    a = ap.parse_args()

    endpoint = os.environ["FOUNDRY_ENDPOINT"]
    key = os.environ["FOUNDRY_KEY"]

    r = Ranker(Path(a.bundle))
    names = r.names
    match = build_matcher(names)

    z = np.load(DERIVED / "recipe_ids.npz", allow_pickle=True)
    flat, offs = z["flat"].astype(np.int64), z["offsets"]
    test = np.load(DERIVED / "recipe_backtest.npz")["test"]

    # Reproduce backtest.py's hidden-ingredient choice exactly so the two
    # evaluations are scored on identical problems.
    rng = np.random.default_rng(SEED)
    held_all, ctx_all = [], []
    B = 4096
    for s in range(0, len(test), B):
        for rid in test[s:s + B]:
            ing = flat[offs[rid]:offs[rid + 1]]
            h = rng.integers(len(ing))
            held_all.append(int(ing[h]))
            ctx_all.append(np.delete(ing, h).tolist())

    pick = np.random.default_rng(1).permutation(len(test))
    if a.non_staple_only:
        # The aggregate task is dominated by staples, where any competent model
        # guesses salt. Stratifying onto non-staple answers tests the case the
        # product actually exists to serve, and needs its own sample size.
        staple_ids = r.staples
        pick = [i for i in pick if held_all[i] not in staple_ids]
    pick = pick[:a.n]
    cases = [(ctx_all[i], held_all[i]) for i in pick]
    print(f"{len(cases)} cases | model {a.model} | "
          f"non_staple_only={a.non_staple_only}")

    def one(case):
        ctx, held = case
        return call_llm(endpoint, key, a.model, [names[c] for c in ctx])

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        replies = list(ex.map(one, cases))
    print(f"llm done in {time.time() - t0:.0f}s")

    llm_rank, ours_rank, unmapped, empty = [], [], 0, 0
    for (ctx, held), rep in zip(cases, replies):
        if not rep:
            empty += 1
        ids: list[int] = []
        for line in rep:
            m = match(line)
            if m is None:
                unmapped += 1
            elif m not in ids and m not in ctx:
                ids.append(m)
        llm_rank.append(ids.index(held) + 1 if held in ids else 999)
        ours = r.rank(list(ctx), k=10)
        oid = [o["id"] for o in ours]
        ours_rank.append(oid.index(held) + 1 if held in oid else 999)

    lr, orr = np.array(llm_rank), np.array(ours_rank)
    staples = r.staples
    ns = np.array([h not in staples for _, h in cases])

    def stat(x):
        n = len(x)
        r10 = float((x <= 10).mean())
        return {"n": n, "recall@10": round(r10, 4),
                "mrr": round(float((1 / x).mean()), 4),
                "ci95": round(1.96 * (r10 * (1 - r10) / n) ** 0.5, 4)}

    out = {"model": a.model, "n": len(cases),
           "llm_empty_replies": empty, "llm_unmapped_names": unmapped,
           "all": {"llm": stat(lr), "ours": stat(orr)},
           "non_staple": {"n": int(ns.sum()), "llm": stat(lr[ns]),
                          "ours": stat(orr[ns])}}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
