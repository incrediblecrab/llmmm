#!/usr/bin/env python
"""Propose lexicon aliases for mined unmapped terms, using Azure Foundry.

    ./.venv/bin/python scripts/propose_aliases.py --lang zh --top 400

How a hallucination is prevented from entering the corpus
---------------------------------------------------------
The model is never shown the 1,790-concept vocabulary and never asked to pick
from it. It is asked only to translate a term into a plain English ingredient
name. That name is then bound to a concept **through llmmm's own English alias
table** — the same matcher that already normalises two million English recipes.

So the model's answer is a hypothesis and the English lexicon is the judge. If
it invents "purple yam cake flour", nothing binds and the term is dropped. It
cannot name a concept that does not exist, because it is not naming concepts at
all. Wrong-but-bindable answers remain possible, which is what `--review` and
the compound check below are for.

Chinese needs one extra check. Matching there is substring-based, so a short
alias can swallow longer words: mapping 鸡 -> chicken would turn 鸡蛋 (egg) into
chicken if 鸡蛋 were not already mapped. Any proposal that is a substring of a
frequent unmapped term is therefore held back unless that longer term is being
mapped in the same pass.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingredient_model.config import PATHS  # noqa: E402
from ingredient_model.data.normalizer import ALIAS_DIR, get_normalizer  # noqa: E402

ENDPOINT = os.environ.get(
    "FOUNDRY_ENDPOINT", "https://llmmm-foundry.cognitiveservices.azure.com")
MODEL = os.environ.get("FOUNDRY_MODEL", "gpt-oss-120b")
BATCH = 40
LANG_NAME = {"zh": "Chinese", "ru": "Russian", "en": "English",
             "ja": "Japanese", "th": "Thai"}

PROMPT = """You are given cooking ingredient names in {lang} taken from recipe \
ingredient lists. For each, give the plain English name of the ingredient.

Rules:
- Answer with the common English ingredient name, lowercase, no quantities.
- Use the most specific ordinary name: "低筋粉" -> "cake flour", not "flour".
- Brand names: give what the product IS. "老干妈" -> "chili sauce".
- If the entry is not an ingredient (a section heading, an instruction, a \
utensil, a quantity on its own), answer exactly: NONE
- Answer with a JSON object mapping each input string to its answer. No other \
text.

Inputs:
{items}"""


def call(key: str, items: list[str], lang: str, retries: int = 3) -> dict:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT.format(
            lang=LANG_NAME.get(lang, lang),
            items=json.dumps(items, ensure_ascii=False))}],
        "max_completion_tokens": 4000,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        ENDPOINT.rstrip("/") + "/openai/v1/chat/completions", data=body,
        headers={"api-key": key, "Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                payload = json.load(r)
            text = payload["choices"][0]["message"]["content"] or ""
            m = re.search(r"\{.*\}", text, re.S)
            return json.loads(m.group(0)) if m else {}
        except (urllib.error.URLError, json.JSONDecodeError, KeyError,
                TimeoutError):
            if attempt == retries - 1:
                return {}
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="zh")
    ap.add_argument("--top", type=int, default=400)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    key = os.environ.get("FOUNDRY_KEY") or Path("/tmp/.fkey").read_text().strip()
    src = PATHS.recipes / "unmapped_terms.json"
    if not src.exists():
        print("run scripts/mine_unmapped.py first", file=sys.stderr)
        return 1
    mined = json.loads(src.read_text(encoding="utf-8"))
    if a.lang not in mined:
        print(f"no mined terms for {a.lang}", file=sys.stderr)
        return 1

    ranked = sorted(mined[a.lang]["terms"].items(), key=lambda kv: -kv[1])
    terms = [t for t, _ in ranked[:a.top]]
    df = dict(ranked)
    frequent = [t for t, c in ranked if c >= 3]
    print(f"{len(terms)} terms for {a.lang}, df {sum(df[t] for t in terms):,}")

    nz = get_normalizer(extra=False)
    asked: dict[str, str] = {}          # term -> raw English answer or NONE
    bound: dict[str, str] = {}          # term -> concept
    not_food: set[str] = set()

    def resolve(batch: list[str]) -> None:
        """Ask the model, then let the English lexicon judge the answers."""
        batch = [t for t in batch if t not in asked]
        if not batch:
            return
        groups = [batch[i:i + BATCH] for i in range(0, len(batch), BATCH)]
        got: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            for part in pool.map(lambda b: call(key, b, a.lang), groups):
                got.update({str(k): str(v) for k, v in part.items()})
        for term in batch:
            answer = got.get(term, "")
            asked[term] = answer
            if not answer or answer.strip().upper() == "NONE":
                not_food.add(term)
                continue
            ids = nz.normalize("en", [answer])
            if len(ids) == 1:
                bound[term] = nz.itos[next(iter(ids))]

    def hazards(term: str) -> list[str]:
        """Frequent terms that contain `term` and would be captured by it.

        A compound already bound is harmless: alternatives are sorted
        longest-first, so the compound matches before the short term ever
        gets a chance. A compound the model called NONE is *not* harmless —
        面团 (dough) contains 面 (noodle) and must keep blocking it.
        """
        out = []
        for other in frequent:
            if other == term or term not in other or other in bound:
                continue
            # 鸡肝 "chicken liver" does not block 鸡 -> chicken: the short term
            # yields a true, if incomplete, reading. 面团 "dough" does block
            # 面 -> noodle, because dough is not noodle.
            answer = asked.get(other, "")
            if answer and answer.strip().upper() != "NONE":
                ids = nz.normalize("en", [answer])
                if bound.get(term) in {nz.itos[i] for i in ids}:
                    continue
            out.append(other)
        return out

    # Pass 1, then keep resolving the compounds that block a proposal until
    # nothing new can be unblocked. Two extra passes reach a fixpoint here.
    resolve(terms)
    for _ in range(2):
        blocked = {o for t in list(bound) for o in hazards(t)}
        pending = [o for o in blocked if o not in asked]
        if not pending:
            break
        print(f"  resolving {len(pending)} blocking compounds")
        resolve(pending)

    held: dict[str, str] = {}
    for term in sorted(bound, key=len):
        if a.lang not in ("zh", "ja", "th"):
            break
        risky = hazards(term)
        if risky:
            held[term] = f"blocked by {risky[:3]}"
    for term in held:
        bound.pop(term, None)

    unbound = {t: asked[t] for t in asked
               if t not in bound and t not in not_food and t not in held}

    ALIAS_DIR.mkdir(parents=True, exist_ok=True)
    dest = ALIAS_DIR / f"{a.lang}.json"
    existing = json.loads(dest.read_text(encoding="utf-8")) if dest.exists() else {}
    if a.lang == "en":
        # The English matcher strips quantities, lowercases and drops
        # punctuation *before* looking a term up, so the stored key has to be
        # the cleaned form or it can never be hit. "2 bay leaves" -> "bay
        # leaves". Keys that clean to nothing, or that already resolve, are
        # dropped rather than written as dead entries.
        import build_recipe_cooc as brc  # type: ignore
        cleaned: dict[str, str] = {}
        for term, concept in bound.items():
            key = brc._QTY.sub("", term.strip().lower())
            key = re.sub(r"[^a-z0-9 \-']+", " ", key)
            key = re.sub(r"\s+", " ", key).strip()
            if key and not nz.normalize("en", [key]):
                # several raw lines collapse to one key; keep the best-attested
                cleaned.setdefault(key, concept)
                df[key] = max(df.get(key, 0), df.get(term, 0))
        bound = cleaned
    existing.update(bound)
    dest.write_text(json.dumps(existing, ensure_ascii=False, indent=1,
                               sort_keys=True), encoding="utf-8")

    gain = sum(df[t] for t in bound)
    print(f"\nbound      {len(bound):>5}   df {gain:,}")
    print(f"unbound    {len(unbound):>5}   (English name matched 0 or >1 concepts)")
    print(f"held back  {len(held):>5}   (substring hazard)")
    print(f"\n-> {dest} now has {len(existing)} aliases")
    for term, concept in list(bound.items())[:15]:
        print(f"   {term:<14}{concept}")
    if held:
        print("\nheld back:")
        for term, why in list(held.items())[:6]:
            print(f"   {term:<14}{why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
