#!/usr/bin/env python3
"""Map raw multilingual ingredient strings onto the 1,790-term Epicure vocab.

This is the step everything else depends on: the co-occurrence graph, the NPMI
edges and ultimately the embeddings are all built from whatever survives here.
The paper reports 99.22% of recipes matched (4,103,118 / 4,135,189), which is
the number to beat.

Three matching regimes, because the corpus spans three writing conventions:

  * English   -- surface forms are close to the vocab already, so try an exact
                 alias lookup with modifier stripping first and only fall back
                 to a regex scan. Cheapest path, and it covers ~2.9M recipes.
  * Spaced    -- every other alphabetic language. A longest-first alternation
                 with \\b boundaries over the language's lexicon.
  * Unspaced  -- zh/ja/th have no word boundaries, so \\b never fires. Plain
                 substring alternation, still longest-first so 低筋面粉 beats
                 面粉 and しいたけ beats たけ.

Longest-first ordering is load-bearing in all three: Python alternation is
first-match-wins, so without it "buttermilk" silently scores as "butter".
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_recipe_cooc as brc
import lexicons
import multilingual

ROOT = Path(__file__).resolve().parents[1]


class Normalizer:
    """Compiled once, reused for every recipe in the corpus."""

    def __init__(self) -> None:
        self.vocab, self.itos = brc.load_vocab()
        self.alias = brc.build_alias(self.vocab)
        self.en_pat = brc.build_matcher(self.alias)

        bad = lexicons.validate(self.vocab)
        if bad:
            raise SystemExit(f"lexicon targets missing from vocab: {bad}")

        # build_maps returns ({lang: {surface: vocab_name}}, unmatched), and the
        # values are names -- convert to ids here so matching stays a dict hit.
        mm, _unmatched = multilingual.build_maps(self.vocab)
        self.maps: dict[str, dict[str, int]] = {}
        for lang, m in mm.items():
            self.maps[lang] = {k: self.vocab[v] for k, v in m.items()
                               if v in self.vocab}
        for lang, m in lexicons.LEXICONS.items():
            self.maps[lang] = {k: self.vocab[v] for k, v in m.items()}

        self.pats = {}
        for lang, m in self.maps.items():
            keys = sorted(m, key=len, reverse=True)
            body = "|".join(re.escape(k) for k in keys)
            # \b is meaningless in scripts without spaces; it would match nothing.
            self.pats[lang] = re.compile(body if lang in lexicons.NO_BOUNDARY
                                         else rf"\b(?:{body})\b")

    # ---------------------------------------------------------------- English
    def _english(self, items) -> set[int]:
        out = set()
        for raw in items:
            t = brc._QTY.sub("", str(raw).strip().lower())
            t = re.sub(r"[^a-z0-9 \-']+", " ", t)
            t = re.sub(r"\s+", " ", t).strip()
            if not t:
                continue
            hit = brc.lookup(self.alias, t)
            if hit is not None:
                out.add(hit)
                continue
            for m in self.en_pat.finditer(t):        # multi-ingredient lines
                out.add(self.alias[m.group(1)])
        return out

    # ------------------------------------------------------------ Non-English
    def _foreign(self, lang: str, items) -> set[int]:
        pat, m = self.pats[lang], self.maps[lang]
        if lang == "zh":
            text = " ".join(brc._ZH_QTY.sub("", str(i)) for i in items)
        else:
            text = " ".join(str(i) for i in items).lower()
        return {m[h] for h in pat.findall(text)}

    def normalize(self, lang: str, items) -> set[int]:
        """Vocab ids for one recipe. English is also tried as a fallback:
        many 'foreign' corpora carry Latin brand names and loanwords, and the
        Indian/Moroccan/Filipino sets are English-tagged but locally spelled."""
        if not items:
            return set()
        if lang == "en" or lang not in self.maps:
            return self._english(items)
        got = self._foreign(lang, items)
        if lang in ("fil", "ro", "id", "es", "de", "tr", "vi"):
            got |= self._english(items)      # Latin-script, safe to double-scan
        return got


def main(limit: int | None) -> None:
    import corpus

    nz = Normalizer()
    print(f"vocab {len(nz.vocab)} | alias {len(nz.alias)} | "
          f"lexicon langs {sorted(nz.maps)}\n")

    per_src = {}
    per_lang = {}
    freq = Counter()
    tot = matched = 0
    for key, lang, items in corpus.iter_all(limit):
        ids = nz.normalize(lang, items)
        s = per_src.setdefault(key, [0, 0, 0, lang])
        l = per_lang.setdefault(lang, [0, 0, 0])
        s[0] += 1
        l[0] += 1
        tot += 1
        if ids:
            s[1] += 1
            l[1] += 1
            matched += 1
            s[2] += len(ids)
            l[2] += len(ids)
            freq.update(ids)
        if tot % 500_000 == 0:
            print(f"  {tot:,} recipes, {matched/tot:.2%} matched", flush=True)

    print(f"\n{'source':26}{'lang':5}{'recipes':>12}{'matched':>10}{'ing/rec':>9}")
    for k, (n, mt, ing, lg) in sorted(per_src.items(), key=lambda x: -x[1][0]):
        print(f"{k:26}{lg:5}{n:>12,}{mt/max(n,1):>9.1%}{ing/max(mt,1):>9.1f}")

    print(f"\n{'lang':6}{'recipes':>12}{'matched':>10}{'ing/rec':>9}")
    for lg, (n, mt, ing) in sorted(per_lang.items(), key=lambda x: -x[1][0]):
        print(f"{lg:6}{n:>12,}{mt/max(n,1):>9.1%}{ing/max(mt,1):>9.1f}")

    cov = len(freq) / len(nz.vocab)
    print(f"\nTOTAL {tot:,} recipes | matched {matched:,} ({matched/tot:.2%})"
          f" | paper 99.22%")
    print(f"vocab terms seen {len(freq):,}/{len(nz.vocab)} ({cov:.1%})")

    out = ROOT / "data" / "derived" / "normalize_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "total": tot, "matched": matched, "rate": matched / max(tot, 1),
        "vocab_seen": len(freq),
        "per_source": {k: {"lang": v[3], "recipes": v[0], "matched": v[1]}
                       for k, v in per_src.items()},
        "per_lang": {k: {"recipes": v[0], "matched": v[1]}
                     for k, v in per_lang.items()},
        "top_terms": [[nz.itos[i], c] for i, c in freq.most_common(60)],
        "unseen": [nz.itos[i] for i in range(len(nz.itos)) if i not in freq][:400],
    }, indent=2))
    print(f"wrote {out.relative_to(ROOT)}")


def dump(limit: int | None = None, path: Path | None = None) -> Path:
    """Normalise the whole corpus once and persist it in CSR form.

    Re-parsing 5.3M recipes across 29 heterogeneous readers takes minutes, and
    the graph builder and the training sampler both need the same result, so it
    is written once as a flat uint16 id array plus int64 offsets. Recipes that
    matched nothing are dropped: they cannot contribute an edge and would only
    distort the co-occurrence denominators if kept.
    """
    import numpy as np

    import corpus

    nz = Normalizer()
    flat: list[int] = []
    off = [0]
    langs: list[str] = []
    srcs: list[str] = []
    seen = 0
    dup = 0
    # The 29 readers overlap: RecipeNLG already contains much of Food.com, and
    # several expansion sets re-publish baseline rows. Left in, those recipes
    # would double-count into the co-occurrence denominators and quietly
    # over-weight whichever cuisine happens to be mirrored most. Identity is the
    # exact raw ingredient text, which is far safer than the normalised id set:
    # two genuinely different recipes often share a normalised set (flour, egg,
    # sugar, butter), but they rarely share verbatim ingredient lines.
    fp: set[int] = set()
    for key, lang, items in corpus.iter_all(limit):
        seen += 1
        h = hash("\u241f".join(sorted(str(i).strip().lower() for i in items)))
        if h in fp:
            dup += 1
            continue
        fp.add(h)
        ids = nz.normalize(lang, items)
        if not ids:
            continue
        flat.extend(sorted(ids))
        off.append(len(flat))
        langs.append(lang)
        srcs.append(key)
        if seen % 500_000 == 0:
            print(f"  {seen:,} scanned, {len(off)-1:,} kept, {dup:,} dup",
                  flush=True)

    path = path or ROOT / "data" / "derived" / "recipe_ids.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        flat=np.asarray(flat, dtype=np.uint16),
        offsets=np.asarray(off, dtype=np.int64),
        lang=np.asarray(langs), source=np.asarray(srcs),
        itos=np.asarray(nz.itos),
    )
    print(f"scanned {seen:,} | duplicates dropped {dup:,} | "
          f"kept {len(off)-1:,} ({(len(off)-1)/max(seen,1):.2%}) | "
          f"{len(flat):,} ingredient slots -> {path.relative_to(ROOT)}")
    return path


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "-"
    if a == "dump":
        b = sys.argv[2] if len(sys.argv) > 2 else "-"
        dump(None if b == "-" else int(b))
    else:
        main(None if a == "-" else int(a))
