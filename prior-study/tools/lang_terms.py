#!/usr/bin/env python3
"""Rank the most frequent ingredient surface forms per language.

Hitting the paper's 99.22% match rate means knowing what actually needs
mapping, not guessing. Ingredient vocabularies are heavily Zipfian: for most
languages a few hundred head terms cover the large majority of token
occurrences, so this ranks candidates by how much coverage each would buy.

Writes data/derived/lang_terms/<lang>.json = [[term, count], ...].
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "derived" / "lang_terms"
OUT.mkdir(parents=True, exist_ok=True)

# quantity/unit noise, per language family
UNITS = {
    "es": r"tazas?|cucharadas?|cucharaditas?|gramos?|kilos?|litros?|ml|gr?|kg|"
          r"dientes?|ramas?|hojas?|pizca|al gusto|de|del|la|el|los|las|un|una|"
          r"unos|unas|y|o|para|con|sin|picad[oa]s?|ralla[dz][oa]s?|fresc[oa]s?",
    "tr": r"adet|su bardağı|çay bardağı|yemek kaşığı|tatlı kaşığı|çay kaşığı|"
          r"gram|kg|ml|litre|paket|diş|demet|tutam|bardak|kaşığı|kaşık|"
          r"yarım|birkaç|az|biraz|ince|kıyılmış|doğranmış|rendelenmiş",
    "id": r"buah|butir|siung|ekor|lembar|batang|sdm|sdt|gram|kg|ml|liter|"
          r"secukupnya|potong|iris|halus|besar|kecil|sedang|bungkus|ruas",
    "de": r"g|kg|ml|l|EL|TL|Prise|Bund|Zehen?|Stück|Packung|Dose|Becher|"
          r"gross|klein|frisch|gehackt|gerieben|m\.-grosse|etwas|n\)|\(n\)",
    "vi": r"trái|quả|củ|cây|nhánh|muỗng canh|muỗng cà phê|muỗng|gram|kg|ml|"
          r"lít|gói|lá|tép|ít|vừa đủ|cái|con|bó",
    "fr": r"cuillères?|tasses?|grammes?|kg|ml|litres?",
    "el": r"κιλά|γραμμάρια|κουταλιές|κουταλιά|φλιτζάνι|ml|γρ|κ\.σ|κ\.γ|"
          r"σκελίδες|ματσάκι|λίγο|φρέσκο|ψιλοκομμένο",
    "fa": r"پیمانه|قاشق|غذاخوری|چایخوری|مرباخوری|گرم|کیلو|عدد|حبه|به میزان لازم",
    "he": r"כפות|כפית|כוס|כוסות|גרם|ק\"ג|מ\"ל|חבילה|שיני|קורט",
    "th": r"ถ้วย|ช้อนโต๊ะ|ช้อนชา|กรัม|กิโลกรัม|ลูก|ใบ|หัว|ต้น",
    "ja": r"個|本|枚|大さじ|小さじ|カップ|グラム|少々|適量|g|ml|cc",
    "fil": r"cups?|tbsp|tsp|lbs?|grams?|kilo|pcs?|cloves?|pieces?",
    "ro": r"linguri|lingurita|cana|pahar|grame|kg|ml|bucata|catei|putin",
    "en": r"cups?|tbsp|tsp|tablespoons?|teaspoons?|ounces?|oz|pounds?|lbs?|"
          r"grams?|kg|ml|liters?|cloves?|pinch|dash|packages?|cans?|large|"
          r"small|medium|fresh|chopped|minced|ground|sliced|diced|to taste",
}
GENERIC = re.compile(r"[0-9０-９½¼¾⅓⅔⅛/().,\-–—:;*•·\[\]\"']+")


def clean(s: str, lang: str) -> str:
    t = GENERIC.sub(" ", str(s))
    u = UNITS.get(lang)
    if u:
        t = re.sub(rf"(?:^|\s)(?:{u})(?=\s|$)", " ", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip().lower()


def main(per_source: int | None) -> None:
    counts: dict[str, Counter] = defaultdict(Counter)
    n_by_lang: Counter = Counter()
    seen = 0
    for key, lang, items in corpus.iter_all(per_source):
        n_by_lang[lang] += 1
        seen += 1
        for it in items:
            t = clean(it, lang)
            if 1 < len(t) < 60:
                counts[lang][t] += 1
        if seen % 250_000 == 0:
            print(f"  {seen:,} recipes", flush=True)

    print(f"\n{'lang':6}{'recipes':>12}{'distinct terms':>16}{'top-400 cover':>15}")
    summary = {}
    for lang, c in sorted(counts.items(), key=lambda x: -n_by_lang[x[0]]):
        tot = sum(c.values())
        top = c.most_common(400)
        cover = sum(n for _, n in top) / max(tot, 1)
        print(f"{lang:6}{n_by_lang[lang]:>12,}{len(c):>16,}{cover:>14.1%}")
        summary[lang] = dict(recipes=n_by_lang[lang], distinct=len(c),
                             top400_coverage=round(cover, 4))
        (OUT / f"{lang}.json").write_text(
            json.dumps(c.most_common(1500), ensure_ascii=False, indent=0))
    (OUT / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}/<lang>.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "-" else None)
