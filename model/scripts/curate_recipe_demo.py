#!/usr/bin/env python3
"""Curate only the pinned public recipes below; never load a training corpus."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from html import unescape
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlencode


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "model/demo_data"
CACHE = ROOT / ".artifacts/demo-sources"
LICENSE = "CC-BY-SA-4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
VOCABULARY_URL = (
    "https://huggingface.co/incrediblecrab/llmmm-recipes/"
    "resolve/v0.4.0-recipe-search/config.json"
)
VOCABULARY_SHA256 = "1bf792345b8face5975534312b676fcea1b53e5fa9c0e844948c4db743aa72ce"
TRANSFORMATION = (
    "Extracted every Ingredients bullet and Procedure step, in source order. "
    "Removed wiki links, formatting and images; decoded entities and normalized "
    "whitespace. Displayed explicit temperature-template arguments without "
    "converting temperatures. Preserved source measurements and alternatives. "
    "Added reviewed vocabulary mappings and parsed only explicit summary metadata; "
    "did not combine component times, infer servings from yields, or add ingredients."
)
SOURCE_LIMITATIONS = (
    "Canonical ingredients cover the source Ingredients section, not every "
    "serving suggestion in the procedure or every constituent of a mixture. "
    "Measurements and instructions are source-reported, not kitchen-tested or corrected."
)


class CurationError(ValueError):
    pass


@dataclass(frozen=True)
class IngredientRule:
    fragment: str
    canonical: tuple[str, ...]
    unmapped_reason: str | None = None


def rule(fragment: str, *names: str, unmapped: str | None = None) -> IngredientRule:
    return IngredientRule(fragment, names, unmapped)


@dataclass(frozen=True)
class Recipe:
    id: str
    title: str
    revision: int
    sha256: str
    origin_revision: int
    rules: tuple[IngredientRule, ...]
    talk_revision: int | None = None
    additional_attribution: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def source_title(self) -> str:
        return "Cookbook:" + self.title


RECIPES = (
    Recipe(
        "oat-porridge", "Oat Porridge", 4606890,
        "c742d8ca191ecfd7618e0706350781261e36f8dcc2bd7216aea1e4dd5ce42b04",
        69760,
        (
            rule("rolled oats or other grain", "oat", unmapped="Other grain is unspecified."),
            rule("water", "water"),
            rule("milk", "milk"),
        ),
        additional_attribution=(
            "Originally moved from Porridge by English Wikipedia contributors; "
            "https://en.wikipedia.org/w/index.php?title=Porridge&action=history"
        ),
        notes=(
            "The earliest revision's edit summary credits the Wikipedia Porridge article; "
            "that additional attribution is retained.",
            "The procedure suggests toppings absent from the Ingredients section; "
            "they are not invented as additional ingredient lines or canonical entries.",
        ),
    ),
    Recipe(
        "potato-curry", "Potato Curry (Aloo Masala)", 4615205,
        "34b9903ea6d149754c1bd0e60bf63dd1bfdc317b316dfbdc186b9f8a5f6f66b9",
        41802,
        (
            rule("oil", "oil"),
            rule("mustard seed", "mustard_seed"),
            rule("chana dhal", "chickpea"),
            rule("curry leaves", "curry_leaf"),
            rule("onion", "onion"),
            rule("green chillies", "green_chili"),
            rule("potatoes", "potato"),
            rule("turmeric", "turmeric"),
            rule("salt", "salt"),
        ),
        notes=("The recipe starts with par-boiled potatoes; no extra preparation time is added.",),
    ),
    Recipe(
        "keralan-vegetable-stew", "Keralan Vegetable Stew", 4615200,
        "8a943d98427e7830edc331e4d4e4ded17e9087fbbe5b61e9c7b0850af45b034a",
        41816,
        (
            rule("coconut", "coconut"),
            rule(
                "boiled vegetables", "carrot", "green_bean", "potato", "pea",
                unmapped="The vegetable mixture ends with 'etc.'; only its named examples are mapped.",
            ),
            rule("oil", "oil"),
            rule("onion", "onion"),
            rule("ginger", "ginger"),
            rule("green chillies", "green_chili"),
            rule("curry leaves", "curry_leaf"),
            rule("cloves", "clove"),
            rule("cinnamon", "cinnamon"),
            rule("cardamom", "cardamom"),
            rule("vinegar", "vinegar"),
            rule(
                "flour or rice flour or cornstarch", "flour", "cornstarch",
                unmapped="The vocabulary has no generic rice_flour entry; glutinous rice flour is not assumed.",
            ),
            rule("salt", "salt"),
            rule("ground black pepper", "black_pepper"),
            rule("mustard seed", "mustard_seed"),
            rule("shallot", "shallot"),
        ),
        talk_revision=4054910,
        notes=("The recipe starts with boiled vegetables; no extra preparation time is added.",),
    ),
    Recipe(
        "garlic-shrimp", "Gambas al Ajillo (Garlic Shrimp)", 4533840,
        "1847e6e41e36124123fc200e6ad7f4735f59519017c1549b4cd9071af9c5d2d9",
        667826,
        (
            rule("shrimp", "shrimp"),
            rule("garlic", "garlic"),
            rule("paprika", "paprika"),
            rule("sherry", "sherry", "cognac"),
            rule("olive oil", "olive_oil"),
            rule("parsley", "parsley"),
            rule("lemon juice", "lemon"),
        ),
        notes=(
            "The procedure mentions salt and pepper absent from the Ingredients section; "
            "no new ingredient lines or canonical entries are supplied.",
        ),
    ),
    Recipe(
        "garlic-croutons", "Garlic Croutons I", 4656175,
        "001a804e7274337f2d563e8cccfabac558aed9127acec51033d7e36a2786573b",
        27996,
        (
            rule("French bread", "bread"),
            rule("garlic", "garlic"),
            rule("olive oil", "olive_oil"),
            rule("salt", "salt"),
        ),
    ),
    Recipe(
        "mozzarella-bruschetta", "Fresh Mozzarella Bruschetta", 4605220,
        "2b516bf72c853c86cc9a7300b6c0941191250121d22fe146d7233bb90b64df6c",
        248146,
        (
            rule("French bread", "bread"),
            rule("Roma tomatoes", "tomato"),
            rule("mozzarella", "mozzarella_cheese"),
            rule("basil leaves", "basil"),
            rule("salt", "salt"),
            rule("Black pepper", "black_pepper"),
            rule("Olive oil", "olive_oil"),
            rule("Oregano", "oregano"),
            rule("Balsamic vinegar", "balsamic_vinegar"),
        ),
        talk_revision=3818481,
    ),
    Recipe(
        "peanut-butter-cookies", "Peanut Butter Cookies", 4525009,
        "53a00b8ba677c1ec38abc872a49024a8e9b83ed0fc30ee673db0c6c4579104a8",
        196223,
        (
            rule("peanut butter", "peanut_butter"),
            rule("white granulated sugar", "sugar"),
            rule("brown sugar", "brown_sugar"),
            rule("butter or margarine", "butter", "margarine"),
            rule("egg", "egg"),
            rule("vanilla essence", "vanilla"),
            rule("all-purpose flour", "flour"),
            rule("salt", "salt"),
            rule("baking soda", "baking_soda"),
        ),
    ),
    Recipe(
        "chocolate-chip-cookies", "Chocolate Chip Cookies I", 4630845,
        "98080b85fe5b1d4fa78d7eecfeccbb9d6585140bde1d1884cc1d001455418bfb",
        3688069,
        (
            rule("butter", "butter"),
            rule("granulated white sugar", "sugar"),
            rule("brown sugar", "brown_sugar"),
            rule("vanilla extract", "vanilla"),
            rule("eggs", "egg"),
            rule("all-purpose flour", "flour"),
            rule("baking soda", "baking_soda"),
            rule("salt", "salt"),
            rule("lemon or orange peel", "lemon", "orange"),
            rule("semi-sweet chocolate morsels / chips", "chocolate"),
            rule("nuts, such as groundnuts", "nut", "peanut"),
        ),
        notes=(
            "The explicitly labeled Total: 32 minutes is used, not a sum of components.",
            "The source's Fahrenheit/Celsius pairs and volume/weight pairs are preserved "
            "even where they appear inconsistent; no conversion corrections are made.",
        ),
    ),
    Recipe(
        "potato-salad", "Potato Salad", 4525019,
        "01e13127b4c848b5e6189bd0c38d0e69050aa4285ff649546c4fc5c15774edfe",
        249537,
        (
            rule("potatoes", "potato"),
            rule("bacon", "bacon"),
            rule("yellow onion", "onion"),
            rule("white granulated sugar", "sugar"),
            rule("celery seed", "celery_seed"),
            rule("hard boiled eggs", "egg"),
            rule("flour", "flour"),
            rule("vinegar", "vinegar"),
            rule("water", "water"),
            rule("parsley", "parsley"),
            rule("salt", "salt"),
            rule("Pepper", unmapped="Generic pepper has no exact vocabulary entry; a pepper variety is not assumed."),
            rule("Paprika", "paprika"),
        ),
        talk_revision=2031516,
    ),
    Recipe(
        "red-lentil-soup", "Red Lentil Soup", 4518501,
        "e391f1ce5c693b2eb4fdeeed0314d4401e46af5a3e3d789b33254ff7910aa092",
        31685,
        (
            rule("olive oil", "olive_oil"),
            rule("onion", "onion"),
            rule("carrots", "carrot"),
            rule("garlic", "garlic"),
            rule("red lentils", "lentil"),
            rule(
                "water or stock", "water",
                unmapped="Generic stock has no exact vocabulary entry; a vegetable or meat stock is not assumed.",
            ),
            rule("bay leaf", "bay_leaf"),
        ),
        talk_revision=4469910,
        notes=("The simmering-time range in a step is not an overall recipe time.",),
    ),
    Recipe(
        "apple-crisp", "Apple Crisp I", 4587326,
        "5be4864521bd4de2ef6e47779486569b1871c6dea9aa828af577abf1965d8861",
        90074,
        (
            rule("baking apples", "apple"),
            rule("all-purpose flour", "flour"),
            rule("rolled oats", "oat"),
            rule("cinnamon", "cinnamon"),
            rule("cold butter", "butter", "margarine"),
            rule("brown sugar", "brown_sugar"),
        ),
        notes=("A baking time appears in a step, but no overall recipe time is published.",),
    ),
    Recipe(
        "waffles", "Waffles", 4519091,
        "1de32ba338d1e18cc60e4aba5da2034b4c9ac4b5530cd173d29938fd140eb5c9",
        695457,
        (
            rule("white granulated sugar", "sugar"),
            rule("baking powder", "baking_powder"),
            rule("milk", "milk"),
            rule("vegetable oil", "vegetable_oil"),
            rule("salt", "salt"),
            rule("eggs", "egg"),
            rule("flour", "flour"),
        ),
        notes=(
            "Only preparation and per-waffle cooking times are reported. "
            "They are not added or multiplied; total_minutes is null.",
        ),
    ),
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def revision_url(title: str, revision: int) -> str:
    return "https://en.wikibooks.org/w/index.php?" + urlencode(
        {"title": title, "oldid": revision}
    )


def history_url(title: str) -> str:
    return "https://en.wikibooks.org/w/index.php?" + urlencode(
        {"title": title, "action": "history"}
    )


def origin_url(recipe: Recipe) -> str:
    return "https://en.wikibooks.org/w/api.php?" + urlencode({
        "action": "query", "format": "json", "formatversion": 2,
        "prop": "revisions", "rvprop": "ids|timestamp|comment|content",
        "rvslots": "main", "titles": recipe.source_title, "rvdir": "newer", "rvlimit": 1,
    })


def talk_url() -> str:
    # Preserve the exact bounded audit query, including the two rejected candidates.
    titles = ["Pancakes (North American)", *[r.title for r in RECIPES],
              "Chokladboll (Swedish Chocolate Balls)"]
    return "https://en.wikibooks.org/w/api.php?" + urlencode({
        "action": "query", "format": "json", "formatversion": 2,
        "prop": "revisions", "rvprop": "ids|timestamp|comment|content",
        "rvslots": "main", "titles": "|".join("Cookbook talk:" + t for t in titles),
    })


class Fetcher:
    """Cache HTTP response bodies, not reconstructed or reserialized source text."""

    def __init__(self, offline: bool = False):
        self.offline = offline
        self.deadline = time.monotonic() + 180
        self.requests = 0
        self.last_request = 0.0

    def get(self, url: str, filename: str, expected: str | None = None) -> tuple[bytes, dict]:
        path = CACHE / filename
        metadata_path = CACHE / (filename + ".metadata.json")
        if not metadata_path.exists():
            # Early audit downloads used the suffix-replacing form.
            legacy = path.with_suffix(".metadata.json")
            if legacy.exists():
                metadata_path = legacy
        if path.exists() and metadata_path.exists():
            content = path.read_bytes()
            metadata = json.loads(metadata_path.read_text())
            if metadata["url"] != url or metadata["file"] != filename:
                raise CurationError(f"Cache URL/filename mismatch: {filename}")
            if metadata["sha256"] != digest(content) or metadata["status"] != 200:
                raise CurationError(f"Cache checksum/status mismatch: {filename}")
        else:
            if self.offline:
                raise CurationError(f"Missing cached source: {filename}; rerun without --offline.")
            remaining = self.deadline - time.monotonic()
            if self.requests >= 45 or remaining < 2:
                raise CurationError("Public-fetch budget exhausted (45 requests / 180 seconds).")
            import httpx

            time.sleep(max(0.0, 1.0 - (time.monotonic() - self.last_request)))
            self.requests += 1
            self.last_request = time.monotonic()
            try:
                with httpx.Client(
                    follow_redirects=True, max_redirects=5, timeout=min(20, remaining),
                    headers={"User-Agent": "llmmm-public-recipe-demo/1.0 (bounded source audit)"},
                ) as client:
                    response = client.get(url)
            except httpx.HTTPError as error:
                raise CurationError(f"Public source request failed without retries: {url}: {error}") from error
            if response.status_code != 200:
                raise CurationError(
                    f"HTTP {response.status_code} for {url}; stopped without automatic retries."
                )
            content = response.content
            metadata = {
                "url": url, "fetched_url": str(response.url), "status": 200,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "sha256": digest(content), "file": filename,
            }
            if expected and metadata["sha256"] != expected:
                raise CurationError(f"Pinned source checksum mismatch: {filename}")
            CACHE.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            metadata_path.write_bytes(json_bytes(metadata))
        if expected and digest(content) != expected:
            raise CurationError(f"Pinned source checksum mismatch: {filename}")
        return content, {
            "url": metadata["url"], "fetched_url": metadata["fetched_url"],
            "sha256": metadata["sha256"], "retrieved_at": metadata["retrieved_at"],
            "cache_path": str(path.relative_to(ROOT)),
        }


def normalize_space(text: str) -> str:
    return " ".join(text.split())


def plain_text(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"\[\[(?:File|Image):.*?\]\]", "", text, flags=re.I | re.S)
    text = re.sub(
        r"\{\{#invoke:temperature\|([fc])\|(-?\d+(?:\.\d+)?)\}\}",
        lambda m: m[2] + "°" + m[1].upper(), text, flags=re.I,
    )
    text = re.sub(
        r"\[\[([^\[\]]+)\]\]",
        lambda m: m[1].split("|")[-1] if "|" in m[1] else m[1].split(":")[-1],
        text,
    )
    text = re.sub(r"'{2,5}", "", text)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = unescape(text)
    if any(token in text for token in ("{{", "}}", "[[", "]]", "<ref")):
        raise CurationError(f"Unreviewed source markup: {text[:100]}")
    return normalize_space(text)


def summary_fields(text: str) -> dict[str, str]:
    start = re.search(r"\{\{recipe\s*summary\b", text, re.I)
    if not start:
        raise CurationError("Missing recipe summary")
    position, braces, links, part, parts = start.end(), 1, 0, "", []
    while position < len(text):
        token = text[position:position + 2]
        if token == "[[":
            links += 1
        elif token == "]]":
            links -= 1
        elif token == "{{":
            braces += 1
        elif token == "}}":
            braces -= 1
            if braces == 0:
                parts.append(part)
                break
        elif text[position] == "|" and braces == 1 and links == 0:
            parts.append(part)
            part = ""
            position += 1
            continue
        if token in ("[[", "]]", "{{", "}}"):
            part += token
            position += 2
        else:
            part += text[position]
            position += 1
    else:
        raise CurationError("Unclosed recipe summary")
    return {
        key.strip().casefold(): value.strip()
        for item in parts if "=" in item
        for key, value in [item.split("=", 1)]
    }


def source_list(text: str, heading: str, marker: str) -> list[str]:
    match = re.search(rf"^==\s*{re.escape(heading)}\s*==\s*$", text, re.I | re.M)
    if not match:
        raise CurationError(f"Missing {heading} section")
    section = re.split(r"^==[^=].*?==\s*$", text[match.end():], maxsplit=1, flags=re.M)[0]
    section = re.split(r"^\[\[Category:", section, maxsplit=1, flags=re.I | re.M)[0]
    result = []
    for line in section.splitlines():
        if not line.strip():
            continue
        if not line.startswith(marker) or line.startswith(marker * 2):
            raise CurationError(f"Unreviewed {heading} structure: {line[:100]}")
        rendered = plain_text(line[len(marker):])
        if not rendered:
            raise CurationError(f"Empty {heading} item")
        result.append(rendered)
    if not result:
        raise CurationError(f"Empty {heading} section")
    return result


@dataclass
class Element:
    tag: str
    attrs: dict[str, str | None] = field(default_factory=dict)
    children: list = field(default_factory=list)

    def text(self) -> str:
        if self.tag in {"script", "style", "figure", "img", "sup"}:
            return ""
        return "".join(c if isinstance(c, str) else c.text() for c in self.children)

    def walk(self):
        yield self
        for child in self.children:
            if isinstance(child, Element):
                yield from child.walk()


class PageParser(HTMLParser):
    def __init__(self, text: str):
        super().__init__(convert_charrefs=True)
        self.root = Element("document")
        self.stack = [self.root]
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        element = Element(tag, dict(attrs))
        self.stack[-1].children.append(element)
        if tag not in {"area", "base", "br", "col", "embed", "hr", "img", "input",
                       "link", "meta", "param", "source", "track", "wbr"}:
            self.stack.append(element)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def validate_rendered_page(page: bytes, recipe: Recipe, ingredients: list, steps: list) -> str:
    html = page.decode()
    revision = re.search(r'"wgRevisionId":(\d+)', html)
    if not revision or int(revision[1]) != recipe.revision:
        raise CurationError(f"Rendered revision mismatch: {recipe.id}")
    tree = PageParser(html).root
    links = [e.attrs.get("href", "") for e in tree.walk()
             if e.tag == "link" and e.attrs.get("rel") == "license"]
    if not any(link.startswith(LICENSE_URL) for link in links):
        raise CurationError(f"Missing per-page CC BY-SA 4.0 license: {recipe.id}")
    body = next((e for e in tree.walk()
                 if "mw-parser-output" in (e.attrs.get("class") or "").split()), None)
    if body is None:
        raise CurationError(f"Missing rendered content: {recipe.id}")
    extracted = {"ingredients": [], "procedure": []}
    section = None
    for element in body.walk():
        if element.tag == "h2":
            section = normalize_space(element.text()).casefold()
        elif element.tag == "li" and section in extracted:
            extracted[section].append(normalize_space(element.text()))
    if extracted["ingredients"] != ingredients or extracted["procedure"] != steps:
        raise CurationError(f"Raw/rendered recipe text differs: {recipe.id}")
    return next(link for link in links if link.startswith(LICENSE_URL))


def duration(text: str) -> int | float | None:
    match = re.fullmatch(r"(\d+(?:\.\d+)?|½)\s*(minutes?|mins?|hours?|hrs?)", text, re.I)
    if not match:
        return None
    value = 0.5 if match[1] == "½" else float(match[1])
    value *= 60 if match[2].lower().startswith("h") else 1
    if value <= 0:
        return None
    return int(value) if value.is_integer() else value


def time_metadata(fields: dict[str, str]) -> tuple[int | float | None, str | None, str | None]:
    value = fields.get("time", "").strip()
    if not value:
        return None, None, "No overall time is reported in the recipe summary."
    explicit = re.search(r"(?:^|<br\s*/?>|\n)\s*(Total:\s*([^<\n]+))", value, re.I)
    if explicit:
        minutes = duration(explicit[2].strip())
        return minutes, explicit[1], None if minutes is not None else "The total is ambiguous."
    minutes = duration(value)
    return minutes, value, (
        None if minutes is not None
        else "Component, per-item or ambiguous time only; no overall duration inferred."
    )


def servings_metadata(fields: dict[str, str]) -> tuple[int | float | None, str | None, str | None]:
    value = fields.get("servings", "").strip()
    if re.fullmatch(r"\d+(?:\.\d+)?(?:\s+(?:persons?|servings?))?", value, re.I):
        number = float(value.split()[0])
        if number > 0:
            return int(number) if number.is_integer() else number, value, None
    return None, value or None, (
        "Servings are a range or otherwise ambiguous; no endpoint or midpoint is chosen."
        if value else "No serving count is reported; an item yield is not a serving count."
    )


def mapping_audit(recipe: Recipe, ingredients: list[str], vocabulary: set[str]) -> list[dict]:
    if len(ingredients) != len(recipe.rules):
        raise CurationError(f"Ingredient count changed: {recipe.id}")
    result = []
    for raw, item in zip(ingredients, recipe.rules):
        if item.fragment.casefold() not in raw.casefold():
            raise CurationError(f"Ingredient mapping guard failed: {recipe.id}: {raw}")
        if not set(item.canonical) <= vocabulary:
            raise CurationError(f"Mapping is not in the public vocabulary: {item.canonical}")
        if not item.canonical and not item.unmapped_reason:
            raise CurationError(f"Unaccounted ingredient: {raw}")
        result.append({
            "raw_ingredient": raw, "canonical_ingredients": list(item.canonical),
            "unmapped_reason": item.unmapped_reason,
        })
    return result


def policies(fetcher: Fetcher) -> list[dict]:
    requests = (
        (
            revision_url("Wikibooks:Copyrights", 4622060),
            "copyright-policy-4622060.html",
            "Text reuse under CC BY-SA 4.0; page-link attribution, retained external credits, "
            "share-alike, change notice and license notice; non-text media and fair use are separate.",
            "Creative Commons Attribution-ShareAlike 4.0",
        ),
        (
            "https://foundation.wikimedia.org/w/index.php?title=Policy%3ATerms_of_Use&oldid=554823",
            "terms-of-use-554823.html",
            "Section 7 permits page-link attribution and requires preserving additional "
            "external attribution and licensing modifications under CC BY-SA 4.0 or later.",
            "7. Licensing of Content",
        ),
        (
            "https://creativecommons.org/licenses/by-sa/4.0/legalcode.en",
            "license-legalcode.source",
            "CC BY-SA 4.0 legal code: attribution, indicating modifications, share-alike, "
            "license link and no additional restrictions.",
            "Attribution-ShareAlike 4.0",
        ),
    )
    result = []
    for url, filename, finding, required in requests:
        content, evidence = fetcher.get(url, filename)
        if required not in normalize_space(PageParser(content.decode()).root.text()):
            raise CurationError(f"License evidence did not contain expected text: {url}")
        result.append(evidence | {"finding": finding})
    return result


def make_catalog(fetcher: Fetcher) -> tuple[list[dict], dict]:
    config_bytes, vocabulary_source = fetcher.get(
        VOCABULARY_URL, "vocabulary-config.source", VOCABULARY_SHA256
    )
    vocabulary = json.loads(config_bytes)["vocabulary"]
    if not isinstance(vocabulary, list) or not all(isinstance(x, str) for x in vocabulary):
        raise CurationError("Expected a public vocabulary list")
    license_evidence = policies(fetcher)
    talk_bytes, talk_source = fetcher.get(talk_url(), "selected-talk-pages.source")
    talk_pages = {p["title"]: p for p in json.loads(talk_bytes)["query"]["pages"]}
    records, sources = [], []
    for recipe in RECIPES:
        url = revision_url(recipe.source_title, recipe.revision)
        raw, fetched = fetcher.get(
            revision_url(recipe.source_title.replace(" ", "_"), recipe.revision) + "&action=raw",
            f"{recipe.id}-{recipe.revision}.wikitext", recipe.sha256,
        )
        rendered, license_page = fetcher.get(url, f"{recipe.id}-{recipe.revision}.html")
        text = raw.decode()
        ingredients = source_list(text, "Ingredients", "*")
        steps = source_list(text, "Procedure", "#")
        page_license = validate_rendered_page(rendered, recipe, ingredients, steps)
        if re.search(r"\{\{\s*(?:copyvio|copyright|permission|1881)\b|https?://", text, re.I):
            raise CurationError(f"Unreviewed provenance notice or external source: {recipe.id}")
        origin_bytes, origin_source = fetcher.get(origin_url(recipe), recipe.id + "-origin.source")
        origin = json.loads(origin_bytes)["query"]["pages"][0]["revisions"][0]
        if origin["revid"] != recipe.origin_revision:
            raise CurationError(f"Original-revision audit changed: {recipe.id}")
        talk = talk_pages["Cookbook talk:" + recipe.title]
        talk_revision = talk.get("revisions", [{}])[0].get("revid")
        if talk_revision != recipe.talk_revision:
            raise CurationError(f"Talk-page audit changed; review before curating: {recipe.id}")
        fields = summary_fields(text)
        minutes, time_evidence, time_note = time_metadata(fields)
        servings, servings_evidence, servings_note = servings_metadata(fields)
        audit = mapping_audit(recipe, ingredients, set(vocabulary))
        attribution = (
            f'English Wikibooks contributors, "{recipe.source_title}". '
            f"{LICENSE} ({LICENSE_URL}); contributors: {history_url(recipe.source_title)}."
        )
        if recipe.additional_attribution:
            attribution += " " + recipe.additional_attribution + "."
        record = {
            "id": recipe.id, "title": recipe.title, "language": "en",
            "canonical_ingredients": sorted({name for item in audit for name in item["canonical_ingredients"]}),
            "raw_ingredients": ingredients, "instructions": steps,
            "total_minutes": minutes, "servings": servings,
            "time_evidence": time_evidence, "servings_evidence": servings_evidence,
            "source_url": url, "source_title": recipe.source_title,
            "source_revision_id": recipe.revision, "retrieved_at": fetched["retrieved_at"],
            "license": LICENSE, "attribution": attribution,
            "attribution_url": history_url(recipe.source_title),
            "changes": TRANSFORMATION,
            "unmapped_ingredients": [item["raw_ingredient"] for item in audit if item["unmapped_reason"]],
        }
        records.append(record)
        sources.append({
            "id": recipe.id, "source_title": recipe.source_title, "source_url": url,
            "source_revision_id": recipe.revision, "fetched_source": fetched,
            "license": LICENSE, "license_url": LICENSE_URL,
            "license_evidence_urls": [url, *[item["url"] for item in license_evidence]],
            "per_page_license_evidence": license_page | {"rel_license": page_license},
            "attribution": attribution, "attribution_url": record["attribution_url"],
            "original_revision_audit": origin_source | {
                "source_revision_id": recipe.origin_revision,
                "revision_url": revision_url(recipe.source_title, recipe.origin_revision),
            },
            "talk_page_audit": {
                "fetched_source_url": talk_source["url"],
                "source_revision_id": talk_revision,
                "revision_url": revision_url("Cookbook talk:" + recipe.title, talk_revision)
                if talk_revision is not None else None,
                "finding": "No additional third-party license/reprint notice found."
                if talk_revision is not None else "Talk page did not exist at retrieval.",
            },
            "transformation": TRANSFORMATION,
            "source_summary_fields": {
                key: fields.get(key) or None for key in ("time", "servings", "yield")
            },
            "time_note": time_note, "servings_note": servings_note,
            "source_limitations": [SOURCE_LIMITATIONS, *recipe.notes],
            "ingredient_mapping": audit,
            "record_sha256": digest(json.dumps(record, ensure_ascii=False, sort_keys=True).encode()),
        })
    manifest = {
        "schema_version": 1,
        "catalog": "Small separately sourced public demonstration catalog; not training data or a benchmark.",
        "license": LICENSE, "license_url": LICENSE_URL,
        "transformation": TRANSFORMATION,
        "vocabulary": vocabulary_source | {
            "key": "vocabulary", "version": "v0.4.0-recipe-search",
            "size": len(vocabulary),
            "used_entries": sorted({name for record in records for name in record["canonical_ingredients"]}),
        },
        "license_evidence": license_evidence,
        "talk_page_audit_source": talk_source,
        "selection_review": (
            "Selected ingredient/procedure text, per-page license links, talk snapshots and "
            "earliest revisions were reviewed. No third-party copyright/reprint restriction "
            "was found for the selected text. Images, captions and other media are not reused. "
            "Wikipedia-origin attribution for Oat Porridge is retained. Pancakes (North American) "
            "was excluded after its talk-page provenance/measurement concerns; Chokladboll "
            "was excluded because its summary time conflicts with mandatory chilling."
        ),
        "sources": sources,
    }
    manifest["counts"] = counts(records)
    manifest["recipes_sha256"] = digest(serialize_records(records))
    validate_catalog(records, manifest)
    return records, manifest


def counts(records: list[dict]) -> dict[str, int]:
    return {
        "recipes": len(records),
        "known_total_minutes": sum(r["total_minutes"] is not None for r in records),
        "total_minutes_at_most_30": sum(
            r["total_minutes"] is not None and r["total_minutes"] <= 30 for r in records
        ),
        "known_servings": sum(r["servings"] is not None for r in records),
    }


RECORD_FIELDS = {
    "id", "title", "language", "canonical_ingredients", "raw_ingredients", "instructions",
    "total_minutes", "servings", "time_evidence", "servings_evidence", "source_url",
    "source_title", "source_revision_id", "retrieved_at", "license", "attribution",
    "attribution_url", "changes", "unmapped_ingredients",
}


def validate_catalog(records: list[dict], manifest: dict) -> None:
    if [r.get("id") for r in records] != [r.id for r in RECIPES]:
        raise CurationError("Catalog IDs/order differ from the pinned selection")
    if len(manifest["sources"]) != len(RECIPES):
        raise CurationError("Source manifest length differs")
    if manifest["vocabulary"]["sha256"] != VOCABULARY_SHA256:
        raise CurationError("Public vocabulary checksum differs")
    vocabulary = set(manifest["vocabulary"]["used_entries"])
    for recipe, record, source in zip(RECIPES, records, manifest["sources"]):
        if set(record) != RECORD_FIELDS:
            raise CurationError(f"Record schema differs: {recipe.id}")
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", record["id"]):
            raise CurationError("Non-ASCII or unstable ID")
        if record["language"] != "en" or record["license"] != LICENSE:
            raise CurationError("Language/license differs")
        if record["source_revision_id"] != recipe.revision or source["fetched_source"]["sha256"] != recipe.sha256:
            raise CurationError(f"Pinned source differs: {recipe.id}")
        if record["source_url"] != revision_url(recipe.source_title, recipe.revision):
            raise CurationError(f"Revision URL differs: {recipe.id}")
        if record["attribution_url"] != history_url(recipe.source_title) or LICENSE_URL not in record["attribution"]:
            raise CurationError(f"Missing attribution/license notice: {recipe.id}")
        if datetime.fromisoformat(record["retrieved_at"].replace("Z", "+00:00")).tzinfo is None:
            raise CurationError("Retrieval timestamp needs a timezone")
        for name in ("raw_ingredients", "canonical_ingredients", "instructions", "unmapped_ingredients"):
            if not isinstance(record[name], list) or not all(isinstance(x, str) and x.strip() for x in record[name]):
                raise CurationError(f"Invalid {name}: {recipe.id}")
        if not record["raw_ingredients"] or not record["instructions"] or not record["canonical_ingredients"]:
            raise CurationError("Empty recipe")
        for name in ("total_minutes", "servings"):
            number = record[name]
            if number is not None and (
                isinstance(number, bool) or not isinstance(number, (int, float))
                or not math.isfinite(number) or number <= 0
            ):
                raise CurationError(f"Invalid {name}: {recipe.id}")
        fields = {key: value or "" for key, value in source["source_summary_fields"].items()}
        if (record["total_minutes"], record["time_evidence"]) != time_metadata(fields)[:2]:
            raise CurationError("Time differs from source evidence")
        if (record["servings"], record["servings_evidence"]) != servings_metadata(fields)[:2]:
            raise CurationError("Servings differ from source evidence")
        audit = mapping_audit(recipe, record["raw_ingredients"], vocabulary)
        if audit != source["ingredient_mapping"]:
            raise CurationError("Ingredient mapping audit differs")
        expected_names = sorted({name for item in audit for name in item["canonical_ingredients"]})
        if record["canonical_ingredients"] != expected_names:
            raise CurationError("Canonical ingredients differ")
        if record["unmapped_ingredients"] != [item["raw_ingredient"] for item in audit if item["unmapped_reason"]]:
            raise CurationError("Unmapped ingredients differ")
        if source["record_sha256"] != digest(json.dumps(record, ensure_ascii=False, sort_keys=True).encode()):
            raise CurationError("Record checksum differs")
    actual = counts(records)
    if not (12 <= actual["recipes"] <= 20 and actual["known_total_minutes"] >= 8
            and actual["total_minutes_at_most_30"] >= 4):
        raise CurationError("Sample selection targets are not met")
    if manifest["counts"] != actual or manifest["recipes_sha256"] != digest(serialize_records(records)):
        raise CurationError("Catalog counts/checksum differ")


def serialize_records(records: list[dict]) -> bytes:
    return "".join(
        json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in records
    ).encode()


def dataset_card(records: list[dict]) -> bytes:
    statistics = counts(records)
    return f"""---
pretty_name: llmmm public recipe demonstration catalog
language:
  - en
license: cc-by-sa-4.0
task_categories:
  - text-retrieval
size_categories:
  - n<1K
tags:
  - recipes
  - wikibooks
  - demonstration
configs:
  - config_name: default
    data_files:
      - split: demo
        path: recipes.jsonl
---

# Public recipe demonstration catalog

[Try the browser demo](https://huggingface.co/spaces/incrediblecrab/llmmm-recipes-demo)
or read the [model and its evidence](https://huggingface.co/incrediblecrab/llmmm-recipes).

This is a small public demonstration catalog for the llmmm browser demo.
It contains {statistics["recipes"]} separately sourced English Wikibooks Cookbook recipes.
It is **not the original 4,653,430-row (4.65m) training dataset** and **not a
held-out quality benchmark**. No private corpus was used to create it.
The 83.5% full-catalog evaluation result must not be attributed to this sample.
Demo-catalog statistics describe only these records.

## Source and license

The adapted recipe text and this catalog are distributed under
[CC BY-SA 4.0]({LICENSE_URL}).
The [Wikibooks copyright policy](https://en.wikibooks.org/w/index.php?title=Wikibooks%3ACopyrights&oldid=4622060),
[Wikimedia Terms of Use, section 7](https://foundation.wikimedia.org/w/index.php?title=Policy%3ATerms_of_Use&oldid=554823#7._Licensing_of_Content),
and [license legal code](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en)
describe attribution, share-alike, change notices and license notices.
Each selected revision's HTML explicitly links to CC BY-SA 4.0. Recipe text,
talk-page snapshots and earliest revisions were reviewed for external-source
notices. Images, captions and other media are not included.

Credit belongs to the English Wikibooks contributors to each linked page.
Every record contains its revision URL, contributor-history URL, attribution,
license and change notice. Oat Porridge's earliest edit credits the
[English Wikipedia Porridge contributors](https://en.wikipedia.org/w/index.php?title=Porridge&action=history);
that additional credit is retained. `sources.json` records the evidence URLs,
actual fetched-byte SHA256s, retrieval timestamps and line-by-line mappings.

When redistributing, retain the attribution, source, license and change notices,
including the additional Wikipedia credit. Keep a link to the license, indicate
further changes, and license adaptations under CC BY-SA 4.0 (or a permitted
later/compatible license). Do not impose additional restrictions. The recipe
content is not relicensed under the application code's license.

## Extraction and limitations

Every source Ingredients bullet and Procedure step is retained in order as
plain text. Wiki links, formatting and images are removed; entities and
whitespace are normalized. Explicit temperature-template arguments are
displayed, not converted. Introductions, optional notes/variations, nutrition
tables, media and category boilerplate are outside the extraction.
Source measurements, ranges, alternatives and unmeasured seasonings remain
unchanged, including apparent source inconsistencies. No quantities, units,
instructions or missing ingredient lines are invented.

`canonical_ingredients` uses reviewed names from the public
[v0.4.0-recipe-search vocabulary]({VOCABULARY_URL}).
Some mappings are deliberately broad: lemon juice becomes `lemon`, red lentils
become `lentil`, and chocolate chips become `chocolate`. Explicit alternatives,
optional items and named mixture examples can all occur in the canonical set;
this does not mean that every alternative is required. Unspecified mixture
contents and unavailable names are flagged in `unmapped_ingredients`, which
contains the original line even when that line is partly mapped. Ingredient
sets do not cover every later serving suggestion or all constituents of a
compound ingredient. Raw ingredient lines remain authoritative; there are no
guessed quantity arrays.

**Ingredient exclusions are not allergy-safety certification.** Incomplete
ingredient coverage, substitutions, product composition and cross-contact are
not resolved here. Read the full source, product labels and relevant safety
guidance. These recipes have not been kitchen-tested or independently validated.

## Missing metadata

{statistics["known_total_minutes"]} recipes have an explicit source-reported overall summary time;
{statistics["total_minutes_at_most_30"]} of those report at most 30 minutes.
Hours may be converted to minutes, but preparation, cooking, resting and
per-item times are never added or multiplied. An explicitly labeled total is
used when supplied. Missing or ambiguous overall times stay `null`; the
available exact summary text is kept in `time_evidence`. Thus Waffles has
component-time evidence but no total; Red Lentil Soup and Apple Crisp have
step-level cooking times but no overall time.

Only unambiguous source serving counts become numbers. Ranges remain `null`
(for example, Potato Curry's `4-6`); no endpoint or midpoint is chosen.
Cookie/waffle counts and other item yields are not treated as servings.
`servings_evidence` retains an available exact serving field; otherwise it is
`null`. Per-recipe explanations and original summary fields are in `sources.json`.

## Reproduce and verify

The [source repository](https://github.com/incrediblecrab/llmmm) contains the
[curation script](https://github.com/incrediblecrab/llmmm/blob/main/model/scripts/curate_recipe_demo.py)
and its declared recipe revisions. Run from that repository's root using the
model environment:

```sh
model/.venv/bin/python model/scripts/curate_recipe_demo.py
model/.venv/bin/python model/scripts/curate_recipe_demo.py --verify
model/.venv/bin/python model/scripts/curate_recipe_demo.py --verify --offline
model/.venv/bin/python -m pytest model/tests/test_recipe_demo_data.py -q
```

The first command fetches only the declared public sources, verifies pinned
recipe/config hashes and page licenses, cross-checks raw text against rendered
lists, and generates `recipes.jsonl`, this card and `sources.json`. Requests are
bounded and HTTP failures stop without automatic retries. Source responses are
cached only under the Git-ignored `.artifacts/demo-sources/`.

`--verify` checks the distributed files without network access or a source
cache. `--verify --offline` additionally regenerates in memory from the cached
responses and requires byte-identical outputs. `--offline` alone regenerates
files from that cache. Fresh retrievals retain pinned recipe content but have
new retrieval timestamps and potentially different rendered-HTML hashes;
retaining the original cache permits byte-for-byte reproduction.
""".encode()


def outputs(records: list[dict], manifest: dict) -> dict[str, bytes]:
    return {
        "recipes.jsonl": serialize_records(records),
        "sources.json": json_bytes(manifest),
        "README.md": dataset_card(records),
    }


def verify_distributed() -> tuple[list[dict], dict]:
    records = [json.loads(line) for line in (OUTPUT / "recipes.jsonl").read_text().splitlines()]
    manifest = json.loads((OUTPUT / "sources.json").read_text())
    validate_catalog(records, manifest)
    for filename, expected in outputs(records, manifest).items():
        if (OUTPUT / filename).read_bytes() != expected:
            raise CurationError(f"Generated file differs: {filename}")
    return records, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Use cached public responses only.")
    parser.add_argument("--verify", action="store_true", help="Check outputs without writing or fetching.")
    args = parser.parse_args(argv)
    try:
        if args.verify:
            records, _ = verify_distributed()
            if args.offline:
                regenerated, manifest = make_catalog(Fetcher(offline=True))
                for filename, content in outputs(regenerated, manifest).items():
                    if (OUTPUT / filename).read_bytes() != content:
                        raise CurationError(f"Cached-source reproduction differs: {filename}")
        else:
            records, manifest = make_catalog(Fetcher(offline=args.offline))
            OUTPUT.mkdir(parents=True, exist_ok=True)
            for filename, content in outputs(records, manifest).items():
                (OUTPUT / filename).write_bytes(content)
            verify_distributed()
    except (CurationError, OSError, KeyError, json.JSONDecodeError) as error:
        print(f"Curation stopped: {error}", file=sys.stderr)
        return 1
    print(json.dumps(counts(records), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
