"""Recipe-card links, derived from each recorded source URL by per-site rules.

LINK_RULE states how the rules were chosen; model/results/recipe_link_health.json holds the per-site counts. A
recorded host or path without a rule fails the build, so a new source cannot ship unmeasured links.
"""
from __future__ import annotations

import re
import unicodedata
from typing import NamedTuple
from urllib.parse import urlsplit, urlunsplit

# Stated once here for the index, dataset card and model card. "Opened" means the page showed the recipe's title.
LINK_RULE = (
    "Each recorded site was checked on September 23, 2026 with a sample of its links, usually 12, and "
    "www.povarenok.ru again on September 24, when its pages were failing. A site's cards open "
    "the recorded page over HTTPS, on the site's current host and with the title slug NYT Cooking now requires, "
    "if at least five in six sampled pages opened showing their recipe title; otherwise the Internet Archive's "
    "copy of the recorded URL, if at least five in six sampled archived copies did; otherwise no link. The rule "
    "is judged per site, so an individual page may still have moved or been removed."
)
LINK_STATUSES = ("none", "source", "archive", "offline")
NONE, SOURCE, ARCHIVE, OFFLINE = range(len(LINK_STATUSES))
ARCHIVE_PREFIX = "https://web.archive.org/web/"
_BARE_RECIPE_ID = re.compile(r"/recipes/\d+")


class SiteRule(NamedTuple):
    status: int
    host: str | None = None
    year: str | None = None
    title_slug: bool = False


def _source(host: str | None = None, *, title_slug: bool = False) -> SiteRule:
    return SiteRule(SOURCE, host=host, title_slug=title_slug)


def _archive(year: str) -> SiteRule:
    return SiteRule(ARCHIVE, year=year)


_OFFLINE = SiteRule(OFFLINE)

# Recorded host -> ((path prefix, rule), ...); the first matching prefix wins and "" matches every path.
SITE_RULES: dict[str, tuple[tuple[str, SiteRule], ...]] = {
    # https://www.cookbooks.com resets TLS connections; its http:// pages redirect to https://cookbooks.com.
    "www.cookbooks.com": (("", _source("cookbooks.com")),),
    "www.food.com": (("", _source()),),
    # Its pages answered 403 or an outage page throughout the September 24 recheck; recheck before going back to source.
    "www.povarenok.ru": (("", _archive("2025")),),
    "www.epicurious.com": (("/recipes/member/views/", _OFFLINE), ("/recipes/food/views/", _source())),
    "www.nefisyemektarifleri.com": (("", _source()),),
    "www.allrecipes.com": (("", _source()),),
    "tastykitchen.com": (("", _OFFLINE),),
    "www.myrecipes.com": (("", _OFFLINE),),
    "cookpad.com": (("", _source()),),
    "cookeatshare.com": (("", _OFFLINE),),
    "www.yummly.com": (("", _OFFLINE),),
    "www.tasteofhome.com": (("", _archive("2019")),),
    "www.foodnetwork.com": (("", _source()),),
    "food52.com": (("", _source()),),
    "www.kraftrecipes.com": (("", _OFFLINE),),
    "recipeland.com": (("", _source()),),
    "recipes-plus.com": (("", _OFFLINE),),
    # Records keep only /recipes/<id>, which returns 404; the site serves /recipes/<id>-<title slug>.
    "cooking.nytimes.com": (("", _source(title_slug=True)),),
    "www.foodandwine.com": (("", _archive("2016")),),
    "www.seriouseats.com": (("", _archive("2020")),),
    "www.cookstr.com": (("", _archive("2015")),),
    "www.foodgeeks.com": (("", _source()),),
    "www.chowhound.com": (("", _archive("2019")),),
    "online-cookbook.com": (("", _source()),),
    "www.vegetariantimes.com": (("", _archive("2015")),),
    "www.delish.com": (("", _archive("2015")),),
    "allrecipes.com": (("", _archive("2015")),),
    "www.landolakes.com": (("", _archive("2015")),),
    "www.foodrepublic.com": (("", _source()),),
    "www.lovefood.com": (("", _source()),),
    "icook.tw": (("", _source()),),
}


def title_slug(title: str) -> str:
    """cooking.nytimes.com's slug: accents stripped, apostrophes dropped, other non-alphanumeric runs become "-"."""
    text = "".join(character for character in unicodedata.normalize("NFKD", title)
                   if not unicodedata.combining(character)).lower()
    return re.sub(r"[^a-z0-9]+", "-", re.sub(r"['\u2019]", "", text)).strip("-")


def recipe_link(url: str | None, title: str | None,
                rules: dict[str, tuple[tuple[str, SiteRule], ...]] = SITE_RULES) -> tuple[str | None, int]:
    """The card link and its LINK_STATUSES code for a URL already normalized by public_source_url."""
    if url is None:
        return None, NONE
    parts = urlsplit(url)
    if parts.hostname not in rules:
        raise ValueError(f"{parts.hostname}: no measured card-link rule; measure this site before building")
    rule = next((rule for prefix, rule in rules[parts.hostname] if parts.path.startswith(prefix)), None)
    if rule is None:
        raise ValueError(f"{parts.hostname}: no card-link rule matches the recorded path")
    if rule.status == ARCHIVE:
        return f"{ARCHIVE_PREFIX}{rule.year}/{url}", ARCHIVE
    if rule.status != SOURCE:
        return None, OFFLINE
    path = parts.path
    if rule.title_slug and _BARE_RECIPE_ID.fullmatch(path):
        slug = title_slug(title or "")
        if not slug:
            # OFFLINE means the site was measured not to serve; a missing slug is a different failure.
            raise ValueError(f"{parts.hostname}: this record's title gives no slug for its recorded path")
        path = f"{path}-{slug}"
    return urlunsplit(("https", rule.host or parts.hostname, path, parts.query, parts.fragment)), SOURCE
