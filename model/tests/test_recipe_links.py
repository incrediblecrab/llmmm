from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from ingredient_model.ingredient_catalog import public_source_url
from ingredient_model.recipe_links import (
    ARCHIVE, ARCHIVE_PREFIX, LINK_STATUSES, NONE, OFFLINE, SITE_RULES, SOURCE, recipe_link, title_slug,
)

HEALTH = Path(__file__).resolve().parents[1] / "results/recipe_link_health.json"


def test_a_missing_recorded_url_has_no_link():
    assert recipe_link(None, "Anything") == (None, NONE)


def test_live_sites_link_to_the_recorded_page_over_https():
    assert recipe_link("http://www.food.com/recipe/1", "Pie") == ("https://www.food.com/recipe/1", SOURCE)
    assert recipe_link("https://www.nefisyemektarifleri.com/tarif/1/?a=1", None) == (
        "https://www.nefisyemektarifleri.com/tarif/1/?a=1", SOURCE)


def test_cookbooks_links_use_the_host_its_redirects_end_on():
    url, _ = public_source_url("www.cookbooks.com/Recipe-Details.aspx?id=1039023")
    assert recipe_link(url, "Chili") == ("https://cookbooks.com/Recipe-Details.aspx?id=1039023", SOURCE)


@pytest.mark.parametrize("title,slug", [
    ("Turkey Meatloaf", "turkey-meatloaf"),
    ("Lentil Soup With Cilantro (Lots of It)", "lentil-soup-with-cilantro-lots-of-it"),
    ("Kittichai's Rice Crackers With Chicken-And-Shrimp Dipping Sauce",
     "kittichais-rice-crackers-with-chicken-and-shrimp-dipping-sauce"),
    ("Oeufs à la tripes au cari", "oeufs-a-la-tripes-au-cari"),
    ("Anne Rosenzweig\u2019s Chili", "anne-rosenzweigs-chili"),
])
def test_nyt_cooking_slugs_drop_accents_and_apostrophes(title, slug):
    assert title_slug(title) == slug
    assert recipe_link("http://cooking.nytimes.com/recipes/1012584", title) == (
        f"https://cooking.nytimes.com/recipes/1012584-{slug}", SOURCE)


@pytest.mark.parametrize("title", [None, "!!!"])
def test_nyt_cooking_records_without_a_usable_title_fail_instead_of_claiming_the_site_is_offline(title):
    with pytest.raises(ValueError, match="gives no slug"):
        recipe_link("http://cooking.nytimes.com/recipes/1", title)


def test_archived_sites_wrap_the_exact_recorded_url():
    url = "http://allrecipes.com/Recipe/Easy-Pie/Detail.aspx?evt19=1"
    link, status = recipe_link(url, "Easy Pie")
    assert status == ARCHIVE
    assert link == f"{ARCHIVE_PREFIX}2015/{url}"


def test_sites_serving_neither_page_nor_archive_get_no_link():
    assert recipe_link("http://tastykitchen.com/recipes/1", "Pie") == (None, OFFLINE)
    assert recipe_link("http://www.epicurious.com/recipes/member/views/pie-1", "Pie") == (None, OFFLINE)
    assert recipe_link("http://www.epicurious.com/recipes/food/views/pie-1", "Pie") == (
        "https://www.epicurious.com/recipes/food/views/pie-1", SOURCE)


@pytest.mark.parametrize("url,match", [
    ("https://unmeasured.example/recipe/1", "no measured card-link rule"),
    ("http://www.epicurious.com/articles/pie", "no card-link rule matches"),
])
def test_unmeasured_sites_and_paths_fail_instead_of_guessing(url, match):
    with pytest.raises(ValueError, match=match):
        recipe_link(url, "Pie")


def test_every_rule_is_well_formed():
    for host, rules in SITE_RULES.items():
        assert host == host.lower() and rules
        for prefix, rule in rules:
            assert prefix == "" or prefix.startswith("/")
            assert rule.status in (SOURCE, ARCHIVE, OFFLINE)
            assert (rule.year is not None) == (rule.status == ARCHIVE)
            assert rule.year is None or (len(rule.year) == 4 and rule.year.isdigit())


def test_every_rule_is_the_decision_its_recorded_link_checks_imply():
    needed = lambda n: math.ceil(5 * n / 6)
    decided = {}
    for site in json.loads(HEALTH.read_text())["sites"]:
        live, archive, retry, recheck = site["live"], site.get("archive"), site.get("archive_round2"), site.get("round3")
        rechecks = (recheck["live"], recheck["archive"]) if recheck else ()
        for check in filter(None, (live, archive, retry, *rechecks)):
            assert check["successes"] == sum(link["success"] for link in check["links"]), site["host"]
        assert site["needed"] == needed(site["n"]) == needed(len(live["links"]))
        if live["successes"] >= site["needed"]:
            expected = ("source", None)
        elif archive and archive["successes"] >= site["needed"]:
            expected = ("archive", archive["year"])
        elif retry and len(retry["links"]) == 12 and retry["successes"] >= needed(12):
            expected = ("archive", retry["year"])
        else:
            expected = ("offline", None)
        if recheck:
            # A site rechecked on a fresh sample after failing the held-out check; the recheck could not make it offline.
            assert recheck["n"] == len(recheck["live"]["links"]) == len(recheck["archive"]["links"]) == 12
            if recheck["live"]["successes"] >= needed(12):
                expected = ("source", None)
            elif recheck["archive"]["successes"] >= needed(12):
                expected = ("archive", recheck["archive"]["year"])
        assert (site["decision"], site["archive_year"]) == expected, site["host"]
        decided[site["host"], site["path_prefix"]] = expected
    assert {(host, prefix): (LINK_STATUSES[rule.status], rule.year)
            for host, rules in SITE_RULES.items() for prefix, rule in rules} == decided


def test_held_out_link_checks_add_up_and_the_shipped_links_passed():
    checks = json.loads(HEALTH.read_text())["held_out_checks"]
    for check in checks:
        links = check["links"]
        assert check["n"] == len(links) == len({link["id"] for link in links}) == 300
        assert check["successes"] == sum(link["success"] for link in links)
        # An automated pass stands unless a hand read found a different recipe; every other row was classified by hand.
        flipped = set(check["hand_read"]["flipped_to_failure"])
        assert all(link["success"] != (link["id"] in flipped) for link in links if link["automated"] == "ok")
        assert all("hand_read" in link for link in links if link["automated"] != "ok")
        hosts = {}
        for link in links:
            counts = hosts.setdefault(link["host"], {"n": 0, "successes": 0})
            counts["n"] += 1
            counts["successes"] += link["success"]
        assert check["by_host"] == hosts
        assert check["gate"] == ("passed" if check["successes"] >= math.ceil(0.95 * check["n"]) else "failed")
        assert check["wilson_95"][0] < check["successes"] / check["n"] < check["wilson_95"][1]
    assert [check["draw"] for check in checks] == list(range(1, len(checks) + 1))
    assert checks[-1]["links_checked"] == "the released index" and checks[-1]["gate"] == "passed"


def test_the_readme_reports_the_held_out_checks_as_recorded():
    first, *_, last = json.loads(HEALTH.read_text())["held_out_checks"]
    low, high = (f"{100 * bound:.1f}" for bound in last["wilson_95"])
    readme = (HEALTH.parents[2] / "README.md").read_text()
    assert (f"Of {last['n']} card links drawn at random after the rules were fixed, {last['successes']} opened their "
            f"recipe (Wilson 95% interval {low}–{high}%). An earlier draw scored {first['successes']} of "
            f"{first['n']}") in readme
