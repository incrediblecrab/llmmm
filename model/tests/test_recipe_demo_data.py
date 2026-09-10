"""Public, offline tests: no training corpus, model weights or network required."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/curate_recipe_demo.py"
SPEC = importlib.util.spec_from_file_location("public_recipe_demo_curation", SCRIPT)
demo = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = demo
SPEC.loader.exec_module(demo)


@pytest.fixture
def catalog():
    records = [
        json.loads(line)
        for line in (demo.OUTPUT / "recipes.jsonl").read_text().splitlines()
    ]
    manifest = json.loads((demo.OUTPUT / "sources.json").read_text())
    return records, manifest


def test_distributed_catalog_is_complete_and_self_consistent(catalog):
    records, manifest = catalog
    demo.validate_catalog(records, manifest)
    assert demo.counts(records) == {
        "recipes": 12,
        "known_total_minutes": 9,
        "total_minutes_at_most_30": 8,
        "known_servings": 6,
    }
    for filename, expected in demo.outputs(records, manifest).items():
        assert (demo.OUTPUT / filename).read_bytes() == expected


def test_all_ingredient_lines_are_accounted_for(catalog):
    records, manifest = catalog
    for recipe, record, source in zip(demo.RECIPES, records, manifest["sources"]):
        assert len(record["raw_ingredients"]) == len(recipe.rules)
        assert [item["raw_ingredient"] for item in source["ingredient_mapping"]] == record["raw_ingredients"]
        assert all(item["canonical_ingredients"] or item["unmapped_reason"]
                   for item in source["ingredient_mapping"])
        assert set(record["unmapped_ingredients"]) <= set(record["raw_ingredients"])
    assert any(record["unmapped_ingredients"] for record in records)
    by_id = {record["id"]: record for record in records}
    assert by_id["potato-salad"]["unmapped_ingredients"] == ["Pepper"]
    assert "½ cup (100 g) all-purpose flour" in by_id["apple-crisp"]["raw_ingredients"]
    assert by_id["chocolate-chip-cookies"]["instructions"][0] == (
        "Preheat oven to 375°F (210°C), or 350°F (195°C) if you want chewy cookies."
    )


@pytest.mark.parametrize(("fields", "minutes", "evidence"), [
    ({}, None, None),
    ({"time": "20 minutes"}, 20, "20 minutes"),
    ({"time": "½ hour"}, 30, "½ hour"),
    ({"time": "1.5 hours"}, 90, "1.5 hours"),
    ({"time": "Prep: 10 minutes<br>Cooking: 5 minutes per waffle"},
     None, "Prep: 10 minutes<br>Cooking: 5 minutes per waffle"),
    ({"time": "Prep: 20 minutes<br>Cooking: 12 minutes<br>Total: 35 minutes"},
     35, "Total: 35 minutes"),
    ({"time": "20–30 minutes"}, None, "20–30 minutes"),
    ({"time": "20-30 minutes"}, None, "20-30 minutes"),
    ({"time": "~10 minutes"}, None, "~10 minutes"),
    ({"time": "Total: 20–30 minutes"}, None, "Total: 20–30 minutes"),
    ({"time": "10 minutes + 5 minutes"}, None, "10 minutes + 5 minutes"),
])
def test_only_explicit_unambiguous_overall_times_are_used(fields, minutes, evidence):
    assert demo.time_metadata(fields)[:2] == (minutes, evidence)


@pytest.mark.parametrize(("fields", "servings", "evidence"), [
    ({}, None, None),
    ({"servings": "4"}, 4, "4"),
    ({"servings": "4 persons"}, 4, "4 persons"),
    ({"servings": "4-6"}, None, "4-6"),
    ({"servings": "4–6"}, None, "4–6"),
    ({"servings": "2 loaves"}, None, "2 loaves"),
    ({"yield": "56 cookies"}, None, None),
    ({"servings": "", "yield": "3 ea. 4-section waffles"}, None, None),
])
def test_serving_ranges_and_item_yields_are_not_invented(fields, servings, evidence):
    assert demo.servings_metadata(fields)[:2] == (servings, evidence)


def test_missing_metadata_and_additional_attribution_are_preserved(catalog):
    records, _ = catalog
    by_id = {record["id"]: record for record in records}
    assert by_id["waffles"]["total_minutes"] is None
    assert by_id["waffles"]["time_evidence"] == "Prep: 10 minutes<br>Cooking: 5 minutes per waffle"
    assert by_id["red-lentil-soup"]["total_minutes"] is None
    assert by_id["red-lentil-soup"]["time_evidence"] is None
    assert by_id["apple-crisp"]["total_minutes"] is None
    assert by_id["potato-curry"]["servings"] is None
    assert by_id["potato-curry"]["servings_evidence"] == "4-6"
    assert by_id["peanut-butter-cookies"]["servings"] is None
    assert by_id["mozzarella-bruschetta"]["servings"] is None
    assert "en.wikipedia.org/w/index.php?title=Porridge&action=history" in by_id["oat-porridge"]["attribution"]


def test_provenance_records_actual_fetched_sources_and_license_evidence(catalog):
    records, manifest = catalog
    assert manifest["vocabulary"]["url"] == demo.VOCABULARY_URL
    assert manifest["vocabulary"]["sha256"] == demo.VOCABULARY_SHA256
    assert len(manifest["license_evidence"]) == 3
    for recipe, record, source in zip(demo.RECIPES, records, manifest["sources"]):
        assert source["fetched_source"]["sha256"] == recipe.sha256
        assert source["fetched_source"]["url"].endswith(f"oldid={recipe.revision}&action=raw")
        assert source["fetched_source"]["retrieved_at"] == record["retrieved_at"]
        assert source["per_page_license_evidence"]["url"] == record["source_url"]
        assert source["per_page_license_evidence"]["rel_license"].startswith(demo.LICENSE_URL)
        assert record["license"] == "CC-BY-SA-4.0"
        assert "action=history" in record["attribution_url"]


@pytest.mark.parametrize("corruption", ["unknown-name", "lost-raw-line", "guessed-time", "nan-time", "boolean-servings"])
def test_catalog_validation_rejects_corrupted_records(catalog, corruption):
    records, manifest = copy.deepcopy(catalog)
    if corruption == "unknown-name":
        records[0]["canonical_ingredients"].append("invented_ingredient")
    elif corruption == "lost-raw-line":
        records[0]["raw_ingredients"].pop()
    elif corruption == "guessed-time":
        records[-1]["total_minutes"] = 25
    elif corruption == "nan-time":
        records[0]["total_minutes"] = float("nan")
    else:
        records[0]["servings"] = True
    with pytest.raises(demo.CurationError):
        demo.validate_catalog(records, manifest)


def test_narrow_extractor_preserves_measurements_and_rejects_unreviewed_markup():
    assert demo.plain_text("¼&ndash;½ [[Cookbook:Cup|cup]] (50–100 ml) [[Cookbook:Milk|milk]]") == (
        "¼–½ cup (50–100 ml) milk"
    )
    with pytest.raises(demo.CurationError, match="Unreviewed source markup"):
        demo.plain_text("{{unreviewed-template|2 cups}}")
    with pytest.raises(demo.CurationError, match="Unreviewed Ingredients structure"):
        demo.source_list("== Ingredients ==\n* 1 egg\nImportant unreviewed prose\n", "Ingredients", "*")


def test_distributed_verification_never_fetches(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Distributed verification must not fetch sources")

    monkeypatch.setattr(demo.Fetcher, "get", forbidden)
    assert demo.main(["--verify"]) == 0


def test_byte_identical_reproduction_from_cached_sources_when_available(catalog):
    _, manifest = catalog

    def cache_paths(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "cache_path":
                    yield demo.ROOT / child
                else:
                    yield from cache_paths(child)
        elif isinstance(value, list):
            for child in value:
                yield from cache_paths(child)

    if not all(path.is_file() for path in cache_paths(manifest)):
        pytest.skip("Optional Git-ignored public-source cache is not restored")
    records, regenerated = demo.make_catalog(demo.Fetcher(offline=True))
    for filename, expected in demo.outputs(records, regenerated).items():
        assert (demo.OUTPUT / filename).read_bytes() == expected
