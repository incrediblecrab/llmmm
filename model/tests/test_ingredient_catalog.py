from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3

import numpy as np
import pytest

from ingredient_model.data.recipe_catalog import _SCHEMA
from ingredient_model.ingredient_catalog import (
    ARRAYS, TEXT_MANIFEST, build_ingredient_catalog, ingredient_lines, load_ingredient_catalog,
    load_text_shards, public_source_url, read_text_shard, recipe_title,
)
from ingredient_model.recipe_links import ARCHIVE, LINK_STATUSES, NONE, OFFLINE, SOURCE, SiteRule

TITLES = ["Egg &amp; Salt\n  Toast", "PB&J; Custard", ""]
RAW_INGREDIENTS = [
    "2 eggs\x1f1 tsp salt",
    "<table><tr><td>milk</td><td>1 cup</td></tr><tr><td>egg</td><td>2</td></tr></table>",
    "",
]
URLS = ["https://example.test/recipe/0", "www.example.test/recipe/1", "https://example.test/recipe/2"]
LINK_RULES = {
    "example.test": (("/recipe/2", SiteRule(ARCHIVE, year="2015")), ("", SiteRule(SOURCE))),
    "www.example.test": (("", SiteRule(OFFLINE)),),
}


def build(*args, **kwargs):
    return build_ingredient_catalog(*args, link_rules=LINK_RULES, **kwargs)


@pytest.fixture
def canonical_catalog(tmp_path):
    corpus = tmp_path / "canonical.npz"
    flat = np.asarray([0, 1, 0, 2, 1], dtype="<u2")
    offsets = np.asarray([0, 2, 4, 5], dtype="<u8")
    np.savez(corpus, flat=flat, offsets=offsets)
    catalog = tmp_path / "catalog.sqlite"
    metadata = {
        "schema_version": 1, "partial": False, "n_recipes": 3, "n_slots": 5,
        "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "text_index_sha256": "a" * 64, "vocabulary": ["egg", "salt", "milk"],
        "ingredient_frequency": [2, 2, 1], "coverage": {},
    }
    with sqlite3.connect(catalog) as connection:
        connection.executescript(_SCHEMA)
        connection.executemany("INSERT INTO metadata VALUES (?,?)",
                               [(key, json.dumps(value)) for key, value in metadata.items()])
        for position in range(3):
            fields = {
                "id": position, "source": "fixture", "language": "en",
                "title": TITLES[position],
                "steps": "PRIVATE INSTRUCTIONS MUST NEVER BE EXPORTED" if position < 2 else "",
                "ingredient_quantities": "PRIVATE QUANTITIES MUST NEVER BE EXPORTED",
                "raw_ingredients": RAW_INGREDIENTS[position],
                "ingredient_ids": flat[offsets[position]:offsets[position + 1]].tobytes(),
                "url": URLS[position],
                "total_minutes": 10.25 if position == 0 else None,
                "time_status": "source_total" if position == 0 else "unknown",
                "servings": 2 if position == 0 else None,
                "servings_status": "source_servings" if position == 0 else "unknown",
                "metadata_status": "fixture", "metadata_match_count": 1,
            }
            connection.execute(
                f"INSERT INTO recipes ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})",
                list(fields.values()))
    return catalog, corpus


def test_all_rows_roundtrip_with_titles_and_lines_but_no_instructions(canonical_catalog, tmp_path):
    catalog, corpus = canonical_catalog
    output = tmp_path / "ingredient-only"
    report = build(catalog, corpus, output, rows_per_shard=2, rows_per_text_shard=2)
    metadata, arrays = load_ingredient_catalog(output)
    assert report["coverage"]["records"] == 3
    assert report["coverage"]["singletons"] == 1
    assert report["coverage"]["source_total_times"] == 1
    assert report["all_rows_compared_with_canonical_arrays"] is True
    assert report["uploaded"] is False
    assert metadata["publication"]["status"] == "local_export_only"
    assert arrays["ingredients"].tolist() == [0, 1, 0, 2, 1]
    assert arrays["lengths"].tolist() == [2, 2, 1]
    assert arrays["total_minutes"][0] == 10.25
    assert np.isnan(arrays["total_minutes"][1:]).all()
    assert metadata["ingredient_frequency"] == [2, 2, 1]
    assert len(metadata["url_shards"]) == 2
    assert report["cooking_instructions_exported"] is False
    assert {"recipe_title", "ingredient_lines"} <= set(metadata["fields_included"])
    assert {"instructions", "steps", "ingredient_quantities"} <= set(metadata["fields_excluded"])
    for path in output.rglob("*"):
        if path.is_file():
            data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
            assert b"PRIVATE" not in data
    first = json.loads(gzip.decompress((output / metadata["url_shards"][0]["file"]).read_bytes()))
    assert first == {"first_id": 0, "urls": ["https://example.test/recipe/0", "http://www.example.test/recipe/1"]}
    links = [json.loads(gzip.decompress((output / record["file"]).read_bytes())) for record in metadata["link_shards"]]
    assert links == [{"first_id": 0, "links": ["https://example.test/recipe/0", None]},
                     {"first_id": 2, "links": ["https://web.archive.org/web/2015/https://example.test/recipe/2"]}]
    assert arrays["link_status"].tolist() == [SOURCE, OFFLINE, ARCHIVE]
    assert report["coverage"]["link_statuses"] == {"source": 1, "offline": 1, "archive": 1}
    shards = load_text_shards(output, metadata)
    assert [(shard["file"], shard["first_id"], shard["rows"]) for shard in shards] == [
        ("text/0000.json.gz", 0, 2), ("text/0001.json.gz", 2, 1)]
    text = [read_text_shard(output, shard) for shard in shards]
    assert text == [
        (["Egg & Salt Toast", "PB&J; Custard"], [["2 eggs", "1 tsp salt"], ["milk: 1 cup", "egg: 2"]]),
        ([None], [None]),
    ]
    assert {name: report["coverage"][name] for name in ("recipe_titles", "ingredient_line_records", "ingredient_lines")} == {
        "recipe_titles": 2, "ingredient_line_records": 2, "ingredient_lines": 4}
    with pytest.raises(FileExistsError):
        build(catalog, corpus, output)


@pytest.mark.parametrize("value,expected", [
    (None, None), ("", None), ("  \n ", None),
    ("Mac &amp; Cheese", "Mac & Cheese"), ("PB&J; &pepper", "PB&J; &pepper"),
    ("&quot;Best&quot;\tPie", '"Best" Pie'), ("A\x00B", "A B"),
])
def test_titles_decode_only_complete_references_and_collapse_whitespace(value, expected):
    assert recipe_title(value) == expected


@pytest.mark.parametrize("value,expected", [
    (None, None), ("", None), ("\x1f \x1f", None),
    ("1 egg\x1f2 cups flour", ["1 egg", "2 cups flour"]),
    ("1 egg\n\n 2 cups  flour\r\n", ["1 egg", "2 cups flour"]),
    ("<table><tr><th>rice</th><td>2 cups</td></tr><tr><td>salt</td><td></td></tr></table>", ["rice: 2 cups", "salt"]),
    ("<b>1</b> egg", ["<b>1</b> egg"]),
    ("salt &amp; pepper &amp", ["salt & pepper &amp"]),
])
def test_ingredient_lines_keep_wording_and_split_only_recorded_breaks(value, expected):
    assert ingredient_lines(value) == expected


@pytest.mark.parametrize("value,source,expected", [
    ("2 cups rice ; 1 egg ;   salt to taste", "allrecipes-33k", ["2 cups rice", "1 egg", "salt to taste"]),
    ("1 cup ricotta (fresh ; 8 ounces)\x1f1 egg", "kaggle-food-13k", ["1 cup ricotta (fresh ; 8 ounces)", "1 egg"]),
    ("Köfteler için ; 1 kaşık un\n2 yumurta", "turkish-102k", ["Köfteler için ; 1 kaşık un", "2 yumurta"]),
    ("3 lbs. pork | 1 onion chopped|salt |", "filipino-2k", ["3 lbs. pork", "1 onion chopped", "salt"]),
    ("知味人生|vlog裱花", "02-xiachufang", ["知味人生|vlog裱花"]),
    ("a|b ; c", None, ["a|b", "c"]),
])
def test_one_line_lists_split_only_on_their_recorded_separator(value, source, expected):
    assert ingredient_lines(value, source) == expected


@pytest.mark.parametrize("value,expected", [
    ("{'count': '3 шт', 'name': 'Яйцо куриное'}\x1f{'count': None, 'name': 'Мука'}\x1f{'count': ' по вкусу', 'name': 'Соль'}",
     ["Яйцо куриное: 3 шт", "Мука", "Соль: по вкусу"]),
    ("{'name': '洋蔥切片', 'unit': '1顆'}\x1f{'name': \"Mom's 'best' sauce\", 'unit': ''}", ["洋蔥切片: 1顆", "Mom's 'best' sauce"]),
    ("{'count': '1 ; 2 ст. л.', 'name': 'Сахар'}", ["Сахар: 1 ; 2 ст. л."]),
    ("{'count': None, 'name': None}\x1f1 egg", ["1 egg"]),
    # Anything else that merely looks like a dict keeps its recorded wording.
    ("{'count': 2, 'name': 'egg'}\x1f{'name': 'egg', 'amount': '2'}\x1f{'count': '2', 'name': 'egg'", [
        "{'count': 2, 'name': 'egg'}", "{'name': 'egg', 'amount': '2'}", "{'count': '2', 'name': 'egg'"]),
    ("{For the dressing}\x1f{'name': 'salt'", ["{For the dressing}", "{'name': 'salt'"]),
])
def test_lines_recorded_as_name_and_amount_fields_read_name_colon_amount(value, expected):
    assert ingredient_lines(value) == expected


@pytest.mark.parametrize("value,source,expected", [
    ("Майонез: None\x1fРис: 1 стак.\x1fСоль : None", "03-povarenok", ["Майонез", "Рис: 1 стак.", "Соль"]),
    ("Соль: None", "03-povarenok", ["Соль"]),
    ("Сахар: None Such", "03-povarenok", ["Сахар: None Such"]),
    # Elsewhere "None" is recorded wording: a cocktail without a garnish, a RecipeNLG line, a brand.
    ("Garnish: None", "kaggle-food-13k", ["Garnish: None"]),
    ("None\x1f1 pkg. None Such mincemeat", "01-recipenlg", ["None", "1 pkg. None Such mincemeat"]),
])
def test_null_amounts_the_catalog_wrote_as_none_are_dropped_only_for_povarenok(value, source, expected):
    assert ingredient_lines(value, source) == expected


def test_scheme_less_urls_use_http_because_https_fails_on_some_hosts():
    assert public_source_url("www.cookbooks.com/Recipe-Details.aspx?id=1") == (
        "http://www.cookbooks.com/Recipe-Details.aspx?id=1", "source_url")


def test_canonical_mismatch_never_publishes_partial_output(canonical_catalog, tmp_path):
    catalog, corpus = canonical_catalog
    with sqlite3.connect(catalog) as connection:
        connection.execute("UPDATE recipes SET ingredient_ids=? WHERE id=1",
                           (np.asarray([0, 1], dtype="<u2").tobytes(),))
    output = tmp_path / "mismatch"
    with pytest.raises(ValueError, match="differ from the canonical"):
        build(catalog, corpus, output)
    assert not output.exists()


@pytest.mark.parametrize("value,status", [
    ("https://example.test/r/1?access_token=secret", "sensitive_query_not_exported"),
    ("https://person:password@example.test/r/1", "invalid_or_oversized"),
    ("javascript:alert(1)", "invalid_or_oversized"),
    ("", "missing"),
    (None, "missing"),
])
def test_invalid_or_sensitive_urls_are_omitted_with_counted_reasons(value, status):
    assert public_source_url(value) == (None, status)


def test_fact_only_urls_are_preserved_without_guessing():
    assert public_source_url("https://example.test/recipe?id=12") == (
        "https://example.test/recipe?id=12", "source_url")


def test_missing_or_invalid_links_do_not_remove_ingredient_records(canonical_catalog, tmp_path):
    catalog, corpus = canonical_catalog
    with sqlite3.connect(catalog) as connection:
        connection.execute("UPDATE recipes SET url=NULL WHERE id=1")
        connection.execute("UPDATE recipes SET url='javascript:alert(1)' WHERE id=2")
    output = tmp_path / "ingredient-only"
    report = build(catalog, corpus, output)
    metadata, arrays = load_ingredient_catalog(output)
    assert metadata["n_recipes"] == 3
    assert arrays["link_status"].tolist() == [SOURCE, NONE, NONE]
    assert report["coverage"]["url_statuses"] == {
        "source_url": 1, "missing": 1, "invalid_or_oversized": 1,
    }
    assert report["coverage"]["link_statuses"] == {"source": 1, "none": 2}


def test_a_recorded_site_without_a_measured_link_rule_fails_the_build(canonical_catalog, tmp_path):
    catalog, corpus = canonical_catalog
    output = tmp_path / "unmeasured"
    with pytest.raises(ValueError, match="no measured card-link rule"):
        build_ingredient_catalog(catalog, corpus, output)
    assert not output.exists()


def test_array_corruption_is_detected(canonical_catalog, tmp_path):
    catalog, corpus = canonical_catalog
    output = tmp_path / "ingredient-only"
    build(catalog, corpus, output)
    path = output / ARRAYS["ingredients"][0]
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="integrity"):
        load_ingredient_catalog(output)


@pytest.mark.parametrize("shards", ["url_shards", "link_shards"])
@pytest.mark.parametrize("patch", [
    {"file": "../private.env"},
    {"first_id": 1},
    {"rows": 2},
])
def test_url_and_link_shards_cannot_escape_the_index_or_misalign_records(canonical_catalog, tmp_path, shards, patch):
    catalog, corpus = canonical_catalog
    output = tmp_path / "ingredient-only"
    build(catalog, corpus, output)
    path = output / "ingredient-index.json"
    metadata = json.loads(path.read_text())
    metadata[shards][0].update(patch)
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="shard"):
        load_ingredient_catalog(output)


def _rewrite_gzip_json(path, value):
    raw = (json.dumps(value) + "\n").encode()
    path.write_bytes(gzip.compress(raw, mtime=0))
    return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "raw_bytes": len(raw), "raw_sha256": hashlib.sha256(raw).hexdigest()}


def test_text_shard_corruption_is_detected(canonical_catalog, tmp_path):
    catalog, corpus = canonical_catalog
    output = tmp_path / "ingredient-only"
    build(catalog, corpus, output)
    metadata, _ = load_ingredient_catalog(output)
    shard = load_text_shards(output, metadata)[0]
    path = output / shard["file"]
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="integrity"):
        read_text_shard(output, shard)


@pytest.mark.parametrize("content", [
    {"first_id": 1, "titles": [None, None, None], "ingredient_lines": [None, None, None]},
    {"first_id": 0, "titles": [None, None], "ingredient_lines": [None, None, None]},
    {"first_id": 0, "titles": ["a\nb", None, None], "ingredient_lines": [None, None, None]},
    {"first_id": 0, "titles": [None, None, None], "ingredient_lines": [[], None, None]},
    {"first_id": 0, "titles": [None, None, None], "ingredient_lines": [["x" * 20_000], None, None]},
    {"first_id": 0, "titles": [None, None, None], "ingredient_lines": [None, None, None], "steps": ["cook"]},
])
def test_text_shards_cannot_misalign_or_smuggle_unbounded_text(canonical_catalog, tmp_path, content):
    catalog, corpus = canonical_catalog
    output = tmp_path / "ingredient-only"
    build(catalog, corpus, output)
    metadata, _ = load_ingredient_catalog(output)
    shard = load_text_shards(output, metadata)[0]
    shard.update(_rewrite_gzip_json(output / shard["file"], content))
    with pytest.raises(ValueError):
        read_text_shard(output, shard)


@pytest.mark.parametrize("patch", [
    lambda manifest: manifest["shards"][0].update(file="../private.env"),
    lambda manifest: manifest["shards"][0].update(first_id=1),
    lambda manifest: manifest["shards"][0].update(rows=2),
    lambda manifest: manifest["shards"].append(dict(manifest["shards"][0])),
])
def test_text_manifest_cannot_escape_the_index_or_misalign_records(canonical_catalog, tmp_path, patch):
    catalog, corpus = canonical_catalog
    output = tmp_path / "ingredient-only"
    build(catalog, corpus, output)
    manifest = json.loads(gzip.decompress((output / TEXT_MANIFEST).read_bytes()))
    patch(manifest)
    index_path = output / "ingredient-index.json"
    metadata = json.loads(index_path.read_text())
    metadata["text_manifest"].update(_rewrite_gzip_json(output / TEXT_MANIFEST, manifest))
    index_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="shard"):
        load_ingredient_catalog(output)


def test_text_manifest_must_match_its_declared_digest(canonical_catalog, tmp_path):
    catalog, corpus = canonical_catalog
    output = tmp_path / "ingredient-only"
    build(catalog, corpus, output)
    _rewrite_gzip_json(output / TEXT_MANIFEST, {"shards": []})
    with pytest.raises(ValueError, match="integrity"):
        load_ingredient_catalog(output)


def test_unknown_link_status_codes_are_rejected(canonical_catalog, tmp_path):
    catalog, corpus = canonical_catalog
    output = tmp_path / "ingredient-only"
    build(catalog, corpus, output)
    path, raw = output / ARRAYS["link_status"][0], bytes([SOURCE, OFFLINE, len(LINK_STATUSES)])
    path.write_bytes(gzip.compress(raw, mtime=0))
    index_path = output / "ingredient-index.json"
    metadata = json.loads(index_path.read_text())
    metadata["arrays"]["link_status"].update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                             raw_sha256=hashlib.sha256(raw).hexdigest())
    index_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="unknown ingredient or metadata codes"):
        load_ingredient_catalog(output)
