import assert from "node:assert/strict";
import test from "node:test";
import { prepareCatalog, searchRecipes, sourceUrl } from "../search.js";
import { policy } from "./fixtures.js";

function recipe(id, ingredients, total, servings = null) {
  return {
    id, title: id, language: "en", canonical_ingredients: ingredients,
    raw_ingredients: ["Synthetic test fixture; not a cooking recipe."],
    instructions: ["Synthetic test fixture."], total_minutes: total, servings,
    time_evidence: total === null ? null : `${total} minutes`,
    servings_evidence: servings === null ? null : `${servings} servings`,
    source_url: "https://en.wikibooks.org/wiki/Cookbook:Test",
    attribution_url: "https://en.wikibooks.org/w/index.php?title=Cookbook:Test&action=history",
    attribution: "Test fixture", license: "cc-by-sa-4.0", unmapped_ingredients: [],
  };
}

function rawCatalog() {
  return {
    schema_version: 1, statistics_scope: "public-sample",
    vocabulary: ["egg", "milk", "salt"], n_recipes: 4, ingredient_frequency: [3, 2, 3],
    recipes: [
      recipe("a", ["egg", "salt"], 10, 2),
      recipe("b", ["egg", "milk"], null, 4),
      recipe("c", ["egg", "milk", "salt"], 20, 1),
      recipe("d", ["salt"], 5),
    ],
  };
}

function catalog() {
  return prepareCatalog(rawCatalog());
}

test("hard constraints precede either ranking method", () => {
  for (const ranking of ["learned", "heuristic"]) {
    const result = searchRecipes(catalog(), policy(), {
      available_ingredients: [" EGG ", "salt"], must_use: ["egg"], exclude: ["milk"],
      max_total_minutes: 10, max_missing: 0, min_servings: 2,
    }, ranking);
    assert.deepEqual(result.matches.map((row) => row.id), ["a"]);
    assert.deepEqual(result.matches[0].missing_ingredients, []);
    assert.equal(result.scanned, 4);
  }
});

test("unknown times and serving counts do not pass corresponding limits", () => {
  const base = { available_ingredients: ["egg", "milk", "salt"], max_missing: 0 };
  const timed = searchRecipes(catalog(), policy(), { ...base, max_total_minutes: 10 });
  assert.deepEqual(new Set(timed.matches.map((row) => row.id)), new Set(["a", "d"]));
  const served = searchRecipes(catalog(), policy(), { ...base, min_servings: 2 });
  assert.deepEqual(new Set(served.matches.map((row) => row.id)), new Set(["a", "b"]));
  assert.equal(searchRecipes(catalog(), policy(), { ...base, max_total_minutes: 0 }).feasible_count, 0);
});

test("missing limits, required ingredients and overlap are not relaxed for empty results", () => {
  assert.equal(searchRecipes(catalog(), policy(), {
    available_ingredients: ["egg"], max_missing: 0,
  }).feasible_count, 0);
  const required = searchRecipes(catalog(), policy(), {
    available_ingredients: ["egg"], must_use: ["milk"], max_missing: 1,
  });
  assert.deepEqual(required.matches.map((row) => row.id), ["b"]);
  assert.deepEqual(required.matches[0].missing_ingredients, ["milk"]);
  assert.equal(searchRecipes(catalog(), policy(), {
    available_ingredients: ["milk"], must_use: ["salt"], exclude: ["egg"], max_missing: null,
  }).feasible_count, 1);
});

test("invalid queries fail explicitly", () => {
  for (const patch of [
    { available_ingredients: [] }, { available_ingredients: ["eggs"] },
    { must_use: ["milk"], exclude: ["milk"] }, { exclude: ["egg"] },
    { max_missing: true }, { max_missing: -1 }, { max_missing: 0.5 },
    { max_total_minutes: "30" }, { max_total_minutes: NaN }, { min_servings: 0 },
    { top_k: 0 }, { top_k: true }, { language: "" },
  ]) {
    assert.throws(() => searchRecipes(catalog(), policy(), { available_ingredients: ["egg"], ...patch }));
  }
});

test("catalog statistics, URLs and source evidence cannot silently drift", () => {
  const badStatistics = rawCatalog();
  badStatistics.ingredient_frequency[2] = 2;
  assert.throws(() => prepareCatalog(badStatistics));
  const badTime = rawCatalog();
  badTime.recipes[0].time_evidence = null;
  assert.throws(() => prepareCatalog(badTime));
  assert.throws(() => sourceUrl("javascript:alert(1)"));
  assert.throws(() => sourceUrl("https://user:password@example.org/"));
  const duplicate = rawCatalog();
  duplicate.recipes[1].id = duplicate.recipes[0].id;
  assert.throws(() => prepareCatalog(duplicate));
});
