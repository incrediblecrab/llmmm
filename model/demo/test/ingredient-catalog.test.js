import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { gzipSync } from "node:zlib";
import test from "node:test";
import { ARRAY_FORMAT, prepareIngredientCatalog, searchIngredientCatalog } from "../ingredient-catalog.js";
import { checkedDecompression } from "../ingredient-loader.js";
import { candidateFeatures, heuristicScores, heuristicStatisticScore } from "../ranker.js";
import { policy } from "./fixtures.js";

function fixture() {
  const arrays = {
    ingredients: new Uint16Array([0, 2, 0, 1, 0, 1, 2, 2]),
    lengths: new Uint16Array([2, 2, 3, 1]),
    total_minutes: new Float64Array([10, NaN, 20, 5]),
    servings: new Float64Array([2, 4, 1, NaN]),
    source_codes: new Uint8Array([0, 0, 0, 0]),
    language_codes: new Uint8Array([0, 1, 0, 0]),
    has_source_url: new Uint8Array([1, 0, 1, 0]),
  };
  const metadata = {
    schema_version: 1, format: "llmmm-ingredient-catalog", endianness: "little",
    statistics_scope: "full-canonical-corpus", n_recipes: 4, n_slots: 8,
    vocabulary: ["egg", "milk", "salt", "pepper"], ingredient_frequency: [3, 2, 3, 0],
    source_names: ["fixture"], language_names: ["en", "fr"],
    arrays: Object.fromEntries(Object.entries(ARRAY_FORMAT).map(([name, [file, dtype]]) => [
      name, { file, dtype, count: arrays[name].length, raw_bytes: arrays[name].byteLength },
    ])),
    rows_per_url_shard: 3,
    url_shards: [
      { file: "urls/0000.json.gz", first_id: 0, rows: 3 },
      { file: "urls/0001.json.gz", first_id: 3, rows: 1 },
    ],
  };
  return { metadata, arrays };
}

test("all ingredient-only rows, including unknown-time records, are searched", async () => {
  const { metadata, arrays } = fixture();
  const catalog = prepareIngredientCatalog(metadata, arrays);
  const result = await searchIngredientCatalog(catalog, policy(), {
    available_ingredients: ["egg", "milk", "salt"], max_missing: 0,
  }, "learned", { yieldToEvents: false });
  assert.equal(result.scanned, 4);
  assert.equal(result.feasible_count, 4);
  assert.equal(result.retrieval_truncated, false);
  assert.equal(result.matches[0].id, 2);
  assert.equal(result.matches.find((row) => row.id === 1).total_minutes, null);
  assert.equal(result.matches.find((row) => row.id === 3).servings, null);
  assert.ok(result.matches.every((row) => !("instructions" in row) && !("title" in row)));
});

test("source time, missing, required, excluded, language and serving limits are hard", async () => {
  const { metadata, arrays } = fixture();
  const catalog = prepareIngredientCatalog(metadata, arrays);
  for (const ranking of ["learned", "heuristic"]) {
    const result = await searchIngredientCatalog(catalog, policy(), {
      available_ingredients: ["egg", "salt"], must_use: ["egg"], exclude: ["milk"],
      max_missing: 0, max_total_minutes: 10, min_servings: 2, language: "en",
    }, ranking, { yieldToEvents: false });
    assert.deepEqual(result.matches.map((row) => row.id), [0]);
    const zero = await searchIngredientCatalog(catalog, policy(), {
      available_ingredients: ["egg", "milk", "salt"], max_total_minutes: 0,
    }, ranking, { yieldToEvents: false });
    assert.equal(zero.feasible_count, 0);
  }
});

test("bounded learned scoring preserves exact baseline shortlisting and reports truncation", async () => {
  const { metadata, arrays } = fixture();
  const catalog = prepareIngredientCatalog(metadata, arrays);
  const query = { available_ingredients: ["egg", "milk", "salt"], max_missing: null, top_k: 1 };
  const complete = await searchIngredientCatalog(catalog, policy(), query, "heuristic", { yieldToEvents: false });
  const limited = await searchIngredientCatalog(catalog, policy(), query, "learned",
    { maxCandidates: 1, yieldToEvents: false });
  assert.equal(limited.matches[0].id, complete.matches[0].id);
  assert.equal(limited.scanned, 4);
  assert.equal(limited.feasible_count, 4);
  assert.equal(limited.candidates_scored, 1);
  assert.equal(limited.retrieval_truncated, true);
});

test("source-link availability is an explicit constraint, not a silent exclusion", async () => {
  const { metadata, arrays } = fixture();
  const catalog = prepareIngredientCatalog(metadata, arrays);
  const query = { available_ingredients: ["egg", "milk", "salt"], max_missing: null };
  const result = await searchIngredientCatalog(catalog, policy(), {
    ...query, require_source_url: true,
  }, "learned", { yieldToEvents: false });
  assert.equal(result.feasible_count, 2);
  assert.deepEqual(new Set(result.matches.map((row) => row.id)), new Set([0, 2]));
  await assert.rejects(searchIngredientCatalog(catalog, policy(), {
    ...query, require_source_url: "true",
  }, "learned", { yieldToEvents: false }), /boolean/);
});

test("shortlisting uses the same rounded baseline features, not a separate formula", () => {
  const stats = { n_recipes: 10, ingredient_frequency: [5, 4, 1] };
  const rows = candidateFeatures([0, 1], [[1, 2]], stats,
    { totalMinutes: [7], maxTotalMinutes: 10 });
  const idf = stats.ingredient_frequency.map((n) => 1 + Math.log1p(10) - Math.log1p(n));
  const value = heuristicStatisticScore(1, 2, 2, idf[1], idf[1] + idf[2], idf[0] + idf[1], 0.3);
  assert.equal(value, heuristicScores(rows)[0]);
});

test("array corruption, incomplete coverage and invalid numeric facts fail closed", () => {
  for (const mutate of [
    ({ arrays }) => { arrays.ingredients[1] = 0; },
    ({ arrays }) => { arrays.lengths[0] = 0; },
    ({ arrays }) => { arrays.total_minutes[0] = 0; },
    ({ arrays }) => { arrays.servings[0] = Infinity; },
    ({ arrays }) => { arrays.source_codes[0] = 1; },
    ({ metadata }) => { metadata.ingredient_frequency[0] = 2; },
    ({ metadata }) => { metadata.url_shards[1].first_id = 2; },
    ({ metadata }) => { metadata.arrays.ingredients.file = "../private.bin"; },
  ]) {
    const data = fixture();
    mutate(data);
    assert.throws(() => prepareIngredientCatalog(data.metadata, data.arrays));
  }
});

test("cancelled or invalid searches do not masquerade as successful empty results", async () => {
  const { metadata, arrays } = fixture();
  const catalog = prepareIngredientCatalog(metadata, arrays);
  await assert.rejects(searchIngredientCatalog(catalog, policy(), {
    available_ingredients: ["egg"],
  }, "learned", { cancelled: () => true, yieldToEvents: false }), { name: "AbortError" });
  await assert.rejects(searchIngredientCatalog(catalog, policy(), {
    available_ingredients: ["unknown"],
  }, "learned", { yieldToEvents: false }), /recognized/);
});

test("gzip decoding verifies exact lengths and hashes", async () => {
  const data = Buffer.from("Ingredient-only test fixture");
  const compressed = gzipSync(data);
  const record = { raw_bytes: data.length, raw_sha256: createHash("sha256").update(data).digest("hex") };
  assert.deepEqual(Buffer.from(await checkedDecompression(compressed, record)), data);
  await assert.rejects(checkedDecompression(compressed, { ...record, raw_bytes: data.length - 1 }), /exceeded/);
  await assert.rejects(checkedDecompression(compressed, { ...record, raw_sha256: "0".repeat(64) }), /integrity/);
});
