import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { gzipSync } from "node:zlib";
import test from "node:test";
import { ARRAY_FORMAT, prepareIngredientCatalog, searchIngredientCatalog } from "../ingredient-catalog.js";
import { checkedBytes, checkedDecompression, loadResultLinks, loadResultText, loadTextManifest } from "../ingredient-loader.js";
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
    link_status: new Uint8Array([1, 0, 2, 3]),
  };
  const metadata = {
    schema_version: 3, format: "llmmm-ingredient-catalog", endianness: "little",
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
    link_shards: [
      { file: "links/0000.json.gz", first_id: 0, rows: 3 },
      { file: "links/0001.json.gz", first_id: 3, rows: 1 },
    ],
    rows_per_text_shard: 3,
    text_manifest: { file: "text-shards.json.gz", compression: "gzip", shards: 2 },
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

test("recipe-link availability is an explicit constraint, not a silent exclusion", async () => {
  const { metadata, arrays } = fixture();
  const catalog = prepareIngredientCatalog(metadata, arrays);
  const query = { available_ingredients: ["egg", "milk", "salt"], max_missing: null };
  const result = await searchIngredientCatalog(catalog, policy(), {
    ...query, require_link: true,
  }, "learned", { yieldToEvents: false });
  assert.equal(result.feasible_count, 2);
  assert.deepEqual(new Set(result.matches.map((row) => row.id)), new Set([0, 2]));
  const all = await searchIngredientCatalog(catalog, policy(), query, "learned", { yieldToEvents: false });
  assert.deepEqual(Object.fromEntries(all.matches.map((row) => [row.id, row.link_status])),
    { 0: "source", 1: "none", 2: "archive", 3: "offline" });
  await assert.rejects(searchIngredientCatalog(catalog, policy(), {
    ...query, require_link: "true",
  }, "learned", { yieldToEvents: false }), /boolean/);
  await assert.rejects(searchIngredientCatalog(catalog, policy(), {
    ...query, require_source_url: true,
  }, "learned", { yieldToEvents: false }), /require_link/);
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
    ({ metadata }) => { metadata.link_shards[1].first_id = 2; },
    ({ metadata }) => { metadata.link_shards[0].file = "../private.json.gz"; },
    ({ metadata }) => { metadata.link_shards.pop(); },
    ({ arrays }) => { arrays.link_status[0] = 4; },
    ({ metadata }) => { metadata.arrays.ingredients.file = "../private.bin"; },
    ({ metadata }) => { metadata.schema_version = 2; },
    ({ metadata }) => { metadata.text_manifest.shards = 1; },
    ({ metadata }) => { metadata.text_manifest.file = "../text-shards.json.gz"; },
    ({ metadata }) => { metadata.rows_per_text_shard = 0; },
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

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function gzipJson(value) {
  const raw = Buffer.from(JSON.stringify(value));
  const data = gzipSync(raw);
  return { data, record: { bytes: data.length, sha256: sha256(data), raw_bytes: raw.length, raw_sha256: sha256(raw), compression: "gzip" } };
}

function textFixture({ shard = (value) => value, manifest = (value) => value } = {}) {
  const files = new Map();
  const shards = [
    { first_id: 0, titles: ["Egg & milk custard", null, "Salted eggs"],
      ingredient_lines: [["2 eggs", "1 cup milk"], null, ["3 eggs", "salt to taste"]] },
    { first_id: 3, titles: ["Pepper egg"], ingredient_lines: [["1 egg", "black pepper"]] },
  ];
  const records = shards.map((value, index) => {
    const file = `text/000${index}.json.gz`;
    const { data, record } = gzipJson(shard(structuredClone(value), index));
    files.set(file, data);
    return { file, first_id: value.first_id, rows: value.titles.length, ...record };
  });
  const { data, record } = gzipJson(manifest({ shards: records }));
  files.set("text-shards.json.gz", data);
  return { files, catalog: { n_recipes: 4, rows_per_text_shard: 3, text_manifest: { file: "text-shards.json.gz", shards: 2, ...record } } };
}

async function servingFiles(files, action) {
  const original = globalThis.fetch;
  globalThis.fetch = async (url) => {
    const path = new URL(url).pathname.replace(/^\/index\//, "");
    return files.has(path) ? new Response(files.get(path)) : new Response("missing", { status: 404 });
  };
  try {
    return await action(new URL("https://example.test/index/ingredient-index.json"));
  } finally {
    globalThis.fetch = original;
  }
}

async function resultText(files, catalog, ids) {
  return servingFiles(files, async (indexUrl) => loadResultText(indexUrl, catalog,
    await loadTextManifest(indexUrl, catalog), ids.map((id) => ({ id }))));
}

test("result cards receive their own recorded title and ingredient lines", async () => {
  const { files, catalog } = textFixture();
  assert.deepEqual(await resultText(files, catalog, [2, 3, 1]), [
    { id: 2, title: "Salted eggs", ingredient_lines: ["3 eggs", "salt to taste"] },
    { id: 3, title: "Pepper egg", ingredient_lines: ["1 egg", "black pepper"] },
    { id: 1, title: null, ingredient_lines: null },
  ]);
});

test("misaligned, oversized, smuggled or corrupt recipe text fails closed", async () => {
  const first = (mutate) => (value, index) => { if (index === 0) mutate(value); return value; };
  for (const [options, pattern] of [
    [{ shard: (value) => ({ ...value, first_id: value.first_id + 1 }) }, /aligned/],
    [{ shard: (value) => ({ ...value, instructions: ["Whisk."] }) }, /aligned/],
    [{ shard: first((value) => { value.titles.pop(); }) }, /aligned/],
    [{ shard: first((value) => { value.ingredient_lines[2] = ["salt\u0007"]; }) }, /invalid/],
    [{ shard: first((value) => { value.titles[2] = "x".repeat(16_385); }) }, /invalid/],
    [{ shard: first((value) => { value.ingredient_lines[2] = []; }) }, /invalid/],
    [{ shard: first((value) => { value.ingredient_lines[2] = "3 eggs"; }) }, /invalid/],
    [{ manifest: (value) => { value.shards[1].file = "../private.json.gz"; return value; } }, /contiguous/],
    [{ manifest: (value) => ({ shards: value.shards.slice(0, 1) }) }, /structure/],
  ]) {
    const { files, catalog } = textFixture(options);
    await assert.rejects(resultText(files, catalog, [2, 3]), pattern);
  }
  const { files, catalog } = textFixture();
  files.set("text/0001.json.gz", gzipSync(Buffer.from('{"first_id":3}')));
  await assert.rejects(resultText(files, catalog, [3]), /integrity|exceeded/);
  files.delete("text/0001.json.gz");
  await assert.rejects(resultText(files, catalog, [3]), /HTTP 404/);
});

function linkFixture(links) {
  const { metadata, arrays } = fixture();
  const files = new Map();
  links.forEach((values, index) => {
    const { data, record } = gzipJson({ first_id: metadata.link_shards[index].first_id, links: values });
    files.set(metadata.link_shards[index].file, data);
    Object.assign(metadata.link_shards[index], record);
  });
  return { files, catalog: prepareIngredientCatalog(metadata, arrays) };
}

async function resultLinks(links, ids) {
  const { files, catalog } = linkFixture(links);
  return servingFiles(files, (indexUrl) => loadResultLinks(indexUrl, catalog, ids.map((id) => ({ id }))));
}

const ARCHIVED = "https://web.archive.org/web/2015/http://example.test/recipe/2";

test("result cards receive the link their status declares", async () => {
  assert.deepEqual(await resultLinks([["https://example.test/recipe/0", null, ARCHIVED], [null]], [2, 0, 3, 1]), [
    { id: 2, link: ARCHIVED }, { id: 0, link: "https://example.test/recipe/0" }, { id: 3, link: null }, { id: 1, link: null },
  ]);
});

test("links that contradict their status or are not plain HTTPS fail closed", async () => {
  for (const [links, pattern] of [
    [[[null, null, ARCHIVED], [null]], /link status/],
    [[["https://example.test/recipe/0", "https://example.test/recipe/1", ARCHIVED], [null]], /link status/],
    [[["https://example.test/recipe/0", null, ARCHIVED], ["https://example.test/recipe/3"]], /link status/],
    [[["http://example.test/recipe/0", null, ARCHIVED], [null]], /HTTPS/],
    [[["https://user:secret@example.test/recipe/0", null, ARCHIVED], [null]], /HTTPS/],
    [[["https://example.test/recipe/0", null], [null]], /aligned/],
  ]) {
    await assert.rejects(resultLinks(links, [0, 1, 2, 3]), pattern);
  }
});

test("a network failure is retried a bounded number of times; HTTP and checksum failures are not", async () => {
  const payload = Buffer.from("verified bytes");
  const expected = { bytes: payload.length, sha256: createHash("sha256").update(payload).digest("hex") };
  const url = new URL("https://example.test/index/text/0000.json.gz");
  const original = globalThis.fetch;
  let calls = 0;
  const serve = (responses) => {
    calls = 0;
    globalThis.fetch = async () => {
      const next = responses[Math.min(calls++, responses.length - 1)];
      if (next instanceof Error) throw next;
      return next();
    };
  };
  const options = { retryDelays: [0, 0] };
  try {
    serve([new TypeError("Failed to fetch"), () => new Response(payload)]);
    assert.deepEqual(Buffer.from(await checkedBytes(url, expected, options)), payload);
    assert.equal(calls, 2);
    serve([new TypeError("Failed to fetch")]);
    await assert.rejects(checkedBytes(url, expected, options),
      /Download failed after 3 attempts \(network error\): \/index\/text\/0000\.json\.gz/);
    assert.equal(calls, 3);
    serve([() => new Response("verified bytez")]);
    await assert.rejects(checkedBytes(url, expected, options), /integrity check/);
    assert.equal(calls, 1);
    serve([() => new Response("busy", { status: 503 })]);
    await assert.rejects(checkedBytes(url, expected, options), /HTTP 503/);
    assert.equal(calls, 1);
  } finally {
    globalThis.fetch = original;
  }
});
