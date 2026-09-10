import {
  candidateFeatures, heuristicScores, heuristicStatisticScore, learnedScores,
  validatePolicy, validateStatistics,
} from "./ranker.js";
import { normalizeIngredient, normalizeQuery } from "./search.js";

export const ARRAY_FORMAT = Object.freeze({
  ingredients: ["ingredients.u16.gz", "<u2", Uint16Array],
  lengths: ["lengths.u16.gz", "<u2", Uint16Array],
  total_minutes: ["total-minutes.f64.gz", "<f8", Float64Array],
  servings: ["servings.f64.gz", "<f8", Float64Array],
  source_codes: ["sources.u8.gz", "|u1", Uint8Array],
  language_codes: ["languages.u8.gz", "|u1", Uint8Array],
  has_source_url: ["source-links.u8.gz", "|u1", Uint8Array],
});

function strings(values, maximum) {
  return Array.isArray(values) && values.length > 0 && values.length <= maximum
    && new Set(values).size === values.length
    && values.every((value) => typeof value === "string" && value.trim() && value.length <= 128);
}

export function ingredientCatalogMetadata(metadata) {
  if (!metadata || metadata.schema_version !== 1 || metadata.format !== "llmmm-ingredient-catalog"
      || metadata.endianness !== "little" || metadata.statistics_scope !== "full-canonical-corpus"
      || !Number.isSafeInteger(metadata.n_recipes) || metadata.n_recipes > 10_000_000
      || !Number.isSafeInteger(metadata.n_slots) || metadata.n_slots < metadata.n_recipes
      || metadata.n_slots > 500_000_000
      || !strings(metadata.vocabulary, 65_536)
      || metadata.vocabulary.some((name) => normalizeIngredient(name) !== name)
      || !strings(metadata.source_names, 256) || !strings(metadata.language_names, 256)
      || metadata.ingredient_frequency?.length !== metadata.vocabulary.length
      || !metadata.arrays || JSON.stringify(Object.keys(metadata.arrays).sort())
        !== JSON.stringify(Object.keys(ARRAY_FORMAT).sort())) {
    throw new Error("Unsupported ingredient-only catalog schema, counts or vocabulary.");
  }
  validateStatistics(metadata);
  for (const [name, [filename, dtype, Type]] of Object.entries(ARRAY_FORMAT)) {
    const record = metadata.arrays[name];
    const count = name === "ingredients" ? metadata.n_slots : metadata.n_recipes;
    if (!record || record.file !== filename || record.dtype !== dtype || record.count !== count
        || record.raw_bytes !== count * Type.BYTES_PER_ELEMENT) {
      throw new Error(`Ingredient-only ${name} has an invalid path, dtype or array shape.`);
    }
  }
  const shardSize = metadata.rows_per_url_shard;
  if (!Number.isSafeInteger(shardSize) || shardSize < 1 || shardSize > 65_536
      || !Array.isArray(metadata.url_shards)
      || metadata.url_shards.length !== Math.ceil(metadata.n_recipes / shardSize)) {
    throw new Error("The source URL shard inventory is incomplete.");
  }
  metadata.url_shards.forEach((shard, index) => {
    const first = index * shardSize;
    if (shard.file !== `urls/${String(index).padStart(4, "0")}.json.gz`
        || shard.first_id !== first || shard.rows !== Math.min(shardSize, metadata.n_recipes - first)) {
      throw new Error("Source URL shards are not contiguous, complete and safely named.");
    }
  });
  return { ...metadata, index: new Map(metadata.vocabulary.map((name, id) => [name, id])) };
}

export function prepareIngredientCatalog(metadata, arrays) {
  const catalog = ingredientCatalogMetadata(metadata);
  for (const [name, [, , Type]] of Object.entries(ARRAY_FORMAT)) {
    if (!(arrays[name] instanceof Type) || arrays[name].length !== catalog.arrays[name].count) {
      throw new Error(`Ingredient-only ${name} did not decode to the declared array.`);
    }
  }
  const offsets = new Uint32Array(catalog.n_recipes + 1);
  const counts = new Uint32Array(catalog.vocabulary.length);
  let position = 0;
  for (let id = 0; id < catalog.n_recipes; id += 1) {
    const length = arrays.lengths[id];
    if (!length || length > catalog.vocabulary.length || position + length > catalog.n_slots
        || arrays.source_codes[id] >= catalog.source_names.length
        || arrays.language_codes[id] >= catalog.language_names.length
        || arrays.has_source_url[id] > 1) {
      throw new Error(`Ingredient-only record ${id} has invalid lengths or metadata codes.`);
    }
    for (const name of ["total_minutes", "servings"]) {
      const value = arrays[name][id];
      if (!Number.isNaN(value) && (!Number.isFinite(value) || value <= 0)) {
        throw new Error(`Ingredient-only record ${id} has invalid source ${name}.`);
      }
    }
    let previous = -1;
    for (let slot = 0; slot < length; slot += 1) {
      const ingredient = arrays.ingredients[position + slot];
      if (ingredient <= previous || ingredient >= catalog.vocabulary.length) {
        throw new Error(`Ingredient-only record ${id} is not a sorted unique canonical set.`);
      }
      previous = ingredient;
      counts[ingredient] += 1;
    }
    position += length;
    offsets[id + 1] = position;
  }
  if (position !== catalog.n_slots
      || counts.some((value, index) => value !== catalog.ingredient_frequency[index])) {
    throw new Error("Ingredient arrays do not cover the complete declared corpus/frequencies.");
  }
  return {
    ...catalog, data: arrays, offsets,
    idf: Float64Array.from(catalog.ingredient_frequency,
      (count) => 1 + Math.log1p(catalog.n_recipes) - Math.log1p(count)),
  };
}

function better(a, b) {
  return a.priority > b.priority || (a.priority === b.priority && a.id < b.id);
}

class Shortlist {
  constructor(limit) { this.limit = limit; this.values = []; }

  add(id, priority) {
    const values = this.values;
    const candidate = { id, priority };
    if (values.length === this.limit) {
      if (!better(candidate, values[0])) return;
      values[0] = candidate;
      let position = 0;
      while (position * 2 + 1 < values.length) {
        let child = position * 2 + 1;
        if (child + 1 < values.length && better(values[child], values[child + 1])) child += 1;
        if (!better(values[position], values[child])) break;
        [values[position], values[child]] = [values[child], values[position]];
        position = child;
      }
    } else {
      values.push(candidate);
      let position = values.length - 1;
      while (position > 0) {
        const parent = Math.floor((position - 1) / 2);
        if (!better(values[parent], values[position])) break;
        [values[parent], values[position]] = [values[position], values[parent]];
        position = parent;
      }
    }
  }
}

export async function searchIngredientCatalog(catalog, policy, query, ranking = "learned", {
  maxCandidates = 2000, onProgress = null, cancelled = () => false, yieldToEvents = true,
} = {}) {
  const start = performance.now();
  if (!["learned", "heuristic"].includes(ranking)) throw new Error("Unknown ranking method.");
  validatePolicy(policy);
  const request = normalizeQuery(query, catalog);
  if (query.require_source_url !== undefined && typeof query.require_source_url !== "boolean") {
    throw new Error("The source-link filter must be a boolean.");
  }
  if (!Number.isSafeInteger(maxCandidates) || maxCandidates < request.topK || maxCandidates > 10_000) {
    throw new Error("The shortlist limit must be an integer from the result count to 10,000.");
  }
  const membership = new Uint8Array(catalog.vocabulary.length);
  const required = new Uint8Array(catalog.vocabulary.length);
  const excluded = new Uint8Array(catalog.vocabulary.length);
  const searchable = new Uint8Array(catalog.vocabulary.length);
  request.available.forEach((id) => { membership[id] = 1; });
  request.required.forEach((id) => { required[id] = 1; });
  request.excluded.forEach((id) => { excluded[id] = 1; });
  request.searchable.forEach((id) => { searchable[id] = 1; });
  const idfPantry = [...request.available].sort((a, b) => a - b)
    .reduce((sum, id) => sum + catalog.idf[id], 0);
  const languageCode = request.language === null ? null : catalog.language_names.indexOf(request.language);
  const shortlist = new Shortlist(maxCandidates);
  let feasible = 0;
  const { ingredients, lengths, total_minutes: totals, servings, language_codes: languages } = catalog.data;
  for (let id = 0; id < catalog.n_recipes; id += 1) {
    if (id % 262_144 === 0) {
      if (cancelled()) throw new DOMException("Superseded by a newer search.", "AbortError");
      if (onProgress) onProgress({ scanned: id, total: catalog.n_recipes });
      if (yieldToEvents) await new Promise((resolve) => setTimeout(resolve, 0));
    }
    const total = totals[id];
    if ((query.require_source_url && !catalog.data.has_source_url[id])
        || (request.budget !== null && !(total <= request.budget))
        || (request.servings !== null && !(servings[id] >= request.servings))
        || (languageCode !== null && languages[id] !== languageCode)
        || (request.maxMissing !== null && lengths[id] > request.available.size + request.maxMissing)) continue;
    let overlap = 0; let requiredFound = 0; let excludedFound = 0; let searchableFound = 0;
    let idfOverlap = 0; let idfRecipe = 0;
    for (let slot = catalog.offsets[id]; slot < catalog.offsets[id + 1]; slot += 1) {
      const ingredient = ingredients[slot];
      const present = membership[ingredient];
      overlap += present;
      requiredFound += required[ingredient];
      excludedFound += excluded[ingredient];
      searchableFound += searchable[ingredient];
      idfRecipe += catalog.idf[ingredient];
      idfOverlap += present * catalog.idf[ingredient];
    }
    if (!searchableFound || excludedFound || requiredFound !== request.required.size
        || (request.maxMissing !== null && lengths[id] - overlap > request.maxMissing)) continue;
    feasible += 1;
    const slack = request.budget !== null && Number.isFinite(total) ? 1 - total / request.budget : 0;
    const priority = heuristicStatisticScore(overlap, lengths[id], request.available.size,
      idfOverlap, idfRecipe, idfPantry, slack);
    shortlist.add(id, priority);
  }
  if (cancelled()) throw new DOMException("Superseded by a newer search.", "AbortError");
  const ordered = shortlist.values.sort((a, b) => a.id - b.id);
  const ids = ordered.map((row) => row.id);
  const candidates = ids.map((id) => Array.from(ingredients.subarray(catalog.offsets[id], catalog.offsets[id + 1])));
  let scores = [];
  if (ids.length) {
    const features = candidateFeatures([...request.available], candidates, catalog, {
      totalMinutes: ids.map((id) => Number.isNaN(totals[id]) ? null : totals[id]),
      maxTotalMinutes: request.budget,
    });
    scores = ranking === "learned" ? learnedScores(features, policy) : heuristicScores(features);
  }
  const matches = ids.map((id, position) => ({
    id, score: scores[position], canonical_ingredients: candidates[position].map((value) => catalog.vocabulary[value]),
    matched_ingredients: candidates[position].filter((value) => membership[value]).map((value) => catalog.vocabulary[value]),
    missing_ingredients: candidates[position].filter((value) => !membership[value]).map((value) => catalog.vocabulary[value]),
    total_minutes: Number.isNaN(totals[id]) ? null : totals[id],
    servings: Number.isNaN(servings[id]) ? null : servings[id],
    source: catalog.source_names[catalog.data.source_codes[id]],
    language: catalog.language_names[languages[id]],
  })).sort((a, b) => b.score - a.score || a.id - b.id).slice(0, request.topK);
  return {
    matches, feasible_count: feasible, scanned: catalog.n_recipes, ranking,
    candidates_scored: ids.length, retrieval_truncated: feasible > maxCandidates,
    shortlist_method: "deterministic-baseline-top-k",
    elapsed_ms: performance.now() - start,
  };
}
