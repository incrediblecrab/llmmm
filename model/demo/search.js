import { candidateFeatures, heuristicScores, learnedScores, validateStatistics } from "./ranker.js";

export function normalizeIngredient(value) {
  return value.trim().toLowerCase().replaceAll(" ", "_");
}

function stringList(values, { nonempty = false } = {}) {
  return Array.isArray(values) && (!nonempty || values.length > 0)
    && values.every((value) => typeof value === "string" && value.trim().length > 0);
}

export function sourceUrl(value) {
  const url = new URL(value);
  if (url.protocol !== "https:" || url.username || url.password) {
    throw new Error("Recipe source links must be ordinary HTTPS URLs.");
  }
  return url.href;
}

export function prepareCatalog(catalog) {
  if (!catalog || catalog.schema_version !== 1 || catalog.statistics_scope !== "public-sample"
      || !stringList(catalog.vocabulary, { nonempty: true })
      || new Set(catalog.vocabulary).size !== catalog.vocabulary.length
      || catalog.vocabulary.some((name) => normalizeIngredient(name) !== name)
      || !Array.isArray(catalog.recipes) || catalog.recipes.length > 10_000
      || catalog.recipes.length !== catalog.n_recipes) {
    throw new Error("The public catalog has an unsupported or inconsistent schema.");
  }
  validateStatistics(catalog);
  if (catalog.ingredient_frequency.length !== catalog.vocabulary.length) {
    throw new Error("Document frequencies do not match the vocabulary.");
  }
  const index = new Map(catalog.vocabulary.map((name, id) => [name, id]));
  const frequency = new Array(catalog.vocabulary.length).fill(0);
  const seen = new Set();
  const recipes = catalog.recipes.map((recipe, position) => {
    if (!recipe || typeof recipe.id !== "string" || !recipe.id || seen.has(recipe.id)
        || typeof recipe.title !== "string" || !recipe.title.trim()
        || typeof recipe.language !== "string" || !recipe.language.trim()
        || !stringList(recipe.canonical_ingredients, { nonempty: true })
        || new Set(recipe.canonical_ingredients).size !== recipe.canonical_ingredients.length
        || !stringList(recipe.raw_ingredients, { nonempty: true })
        || !stringList(recipe.instructions, { nonempty: true })
        || !stringList(recipe.unmapped_ingredients)
        || (recipe.source_limitations !== undefined && !stringList(recipe.source_limitations))
        || typeof recipe.attribution !== "string" || !recipe.attribution.trim()
        || typeof recipe.license !== "string" || !recipe.license.trim()) {
      throw new Error(`Public recipe row ${position} has missing, duplicate or malformed fields.`);
    }
    for (const name of ["total_minutes", "servings"]) {
      if (recipe[name] !== null && (typeof recipe[name] !== "number"
          || !Number.isFinite(recipe[name]) || recipe[name] <= 0)) {
        throw new Error(`${recipe.id}: ${name} must be positive or explicitly unknown.`);
      }
    }
    if ((recipe.total_minutes !== null && !recipe.time_evidence)
        || (recipe.servings !== null && !recipe.servings_evidence)) {
      throw new Error(`${recipe.id}: reported metadata needs source evidence.`);
    }
    sourceUrl(recipe.source_url);
    sourceUrl(recipe.attribution_url);
    const ids = recipe.canonical_ingredients.map((name) => {
      if (!index.has(name)) throw new Error(`${recipe.id}: unknown canonical ingredient ${name}.`);
      return index.get(name);
    }).sort((a, b) => a - b);
    ids.forEach((id) => { frequency[id] += 1; });
    seen.add(recipe.id);
    return { ...recipe, ingredient_ids: ids, position,
      source_limitations: recipe.source_limitations === undefined ? [] : recipe.source_limitations };
  });
  if (frequency.some((count, index) => count !== catalog.ingredient_frequency[index])) {
    throw new Error("Stored statistics do not describe this exact public sample.");
  }
  return { ...catalog, recipes, index };
}

function names(values, label, catalog) {
  if (!Array.isArray(values) || values.length > 100
      || values.some((value) => typeof value !== "string")) {
    throw new Error(`${label} must contain at most 100 ingredient names.`);
  }
  return new Set(values.map((value) => {
    const normalized = normalizeIngredient(value);
    if (!catalog.index.has(normalized)) {
      throw new Error(`${label}: "${value}" is not a recognized ingredient. Choose a suggested name.`);
    }
    return catalog.index.get(normalized);
  }));
}

function optionalNumber(value, label, minimum) {
  if (value !== null && (typeof value !== "number" || !Number.isFinite(value) || value < minimum)) {
    throw new Error(`${label} must be a finite number of at least ${minimum}, or left blank.`);
  }
  return value;
}

export function normalizeQuery(query, catalog) {
  const available = names(query.available_ingredients, "Pantry", catalog);
  const required = names(query.must_use ?? [], "Required ingredients", catalog);
  const excluded = names(query.exclude ?? [], "Excluded ingredients", catalog);
  if (!available.size) throw new Error("Add at least one ingredient to your pantry.");
  if ([...required].some((id) => excluded.has(id))) {
    throw new Error("An ingredient cannot be both required and excluded.");
  }
  const searchable = new Set([...available].filter((id) => !excluded.has(id)).concat([...required]));
  if (!searchable.size) throw new Error("No searchable ingredients remain after exclusions.");
  const topK = query.top_k === undefined ? 5 : query.top_k;
  if (!Number.isInteger(topK) || topK < 1 || topK > 100) {
    throw new Error("The result limit must be an integer between 1 and 100.");
  }
  const maxMissing = query.max_missing === undefined ? 2 : query.max_missing;
  if (maxMissing !== null && (!Number.isInteger(maxMissing) || maxMissing < 0)) {
    throw new Error("Missing ingredients must be a nonnegative integer, or unlimited.");
  }
  const budget = optionalNumber(query.max_total_minutes ?? null, "Total time limit", 0);
  const servings = optionalNumber(query.min_servings ?? null, "Minimum reported servings", 1);
  const language = query.language ?? null;
  if (language !== null && (typeof language !== "string" || !language.trim() || language.length > 32)) {
    throw new Error("Language must be a nonempty code of at most 32 characters.");
  }
  return { available, required, excluded, searchable, topK, maxMissing, budget, servings, language };
}

export function searchRecipes(catalog, policy, query, ranking = "learned") {
  if (!["learned", "heuristic"].includes(ranking)) throw new Error("Unknown ranking method.");
  const request = normalizeQuery(query, catalog);
  const candidates = catalog.recipes.filter((recipe) => {
    const ids = new Set(recipe.ingredient_ids);
    return [...request.searchable].some((id) => ids.has(id))
      && [...request.required].every((id) => ids.has(id))
      && [...request.excluded].every((id) => !ids.has(id))
      && (request.budget === null
        || (recipe.total_minutes !== null && recipe.total_minutes <= request.budget))
      && (request.servings === null
        || (recipe.servings !== null && recipe.servings >= request.servings))
      && (request.language === null || recipe.language === request.language)
      && (request.maxMissing === null
        || recipe.ingredient_ids.filter((id) => !request.available.has(id)).length <= request.maxMissing);
  });
  if (!candidates.length) return { matches: [], feasible_count: 0, scanned: catalog.n_recipes, ranking };
  const features = candidateFeatures(
    [...request.available], candidates.map((recipe) => recipe.ingredient_ids), catalog,
    { totalMinutes: candidates.map((recipe) => recipe.total_minutes), maxTotalMinutes: request.budget },
  );
  const scores = ranking === "learned" ? learnedScores(features, policy) : heuristicScores(features);
  const matches = candidates.map((recipe, index) => ({
    ...recipe,
    score: scores[index],
    matched_ingredients: recipe.ingredient_ids.filter((id) => request.available.has(id))
      .map((id) => catalog.vocabulary[id]),
    missing_ingredients: recipe.ingredient_ids.filter((id) => !request.available.has(id))
      .map((id) => catalog.vocabulary[id]),
  })).sort((a, b) => b.score - a.score || a.position - b.position).slice(0, request.topK);
  return { matches, feasible_count: candidates.length, scanned: catalog.n_recipes, ranking };
}
