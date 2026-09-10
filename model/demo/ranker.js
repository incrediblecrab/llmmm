export const FEATURE_NAMES = Object.freeze([
  "overlap_log_fraction", "missing_log_fraction", "recipe_size_log_fraction",
  "pantry_size_log_fraction", "recipe_coverage", "pantry_coverage", "jaccard",
  "idf_recipe_coverage", "idf_pantry_coverage", "idf_jaccard", "missing_fraction",
  "unused_pantry_fraction", "exact_set_match", "fully_available", "time_known",
  "budget_known", "time_log_fraction", "time_budget_fraction", "time_budget_slack",
  "within_time_budget",
]);

function integer(value, name, minimum, maximum) {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new Error(`${name} must be an integer between ${minimum} and ${maximum}.`);
  }
  return value;
}

function minutes(value, name, positive = false) {
  if (typeof value !== "number" || !Number.isFinite(value)
      || value < 0 || (positive && value === 0)) {
    throw new Error(`${name} must be a finite ${positive ? "positive" : "nonnegative"} number.`);
  }
  return value;
}

function ingredientIds(values, size) {
  if (!Array.isArray(values) || !values.length || values.length > 2_000_000) {
    throw new Error("Ingredient IDs must be a bounded, nonempty array.");
  }
  values.forEach((value) => integer(value, "Ingredient ID", 0, size - 1));
  return [...new Set(values)].sort((a, b) => a - b);
}

export function validateStatistics(statistics) {
  integer(statistics.n_recipes, "Recipe count", 1, Number.MAX_SAFE_INTEGER);
  const counts = statistics.ingredient_frequency;
  if (!Array.isArray(counts) || !counts.length || counts.length > 65_536) {
    throw new Error("Document frequencies must be a bounded, nonempty array.");
  }
  counts.forEach((count) => integer(count, "Document frequency", 0, statistics.n_recipes));
}

export function candidateFeatures(availableIds, candidates, statistics, {
  totalMinutes = null, maxTotalMinutes = null,
} = {}) {
  validateStatistics(statistics);
  const counts = statistics.ingredient_frequency;
  const available = ingredientIds(availableIds, counts.length);
  if (!Array.isArray(candidates) || candidates.length > 10_000) {
    throw new Error("At most 10,000 candidates are supported.");
  }
  if (maxTotalMinutes !== null) minutes(maxTotalMinutes, "Time budget", true);
  if (totalMinutes !== null
      && (!Array.isArray(totalMinutes) || totalMinutes.length !== candidates.length)) {
    throw new Error("Source times must align one-to-one with candidates.");
  }
  if (totalMinutes !== null) {
    totalMinutes.forEach((value) => {
      if (value !== null) minutes(value, "Source time");
    });
  }
  const pantry = new Set(available);
  const idf = counts.map((count) => 1 + Math.log1p(statistics.n_recipes) - Math.log1p(count));
  const idfPantry = available.reduce((sum, id) => sum + idf[id], 0);
  const normalizer = Math.log1p(counts.length);
  let slots = availableIds.length;
  return candidates.map((values, index) => {
    const ids = ingredientIds(values, counts.length);
    slots += values.length;
    if (slots > 2_000_000) throw new Error("Ingredient inputs exceed the slot limit.");
    const matched = ids.filter((id) => pantry.has(id));
    const overlap = matched.length;
    const recipeSize = ids.length;
    const pantrySize = available.length;
    const missing = recipeSize - overlap;
    const idfRecipe = ids.reduce((sum, id) => sum + idf[id], 0);
    const idfOverlap = matched.reduce((sum, id) => sum + idf[id], 0);
    const total = totalMinutes === null ? null : totalMinutes[index];
    const known = total !== null;
    const budgetKnown = maxTotalMinutes !== null;
    const bothKnown = known && budgetKnown;
    const timeFraction = bothKnown ? Math.min(total, maxTotalMinutes) / maxTotalMinutes : 0;
    const features = [
      Math.log1p(overlap) / normalizer,
      Math.log1p(missing) / normalizer,
      Math.log1p(recipeSize) / normalizer,
      Math.log1p(pantrySize) / normalizer,
      overlap / recipeSize,
      overlap / pantrySize,
      overlap / (recipeSize + pantrySize - overlap),
      idfOverlap / idfRecipe,
      idfOverlap / idfPantry,
      idfOverlap / (idfRecipe + idfPantry - idfOverlap),
      missing / recipeSize,
      (pantrySize - overlap) / pantrySize,
      Number(overlap === recipeSize && overlap === pantrySize),
      Number(missing === 0),
      Number(known),
      Number(budgetKnown),
      Math.log1p(Math.min(known ? total : 0, 10_080)) / Math.log1p(10_080),
      timeFraction,
      bothKnown ? 1 - timeFraction : 0,
      Number(bothKnown && total <= maxTotalMinutes),
    ];
    return Float32Array.from(features, (value) => Math.min(1, Math.max(0, value)));
  });
}

function featureRows(features) {
  if (!Array.isArray(features) || features.length > 10_000
      || features.some((row) => !(Array.isArray(row) || row instanceof Float32Array)
        || row.length !== FEATURE_NAMES.length
        || row.some((value) => typeof value !== "number" || !Number.isFinite(value)
          || value < 0 || value > 1))) {
    throw new Error("Features must be bounded rows of 20 finite values in [0, 1].");
  }
}

function vector(values, length) {
  return Array.isArray(values) && values.length === length
    && values.every((value) => typeof value === "number" && Number.isFinite(value)
      && Number.isFinite(Math.fround(value)));
}

export function validatePolicy(policy) {
  if (!policy || policy.schema_version !== 1 || policy.feature_version !== 1
      || policy.model_type !== "recipe-ranking-mlp-browser" || policy.activation !== "tanh"
      || typeof policy.time_features_enabled !== "boolean"
      || JSON.stringify(policy.feature_names) !== JSON.stringify(FEATURE_NAMES)) {
    throw new Error("Unsupported browser ranking policy or feature definition.");
  }
  integer(policy.hidden_dim, "Hidden dimension", 1, 256);
  const tensors = policy.tensors;
  const keys = ["feature_mask", "network.0.bias", "network.0.weight",
    "network.2.bias", "network.2.weight"];
  if (!tensors || JSON.stringify(Object.keys(tensors).sort()) !== JSON.stringify(keys)
      || !vector(tensors.feature_mask, 20)
      || tensors.feature_mask.some((value, index) => value
        !== Number(index < 14 || policy.time_features_enabled))
      || !vector(tensors["network.0.bias"], policy.hidden_dim)
      || !Array.isArray(tensors["network.0.weight"])
      || tensors["network.0.weight"].length !== policy.hidden_dim
      || !tensors["network.0.weight"].every((row) => vector(row, 20))
      || !vector(tensors["network.2.bias"], 1)
      || !Array.isArray(tensors["network.2.weight"])
      || tensors["network.2.weight"].length !== 1
      || !vector(tensors["network.2.weight"][0], policy.hidden_dim)) {
    throw new Error("The ranking tensors have invalid shapes, values or feature masks.");
  }
}

export function learnedScores(features, policy) {
  featureRows(features);
  validatePolicy(policy);
  const tensors = policy.tensors;
  const scores = features.map((row) => {
    const hidden = tensors["network.0.weight"].map((weights, index) => {
      const activation = weights.reduce((sum, weight, feature) => sum
        + weight * Math.fround(row[feature]) * tensors.feature_mask[feature],
      tensors["network.0.bias"][index]);
      return Math.fround(Math.tanh(Math.fround(activation)));
    });
    const score = Math.fround(tensors["network.2.weight"][0].reduce(
      (sum, weight, index) => sum + weight * hidden[index], tensors["network.2.bias"][0]));
    if (!Number.isFinite(score)) throw new Error("The ranking policy produced a non-finite score.");
    return score;
  });
  return scores;
}

export function heuristicScores(features) {
  featureRows(features);
  return features.map((row) => combineBaseline(row[9], row[6], row[4], row[5],
    row[10], row[13], row[18]));
}

function combineBaseline(idfJaccard, jaccard, recipeCoverage, pantryCoverage,
  missingFraction, fullyAvailable, timeSlack) {
  return 3 * idfJaccard + 2 * jaccard + 0.25 * recipeCoverage + 0.25 * pantryCoverage
    - 0.5 * missingFraction + 0.1 * fullyAvailable + 0.025 * timeSlack;
}

export function heuristicStatisticScore(overlap, recipeSize, pantrySize,
  idfOverlap, idfRecipe, idfPantry, timeSlack) {
  const f32 = (value) => Math.fround(Math.max(0, Math.min(1, value)));
  return combineBaseline(f32(idfOverlap / (idfRecipe + idfPantry - idfOverlap)),
    f32(overlap / (recipeSize + pantrySize - overlap)), f32(overlap / recipeSize),
    f32(overlap / pantrySize), f32((recipeSize - overlap) / recipeSize),
    Number(overlap === recipeSize), f32(timeSlack));
}
