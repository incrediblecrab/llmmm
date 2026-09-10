import { validatePolicy } from "./ranker.js";
import { normalizeIngredient, prepareCatalog, searchRecipes, sourceUrl } from "./search.js";

const $ = (id) => document.getElementById(id);
const selected = new Set();
let catalog;
let policy;
let topK = 5;

function element(tag, text = "", className = "") {
  const node = document.createElement(tag);
  if (text) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function link(text, url) {
  const node = element("a", text);
  node.href = sourceUrl(url);
  node.target = "_blank";
  node.rel = "noopener noreferrer";
  return node;
}

function displayName(name) {
  return name.replaceAll("_", " ");
}

function tokens(value) {
  return value.split(",").map((name) => name.trim()).filter(Boolean);
}

async function loadJson(filename, expected = null) {
  const response = await fetch(new URL(filename, import.meta.url), {
    credentials: "omit", cache: "no-store", signal: AbortSignal.timeout(20_000),
  });
  if (!response.ok) throw new Error(`Could not load ${filename} (HTTP ${response.status}).`);
  const bytes = await response.arrayBuffer();
  if (expected) {
    const digest = await crypto.subtle.digest("SHA-256", bytes);
    const actual = [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, "0")).join("");
    if (bytes.byteLength !== expected.bytes || actual !== expected.sha256) {
      throw new Error(`${filename} failed its integrity check. Reload to fetch a consistent release.`);
    }
  }
  return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
}

function renderPantry() {
  $("pantry-list").replaceChildren(...[...selected].map((name) => {
    const item = element("li");
    const button = element("button", "", "chip");
    button.type = "button";
    button.setAttribute("aria-label", `Remove ${displayName(name)}`);
    const mark = element("span", "\u00d7", "remove-mark");
    mark.setAttribute("aria-hidden", "true");
    button.append(element("span", displayName(name)), mark);
    button.addEventListener("click", () => {
      selected.delete(name);
      renderPantry();
      topK = 5;
      runSearch();
    });
    item.append(button);
    return item;
  }));
}

function addPendingIngredients() {
  const additions = tokens($("pantry-entry").value).map((value) => {
    const name = normalizeIngredient(value);
    if (!catalog.index.has(name)) {
      throw new Error(`"${value}" is not a recognized ingredient. Choose a suggested name.`);
    }
    return name;
  });
  if (new Set([...selected, ...additions]).size > 100) {
    throw new Error("The pantry is limited to 100 distinct ingredients.");
  }
  additions.forEach((name) => selected.add(name));
  $("pantry-entry").value = "";
  renderPantry();
}

function optionalInput(id) {
  const input = $(id);
  if (input.validity.badInput) {
    throw new Error("Time and servings must be valid numbers, or left blank.");
  }
  return input.value === "" ? null : input.valueAsNumber;
}

function ingredientLine(label, ingredients, className, empty) {
  const row = element("div", "", `ingredient-line ${className}`);
  row.append(element("span", label, "ingredient-label"),
    element("span", ingredients.length ? ingredients.map(displayName).join(", ") : empty));
  return row;
}

function recipeCard(recipe, index) {
  const article = element("article", "", "recipe-card");
  article.dataset.recipeId = recipe.id;
  const titleRow = element("div", "", "recipe-topline");
  const title = element("h3", "", "recipe-title");
  title.append(link(recipe.title, recipe.source_url));
  titleRow.append(element("span", String(index + 1).padStart(2, "0"), "recipe-number"), title);
  const time = recipe.total_minutes === null ? "Total time not reported"
    : `Source total: ${recipe.total_minutes} min`;
  const servings = recipe.servings === null ? "Servings not reported"
    : `Source servings: ${recipe.servings}`;
  article.append(titleRow, element("p", `${time} / ${servings}`, "recipe-meta"),
    ingredientLine("Have", recipe.matched_ingredients, "have", "No pantry ingredients matched"),
    ingredientLine("Need", recipe.missing_ingredients,
      recipe.missing_ingredients.length ? "need" : "have", "No other canonical ingredients"));
  if (recipe.unmapped_ingredients.length) {
    article.append(element("p", `Not covered by the ingredient filters: ${
      recipe.unmapped_ingredients.join("; ")}. Read the full list below.`, "recipe-warning"));
  }
  const details = element("details", "", "recipe-details");
  details.append(element("summary", "Ingredients and instructions"));
  if (recipe.source_limitations.length) {
    const notes = element("div", "", "source-notes");
    notes.append(element("h4", "Source notes"),
      ...recipe.source_limitations.map((text) => element("p", text)));
    details.append(notes);
  }
  details.append(element("h4", "Ingredients"));
  const ingredients = element("ul");
  ingredients.append(...recipe.raw_ingredients.map((text) => element("li", text)));
  const instructions = element("ol");
  instructions.append(...recipe.instructions.map((text) => element("li", text)));
  details.append(ingredients, element("h4", "Instructions"), instructions);
  const attribution = element("p", `${recipe.attribution}. `, "attribution");
  attribution.append(link("Source revision", recipe.source_url), document.createTextNode(" / "),
    link("Contributor history", recipe.attribution_url),
    document.createTextNode(` / ${recipe.license.toUpperCase()}. ${recipe.changes}`));
  details.append(attribution);
  article.append(details);
  return article;
}

function showEmpty(title, explanation) {
  const box = element("div", "", "empty-state");
  box.append(element("strong", title), element("span", explanation));
  $("results").replaceChildren(box);
}

function runSearch() {
  $("query-error").hidden = true;
  $("show-all").hidden = true;
  try {
    addPendingIngredients();
    const query = {
      available_ingredients: [...selected],
      must_use: tokens($("must-use").value),
      exclude: tokens($("exclude").value),
      max_total_minutes: optionalInput("max-time"),
      min_servings: optionalInput("min-servings"),
      max_missing: $("max-missing").value === "" ? null : Number($("max-missing").value),
      top_k: topK,
    };
    const result = searchRecipes(catalog, policy, query, $("ranking").value);
    $("results").dataset.ranking = result.ranking;
    $("result-count").textContent = `${result.feasible_count} match${
      result.feasible_count === 1 ? "" : "es"} in ${catalog.n_recipes} sample recipes${
      result.matches.length < result.feasible_count ? ` / showing ${result.matches.length}` : ""}`;
    const absent = [...selected].filter((name) => catalog.ingredient_frequency[catalog.index.get(name)] === 0);
    $("sample-warning").hidden = !absent.length;
    $("sample-warning").textContent = `Not represented in this sample: ${absent.map(displayName).join(", ")}.`;
    if (result.matches.length) {
      $("results").replaceChildren(...result.matches.map(recipeCard));
      $("show-all").hidden = result.feasible_count <= result.matches.length || topK === 100;
      $("show-all").textContent = result.feasible_count <= 100
        ? "Show all matches" : "Show the first 100 matches";
    } else {
      showEmpty("No sample recipes fit.", "The filters have not been relaxed. Try a wider pantry, "
        + "allow more missing ingredients, or remove a limit. This small sample cannot cover every meal.");
    }
  } catch (error) {
    $("query-error").textContent = error.message;
    $("query-error").hidden = false;
    $("result-count").textContent = "Search not run";
    $("sample-warning").hidden = true;
    showEmpty("Check your search.", "The message beside the search controls explains what needs changing.");
  }
}

function useExample(example) {
  selected.clear();
  example.available_ingredients.forEach((name) => selected.add(name));
  $("pantry-entry").value = "";
  $("must-use").value = "";
  $("exclude").value = "";
  $("max-time").value = example.max_total_minutes ?? "";
  $("min-servings").value = "";
  $("max-missing").value = String(example.max_missing);
  topK = 5;
  runSearch();
}

async function start() {
  const manifest = await loadJson("manifest.json");
  if (manifest.schema_version !== 1 || !manifest.files?.["catalog.json"] || !manifest.files?.["policy.json"]) {
    throw new Error("The release manifest is missing the public catalog or policy.");
  }
  const [bundle, model] = await Promise.all([
    loadJson("catalog.json", manifest.files["catalog.json"]),
    loadJson("policy.json", manifest.files["policy.json"]),
  ]);
  catalog = prepareCatalog(bundle.catalog);
  validatePolicy(model);
  policy = model;
  if (!Array.isArray(bundle.examples) || !bundle.examples.length) {
    throw new Error("The demo release contains no usable example pantries.");
  }
  $("sample-size").textContent = `${catalog.n_recipes} public recipes / Browser only`;
  const provenance = bundle.provenance;
  $("model-link").href = sourceUrl(`https://huggingface.co/${provenance.model_repository}/tree/${provenance.model_revision}`);
  $("dataset-link").href = sourceUrl(`https://huggingface.co/datasets/${provenance.dataset_repository}${
    provenance.dataset_revision ? `/tree/${provenance.dataset_revision}` : ""}`);
  if (provenance.source_revision) {
    $("code-link").href = sourceUrl(`https://github.com/incrediblecrab/llmmm/tree/${provenance.source_revision}/model/demo`);
  }
  $("license-note").textContent = `Recipe text: ${[...new Set(catalog.recipes.map((row) => row.license.toUpperCase()))].join(", ")}; `
    + "Wikibooks contributors. See each recipe and the dataset for attribution and changes. "
    + "The ranking weights retain their separate model terms.";
  const orderedNames = [...catalog.vocabulary].sort((a, b) =>
    catalog.ingredient_frequency[catalog.index.get(b)] - catalog.ingredient_frequency[catalog.index.get(a)]
    || a.localeCompare(b));
  $("ingredient-names").replaceChildren(...orderedNames.map((name) => {
    const option = element("option");
    option.value = displayName(name);
    return option;
  }));
  $("examples").replaceChildren(...bundle.examples.map((example) => {
    const button = element("button", example.label, "example-button");
    button.type = "button";
    button.addEventListener("click", () => useExample(example));
    return button;
  }));
  $("recipe-query").addEventListener("submit", (event) => {
    event.preventDefault();
    topK = 5;
    runSearch();
  });
  $("add-ingredient").addEventListener("click", () => { topK = 5; runSearch(); });
  $("pantry-entry").addEventListener("keydown", (event) => {
    if (event.key === "Enter") { event.preventDefault(); topK = 5; runSearch(); }
  });
  $("clear-pantry").addEventListener("click", () => {
    selected.clear();
    $("pantry-entry").value = "";
    $("must-use").value = "";
    $("exclude").value = "";
    runSearch();
    $("pantry-entry").focus();
  });
  for (const id of ["max-time", "max-missing", "min-servings", "must-use", "exclude", "ranking"]) {
    $(id).addEventListener("change", () => { topK = 5; runSearch(); });
  }
  $("show-all").addEventListener("click", () => { topK = 100; runSearch(); });
  $("controls").disabled = false;
  useExample(bundle.examples[0]);
}

start().catch((error) => {
  console.error("llmmm demo could not start:", error);
  $("fatal-error").textContent = `The demo could not start: ${error.message} `
    + "Please reload. No fallback recommendations have been substituted.";
  $("fatal-error").hidden = false;
  $("controls").disabled = true;
  $("result-count").textContent = "Demo unavailable";
  showEmpty("Unable to load this release.", "The public model and dataset can still be accessed using the links below.");
});
