import { validatePolicy } from "./ranker.js";
import { normalizeIngredient, prepareCatalog, searchRecipes, sourceUrl } from "./search.js";
import { IngredientBrowserClient } from "./ingredient-client.js";
import { ingredientCatalogMetadata } from "./ingredient-catalog.js";

const $ = (id) => document.getElementById(id);
const selected = new Set();
let catalog;
let policy;
let topK = 5;
let ingredientMode = false;
let ingredientClient;
let searchSequence = 0;

function element(tag, text = "", className = "") {
  const node = document.createElement(tag);
  if (text) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function link(text, url, allowHttp = false) {
  const node = element("a", text);
  if (allowHttp) {
    const parsed = new URL(url);
    if (!["http:", "https:"].includes(parsed.protocol) || parsed.username || parsed.password) {
      throw new Error("Source links must be ordinary HTTP(S) URLs.");
    }
    node.href = parsed.href;
  } else {
    node.href = sourceUrl(url);
  }
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
  const heading = ingredientMode
    ? recipe.canonical_ingredients.slice(0, 3).map(displayName).join(", ")
      + (recipe.canonical_ingredients.length > 3 ? ` + ${recipe.canonical_ingredients.length - 3} more` : "")
    : recipe.title;
  title.append(recipe.source_url ? link(heading, recipe.source_url, ingredientMode) : document.createTextNode(heading));
  titleRow.append(element("span", String(index + 1).padStart(2, "0"), "recipe-number"), title);
  const time = recipe.total_minutes === null ? "Total time not reported"
    : `Source total: ${recipe.total_minutes} min`;
  const servings = recipe.servings === null ? "Servings not reported"
    : `Source servings: ${recipe.servings}`;
  article.append(titleRow, element("p", `${time} / ${servings}`, "recipe-meta"),
    ingredientLine("Have", recipe.matched_ingredients, "have", "No pantry ingredients matched"),
    ingredientLine("Need", recipe.missing_ingredients,
      recipe.missing_ingredients.length ? "need" : "have", "No other canonical ingredients"));
  if (ingredientMode) {
    const details = element("details", "", "recipe-details");
    details.append(element("summary", "All canonical ingredients"));
    details.append(element("p", recipe.canonical_ingredients.map(displayName).join(", ")));
    details.append(element("p", "Ingredient names only. Quantities, complete ingredient coverage "
      + "and cooking instructions are not supplied by this index.", "hint"));
    article.append(details);
    const source = element("p", `${recipe.source} / Record ${recipe.id.toLocaleString()} / ${recipe.language}`, "attribution");
    article.append(source);
    if (recipe.source_url) {
      const original = element("p", "", "original-recipe");
      original.append(link("Open original recipe", recipe.source_url, true));
      article.append(original);
    } else {
      article.append(element("p", "No original recipe link was recorded for this ingredient set.", "hint"));
    }
    return article;
  }
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

async function runSearch() {
  const sequence = ++searchSequence;
  $("query-error").hidden = true;
  $("show-all").hidden = true;
  $("shortlist-warning").hidden = true;
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
    if (ingredientMode) query.require_source_url = $("require-source-link").checked;
    if (ingredientMode) {
      $("result-count").textContent = "Searching the ingredient index...";
    }
    const result = ingredientMode
      ? await ingredientClient.search(query, $("ranking").value)
      : searchRecipes(catalog, policy, query, $("ranking").value);
    if (sequence !== searchSequence) return;
    $("results").dataset.ranking = result.ranking;
    const scope = ingredientMode ? "ingredient records" : "sample recipes";
    $("result-count").textContent = `${result.feasible_count.toLocaleString()} match${
      result.feasible_count === 1 ? "" : "es"} in ${catalog.n_recipes.toLocaleString()} ${scope}${
      result.matches.length < result.feasible_count ? ` / showing ${result.matches.length}` : ""}`;
    const absent = [...selected].filter((name) => catalog.ingredient_frequency[catalog.index.get(name)] === 0);
    $("sample-warning").hidden = !absent.length;
    $("sample-warning").textContent = `Not represented in this ${ingredientMode ? "index" : "sample"}: ${absent.map(displayName).join(", ")}.`;
    if (result.retrieval_truncated) {
      $("shortlist-warning").hidden = false;
      $("shortlist-warning").textContent = `All ${result.scanned.toLocaleString()} records were checked. `
        + `The baseline retained ${result.candidates_scored.toLocaleString()} of ${result.feasible_count.toLocaleString()} `
        + "feasible records for ranking; these are not necessarily the learned ranker's global top results.";
    }
    if (result.matches.length) {
      $("results").replaceChildren(...result.matches.map(recipeCard));
      $("show-all").hidden = result.feasible_count <= result.matches.length || topK === 100;
      $("show-all").textContent = result.feasible_count <= 100
        ? "Show all matches" : "Show the first 100 matches";
    } else {
      showEmpty(ingredientMode ? "No ingredient sets fit." : "No sample recipes fit.",
        "The filters have not been relaxed. Try a wider pantry, allow more missing ingredients, or remove a limit."
        + (ingredientMode ? " Unknown source times and serving counts cannot pass their respective limits."
          : " This small sample cannot cover every meal."));
    }
  } catch (error) {
    if (sequence !== searchSequence) return;
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
  ingredientMode = bundle.mode === "ingredient-only";
  if (bundle.mode !== undefined && !ingredientMode) throw new Error("Unsupported recipe demo mode.");
  catalog = ingredientMode ? ingredientCatalogMetadata(bundle.catalog) : prepareCatalog(bundle.catalog);
  validatePolicy(model);
  policy = model;
  if (!Array.isArray(bundle.examples) || !bundle.examples.length) {
    throw new Error("The demo release contains no usable example pantries.");
  }
  $("sample-size").textContent = `${catalog.n_recipes.toLocaleString()} ${ingredientMode ? "ingredient records" : "public recipes"} / Browser only`;
  const provenance = bundle.provenance;
  $("model-link").href = sourceUrl(`https://huggingface.co/${provenance.model_repository}/tree/${provenance.model_revision}`);
  if (provenance.dataset_repository) {
    $("dataset-link").href = sourceUrl(`https://huggingface.co/datasets/${provenance.dataset_repository}${
      provenance.dataset_revision ? `/tree/${provenance.dataset_revision}` : ""}`);
  } else if (ingredientMode) {
    $("dataset-link").href = new URL(bundle.index.path, import.meta.url).href;
    $("dataset-link").textContent = "Local index manifest";
  }
  if (provenance.source_revision) {
    $("code-link").href = sourceUrl(`https://github.com/incrediblecrab/llmmm/tree/${provenance.source_revision}/model/demo`);
  }
  if (ingredientMode) {
    $("scope-label").textContent = "Ingredient-only recipe search";
    $("demo-description").textContent = "Match canonical ingredient sets and source-reported limits. "
      + "Open the original recipe for quantities and cooking instructions; none are copied into this index.";
    $("result-scope-hint").textContent = "These are ingredient sets, not complete recipes. Names may omit "
      + "compound ingredients, optional items and alternatives. Exclusions are not an allergy check.";
    $("scope-limits").textContent = "This index contains every canonical record, including duplicates "
      + "and records without instructions or source links. Browser retrieval uses a baseline shortlist "
      + "before trained ranking; the previous catalog's recovery scores do not measure this search.";
    $("license-note").textContent = "No original titles, descriptive prose, images, quantities or instructions "
      + "are included. Model-weight terms are unchanged. "
      + (provenance.dataset_repository ? "See the dataset's publication terms and source inventory."
        : "This is a local preview; the complete index has not been published.");
    $("index-download").hidden = false;
    $("source-link-filter").hidden = false;
    $("download-size").textContent = `${(bundle.catalog.bytes.initial_compressed_download / 1024 ** 2).toFixed(1)} MiB`;
    $("index-coverage").textContent = `${catalog.coverage.source_total_times.toLocaleString()} records have source total times; `
      + `${catalog.coverage.source_servings.toLocaleString()} have serving counts; `
      + `${(catalog.coverage.url_statuses.source_url || 0).toLocaleString()} have source links. `
      + "Missing metadata is not invented.";
    $("result-count").textContent = "Load the index to search";
    showEmpty("The complete ingredient index.", "Loading is optional and uses your connection's data. "
      + "Search then runs in a browser worker, without an inference server or per-search API charge.");
    $("load-index").addEventListener("click", async () => {
      $("load-index").disabled = true;
      $("download-status").textContent = "Starting the verified index download...";
      try {
        ingredientClient = new IngredientBrowserClient();
        const ready = await ingredientClient.load(new URL(bundle.index.path, import.meta.url), bundle.index, policy,
          (progress) => {
            $("download-status").textContent = progress.phase === "download"
              ? `Downloaded ${(progress.bytes / 1024 ** 2).toFixed(1)} of ${(progress.total / 1024 ** 2).toFixed(1)} MiB`
              : "Checking every ingredient record and its document frequencies...";
          });
        if (ready.n_recipes !== catalog.n_recipes || ready.n_slots !== catalog.n_slots
            || JSON.stringify(ready.vocabulary) !== JSON.stringify(catalog.vocabulary)
            || JSON.stringify(ready.ingredient_frequency) !== JSON.stringify(catalog.ingredient_frequency)) {
          throw new Error("The loaded index differs from the displayed catalog declaration.");
        }
        $("download-status").textContent = `${ready.n_recipes.toLocaleString()} ingredient records verified and ready.`;
        $("load-index").hidden = true;
        $("controls").disabled = false;
        useExample(bundle.examples[0]);
      } catch (error) {
        if (ingredientClient) ingredientClient.close();
        $("download-status").textContent = `The index could not load: ${error.message} Reload to retry.`;
        $("fatal-error").textContent = $("download-status").textContent;
        $("fatal-error").hidden = false;
        $("controls").disabled = true;
        $("result-count").textContent = "Index unavailable";
      }
    });
  } else {
    $("license-note").textContent = `Recipe text: ${[...new Set(catalog.recipes.map((row) => row.license.toUpperCase()))].join(", ")}; `
      + "Wikibooks contributors. See each recipe and the dataset for attribution and changes. "
      + "The ranking weights retain their separate model terms.";
  }
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
  for (const id of ["max-time", "max-missing", "min-servings", "must-use", "exclude", "ranking", "require-source-link"]) {
    $(id).addEventListener("change", () => { topK = 5; runSearch(); });
  }
  $("show-all").addEventListener("click", () => { topK = 100; runSearch(); });
  if (!ingredientMode) {
    $("controls").disabled = false;
    useExample(bundle.examples[0]);
  }
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
