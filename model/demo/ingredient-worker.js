import { searchIngredientCatalog } from "./ingredient-catalog.js";
import { loadIngredientCatalog, loadResultUrls } from "./ingredient-loader.js";
import { validatePolicy } from "./ranker.js";

let catalog;
let policy;
let indexUrl;
let currentSearch = 0;
let loading = false;

self.addEventListener("message", async ({ data }) => {
  const { id, operation } = data;
  try {
    if (operation === "load") {
      if (loading || catalog) throw new Error("The ingredient index is already loading or loaded.");
      loading = true;
      validatePolicy(data.policy);
      policy = data.policy;
      indexUrl = new URL(data.index_url);
      catalog = await loadIngredientCatalog(indexUrl, data.index_record,
        (progress) => self.postMessage({ id, progress }));
      self.postMessage({ id, result: {
        n_recipes: catalog.n_recipes, n_slots: catalog.n_slots, coverage: catalog.coverage,
        vocabulary: catalog.vocabulary, ingredient_frequency: catalog.ingredient_frequency,
      } });
    } else if (operation === "search") {
      if (!catalog) throw new Error("Load the ingredient index before searching.");
      currentSearch = id;
      const result = await searchIngredientCatalog(catalog, policy, data.query, data.ranking, {
        maxCandidates: data.max_candidates ?? 2000,
        cancelled: () => currentSearch !== id,
      });
      result.matches = await loadResultUrls(indexUrl, catalog, result.matches);
      if (currentSearch !== id) throw new DOMException("Superseded by a newer search.", "AbortError");
      self.postMessage({ id, result });
    } else {
      throw new Error("Unknown ingredient-search worker operation.");
    }
  } catch (error) {
    self.postMessage({ id, error: { name: error.name, message: error.message } });
  } finally {
    if (operation === "load") loading = false;
  }
});
