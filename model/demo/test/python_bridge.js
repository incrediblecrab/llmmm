import { readFileSync } from "node:fs";
import { candidateFeatures, heuristicScores, learnedScores } from "../ranker.js";
import { prepareCatalog, searchRecipes } from "../search.js";

const input = JSON.parse(readFileSync(0, "utf8"));
const features = input.feature_cases.map((item) => {
  const rows = candidateFeatures(item.available, item.candidates, input.catalog, {
    totalMinutes: item.total_minutes, maxTotalMinutes: item.max_total_minutes,
  });
  return {
    features: rows.map((row) => [...row]),
    learned: learnedScores(rows, input.policy),
    heuristic: heuristicScores(rows),
  };
});
const catalog = prepareCatalog(input.catalog);
const searches = input.queries.map((query) => Object.fromEntries(
  ["learned", "heuristic"].map((ranking) => {
    const result = searchRecipes(catalog, input.policy, query, ranking);
    return [ranking, {
      feasible_count: result.feasible_count,
      matches: result.matches.map((recipe) => ({
        id: recipe.id, score: recipe.score,
        matched_ingredients: recipe.matched_ingredients,
        missing_ingredients: recipe.missing_ingredients,
      })),
    }];
  }),
));
process.stdout.write(JSON.stringify({ features, searches }));
