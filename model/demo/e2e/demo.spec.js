import { test, expect } from "@playwright/test";
import { bindServingManifest } from "./serving-manifest.js";

bindServingManifest(test);

async function openDemo(page) {
  await page.goto("/");
  await expect(page.locator("#find-recipes")).toBeEnabled();
  await expect(page.locator("#fatal-error")).toBeHidden();
  return (await (await page.request.get("catalog.json")).json()).catalog;
}

test("anonymous landing runs the trained ranker and renders original recipe text", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const catalog = await openDemo(page);
  expect(await page.locator(".recipe-card").count()).toBeGreaterThanOrEqual(2);
  await expect(page.locator("#results")).toHaveAttribute("data-ranking", "learned");
  await expect(page.locator("#result-count")).toContainText(`${catalog.n_recipes} sample recipes`);
  const first = page.locator(".recipe-card").first();
  const recipeId = await first.getAttribute("data-recipe-id");
  const recipe = catalog.recipes.find((item) => item.id === recipeId);
  expect(recipe).toBeTruthy();
  await first.locator("summary").click();
  await expect(first.locator(".recipe-details ul li").first()).toHaveText(recipe.raw_ingredients[0]);
  await expect(first.locator(".recipe-details ol li").first()).toHaveText(recipe.instructions[0]);
  await expect(first.locator(".source-notes")).toContainText(recipe.source_limitations[0]);
  await expect(first.getByRole("link", { name: "Contributor history" })).toHaveAttribute("href", recipe.attribution_url);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  expect(errors).toEqual([]);
});

test("zero minutes produces no matches, without relaxing the limit", async ({ page }) => {
  await openDemo(page);
  await page.locator("#max-time").fill("0");
  await page.locator("#max-time").blur();
  await expect(page.locator("#result-count")).toContainText("0 matches");
  await expect(page.locator(".recipe-card")).toHaveCount(0);
  await expect(page.locator("#results")).toContainText("filters have not been relaxed");
  await page.locator("#max-time").fill("");
  await page.locator("#max-time").blur();
  expect(await page.locator(".recipe-card").count()).toBeGreaterThan(0);
});

test("exact pantry, source time, requirements and exclusions survive both rankers", async ({ page }) => {
  const catalog = await openDemo(page);
  const selected = catalog.recipes.find((recipe) => recipe.total_minutes !== null);
  expect(selected).toBeTruthy();
  await page.getByRole("button", { name: "Clear", exact: true }).click();
  await page.locator("#pantry-entry").fill(selected.canonical_ingredients.join(", "));
  await page.getByRole("button", { name: "Add", exact: true }).click();
  await page.locator("#max-time").fill(String(selected.total_minutes));
  await page.locator("#max-time").blur();
  await page.locator("#max-missing").selectOption("0");
  await page.locator(".extra-filters summary").click();
  await page.locator("#must-use").fill(selected.canonical_ingredients[0]);
  await page.locator("#must-use").blur();
  const excluded = catalog.vocabulary.find((name) => !selected.canonical_ingredients.includes(name));
  await page.locator("#exclude").fill(excluded);
  await page.locator("#exclude").blur();
  if (selected.servings !== null) {
    await page.locator("#min-servings").fill(String(selected.servings));
    await page.locator("#min-servings").blur();
  }
  for (const ranking of ["heuristic", "learned"]) {
    await page.locator("#ranking").selectOption(ranking);
    await expect(page.locator("#query-error")).toBeHidden();
    const ids = await page.locator(".recipe-card").evaluateAll((nodes) => nodes.map((node) => node.dataset.recipeId));
    expect(ids).toContain(selected.id);
    for (const id of ids) {
      const row = catalog.recipes.find((recipe) => recipe.id === id);
      expect(row.total_minutes).not.toBeNull();
      expect(row.total_minutes).toBeLessThanOrEqual(selected.total_minutes);
      expect(row.canonical_ingredients.every((name) => selected.canonical_ingredients.includes(name))).toBe(true);
      expect(row.canonical_ingredients).toContain(selected.canonical_ingredients[0]);
      expect(row.canonical_ingredients).not.toContain(excluded);
      if (selected.servings !== null) expect(row.servings).toBeGreaterThanOrEqual(selected.servings);
    }
  }
});

test("invalid input is text, not markup, and pantry state stays local", async ({ page }) => {
  await openDemo(page);
  const requests = [];
  page.on("request", (request) => requests.push({ url: request.url(), body: request.postData() }));
  const invalid = "<img src=x onerror=alert(1)>";
  await page.locator("#pantry-entry").fill(invalid);
  await page.getByRole("button", { name: "Add", exact: true }).click();
  await expect(page.locator("#query-error")).toContainText(invalid);
  await expect(page.locator("#results img")).toHaveCount(0);
  await expect(page.locator(".recipe-card")).toHaveCount(0);
  expect(requests.some((request) => request.url.includes("onerror") || request.body)).toBe(false);
  expect(await page.evaluate(() => ({ local: localStorage.length, session: sessionStorage.length })))
    .toEqual({ local: 0, session: 0 });
  await page.reload();
  await expect(page.locator("#find-recipes")).toBeEnabled();
  await expect(page.locator("#query-error")).toBeHidden();
  await expect(page.locator("#pantry-entry")).toHaveValue("");
});

test("a tampered policy fails closed instead of producing fallback recipes", async ({ page }) => {
  await page.route("**/policy.json", async (route) => {
    const response = await route.fetch();
    const policy = await response.json();
    policy.tensors["network.2.bias"][0] += 1;
    await route.fulfill({ response, json: policy });
  });
  await page.goto("/");
  await expect(page.locator("#fatal-error")).toContainText("integrity check");
  await expect(page.locator("#find-recipes")).toBeDisabled();
  await expect(page.locator("#pantry-entry")).toBeDisabled();
  await expect(page.locator(".recipe-card")).toHaveCount(0);
});

test("a failed catalog download is an explicit unavailable state", async ({ page }) => {
  await page.route("**/catalog.json", (route) => route.abort("failed"));
  await page.goto("/");
  await expect(page.locator("#fatal-error")).toContainText("could not start");
  await expect(page.locator("#find-recipes")).toBeDisabled();
  await expect(page.locator("#pantry-entry")).toBeDisabled();
  await expect(page.locator(".recipe-card")).toHaveCount(0);
});
