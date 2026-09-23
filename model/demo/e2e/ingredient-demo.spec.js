import { expect, test } from "@playwright/test";
import { bindServingManifest } from "./serving-manifest.js";

bindServingManifest(test);

const examples = ["Tomato and basil", "Chicken, rice, broccoli", "Eggs and potatoes"];

function renderedCards(page) {
  return page.locator(".recipe-card").evaluateAll((cards) => cards.map((card) => ({
    id: Number(card.dataset.recipeId),
    title: card.querySelector(".recipe-title").textContent,
    lines: [...card.querySelectorAll(".recipe-ingredients li")].map((item) => item.textContent),
    link: card.querySelector(".original-recipe a")?.getAttribute("href") ?? null,
  })));
}

function expectedCards(example) {
  return example.results.matches.map((match) => ({
    id: match.id, title: match.title, lines: match.ingredient_lines, link: new URL(match.source_url).href,
  }));
}

function trackIndexRequests(page) {
  const requests = [];
  page.on("request", (request) => {
    const path = new URL(request.url()).pathname;
    if (path.endsWith(".gz") || path.endsWith("ingredient-index.json")) requests.push(path);
  });
  return requests;
}

test("example recipe cards appear before any download and match the complete index live", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const assets = [];
  page.on("request", (request) => assets.push({ url: request.url(), method: request.method() }));
  const indexRequests = trackIndexRequests(page);
  await page.goto("/");
  const bundle = await (await page.request.get("catalog.json")).json();
  expect(bundle.catalog.n_recipes).toBe(4_653_430);
  expect(bundle.catalog.n_slots).toBe(36_707_624);
  expect(bundle.examples.map((example) => example.label)).toEqual(examples);
  expect(bundle.examples.every((example) => example.results.matches.length === 5 && example.results.matches.every(
    (match) => match.title && match.ingredient_lines?.length && /^https?:\/\//.test(match.source_url)))).toBe(true);
  expect(bundle.examples[0].results.matches.some((match) => match.source_url.startsWith("http://www.cookbooks.com/"))).toBe(true);

  await expect(page.locator("#results")).toHaveAttribute("data-origin", "precomputed");
  await expect(page.locator("#precomputed-note")).toBeVisible();
  await expect(page.locator("#load-index")).toBeVisible();
  await expect(page.locator("#find-recipes")).toBeDisabled();
  for (const [position, label] of examples.entries()) {
    await page.getByRole("button", { name: label, exact: true }).click();
    await expect.poll(() => renderedCards(page)).toEqual(expectedCards(bundle.examples[position]));
    await expect(page.locator("#results")).toHaveAttribute("data-origin", "precomputed");
  }
  await expect(page.locator("#result-count")).toContainText("in 4,653,430 ingredient records");
  await expect(page.locator(".recipe-card").first().getByRole("link", { name: "Open original recipe" })).toBeVisible();
  await expect(page.locator(".recipe-card ol")).toHaveCount(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(indexRequests).toEqual([]);

  await page.getByRole("button", { name: examples[0], exact: true }).click();
  await page.locator("#load-index").click();
  await expect(page.locator("#find-recipes")).toBeEnabled({ timeout: 90_000 });
  await expect(page.locator("#download-status")).toContainText("4,653,430 ingredient records verified");
  await expect(page.locator("#results")).toHaveAttribute("data-origin", "live", { timeout: 60_000 });
  await expect(page.locator("#results")).toHaveAttribute("data-ranking", "learned");
  await expect(page.locator("#precomputed-note")).toBeHidden();
  for (const [position, label] of [...examples.entries()].reverse()) {
    await page.getByRole("button", { name: label, exact: true }).click();
    await expect.poll(() => renderedCards(page), { timeout: 60_000 }).toEqual(expectedCards(bundle.examples[position]));
    await expect(page.locator("#results")).toHaveAttribute("data-origin", "live");
  }
  await expect(page.locator("#shortlist-warning")).toContainText("All 4,653,430 records were checked");
  await expect(page.locator("#shortlist-warning")).toContainText("2,000");
  await page.locator(".recipe-card").first().locator("summary").click();
  await expect(page.locator(".recipe-card").first()).toContainText("Search matches these normalized names");

  const started = Date.now();
  await page.locator("#show-all").click();
  await expect(page.locator(".recipe-card")).toHaveCount(100, { timeout: 60_000 });
  test.info().annotations.push({ type: "show-100-ms", description: String(Date.now() - started) });
  const hundred = await renderedCards(page);
  expect(hundred.every((card) => card.title && /^https?:\/\//.test(card.link))).toBe(true);
  expect(hundred.filter((card) => card.lines.length).length).toBeGreaterThan(0);

  await page.locator("#max-time").fill("0");
  await page.locator("#max-time").blur();
  await expect(page.locator("#result-count")).toContainText("0 matches");
  await expect(page.locator(".recipe-card")).toHaveCount(0);
  await page.locator("#require-source-link").uncheck();
  await page.getByRole("button", { name: "Chicken, rice, broccoli", exact: true }).click();
  await expect(page.locator(".recipe-card").first()).toContainText("Source total:");
  await page.getByRole("button", { name: "Tomato and basil", exact: true }).click();
  await page.locator("#ranking").selectOption("heuristic");
  await expect(page.locator("#results")).toHaveAttribute("data-ranking", "heuristic");
  await page.locator(".extra-filters summary").click();
  await page.locator("#exclude").fill("tomato");
  await page.locator("#must-use").fill("tomato");
  await page.locator("#must-use").blur();
  await expect(page.locator("#query-error")).toContainText("both required and excluded");
  await expect(page.locator(".recipe-card")).toHaveCount(0);
  const marker = 'private-pantry-marker<img src=x onerror="window.injected=true">';
  await page.locator("#pantry-entry").fill(marker);
  await page.locator("#add-ingredient").click();
  await expect(page.locator("#query-error")).toContainText("not a recognized ingredient");
  expect(await page.evaluate(() => window.injected)).toBeUndefined();
  expect(await page.evaluate(() => localStorage.length + sessionStorage.length)).toBe(0);
  expect(assets.every(({ url, method }) => method === "GET" && !url.includes("private-pantry-marker"))).toBe(true);
  expect(errors).toEqual([]);
});

test("corrupt compressed data disables search and keeps only the labeled precomputed examples", async ({ page, context }) => {
  await context.route("**/ingredients.u16.gz", (route) => route.fulfill({
    status: 200, contentType: "application/gzip", headers: { "access-control-allow-origin": "*" },
    body: Buffer.from("corrupt index fixture"),
  }));
  await page.goto("/");
  await expect(page.locator(".recipe-card")).toHaveCount(5);
  await page.locator("#load-index").click();
  await expect(page.locator("#fatal-error")).toContainText("integrity check", { timeout: 30_000 });
  await expect(page.locator("#find-recipes")).toBeDisabled();
  await expect(page.locator("#results")).toHaveAttribute("data-origin", "precomputed");
  await expect(page.locator("#precomputed-note")).toBeVisible();
  await expect(page.locator(".recipe-card")).toHaveCount(5);
});

test("failed index download is explicit and never substitutes live-looking results", async ({ page, context }) => {
  await context.route("**/ingredient-index.json", (route) => route.abort("failed"));
  await page.goto("/");
  await expect(page.locator(".recipe-card")).toHaveCount(5);
  await page.locator("#load-index").click();
  await expect(page.locator("#fatal-error")).toContainText("index could not load");
  await expect(page.locator("#find-recipes")).toBeDisabled();
  await expect(page.locator("#results")).toHaveAttribute("data-origin", "precomputed");
  await expect(page.locator(".recipe-card")).toHaveCount(5);
});

test("corrupt recipe text fails the search instead of showing cards without their text", async ({ page, context }) => {
  await context.route("**/text-shards.json.gz", (route) => route.fulfill({
    status: 200, contentType: "application/gzip", headers: { "access-control-allow-origin": "*" },
    body: Buffer.from("corrupt text fixture"),
  }));
  await page.goto("/");
  await page.locator("#load-index").click();
  await expect(page.locator("#find-recipes")).toBeEnabled({ timeout: 90_000 });
  await expect(page.locator("#query-error")).toContainText("integrity check", { timeout: 60_000 });
  await expect(page.locator("#result-count")).toHaveText("Search not run");
  await expect(page.locator(".recipe-card")).toHaveCount(0);
});

test("a dropped recipe-text download is retried instead of failing the search", async ({ page, context }) => {
  let dropped = 0;
  await context.route("**/text/*.json.gz", (route) => {
    if (dropped > 0) return route.fallback();
    dropped += 1;
    return route.abort("failed");
  });
  await page.goto("/");
  const bundle = await (await page.request.get("catalog.json")).json();
  await page.locator("#load-index").click();
  await expect(page.locator("#results")).toHaveAttribute("data-origin", "live", { timeout: 90_000 });
  await expect.poll(() => renderedCards(page), { timeout: 60_000 }).toEqual(expectedCards(bundle.examples[0]));
  await expect(page.locator("#query-error")).toBeHidden();
  expect(dropped).toBe(1);
});
