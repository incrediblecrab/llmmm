import { expect, test } from "@playwright/test";

test("complete index loads only on request, then searches without copied recipe prose", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const assets = [];
  page.on("request", (request) => assets.push(request.url()));
  await page.goto("/");
  await expect(page.locator("#load-index")).toBeVisible();
  await expect(page.locator("#find-recipes")).toBeDisabled();
  expect(assets.some((url) => url.endsWith(".gz"))).toBe(false);
  const bundle = await (await page.request.get("catalog.json")).json();
  expect(bundle.catalog.n_recipes).toBe(4_653_430);
  expect(bundle.catalog.n_slots).toBe(36_707_624);
  await page.locator("#load-index").click();
  await expect(page.locator("#find-recipes")).toBeEnabled({ timeout: 90_000 });
  await expect(page.locator("#download-status")).toContainText("4,653,430 ingredient records verified");
  await expect(page.locator(".recipe-card").first()).toBeVisible({ timeout: 30_000 });
  await expect(page.locator("#result-count")).toContainText("4,653,430 ingredient records");
  expect(await page.locator(".recipe-card").count()).toBe(5);
  await expect(page.locator(".recipe-card").first().getByRole("link", { name: "Open original recipe" })).toBeVisible();
  await page.locator(".recipe-card").first().locator("summary").click();
  await expect(page.locator(".recipe-card").first()).toContainText("Ingredient names only");
  await expect(page.locator(".recipe-card ol")).toHaveCount(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.locator("#max-time").fill("0");
  await page.locator("#max-time").blur();
  await expect(page.locator("#result-count")).toContainText("0 matches");
  await expect(page.locator(".recipe-card")).toHaveCount(0);
  await page.getByRole("button", { name: "Tomato and basil", exact: true }).click();
  await expect(page.locator(".recipe-card").first()).toBeVisible();
  await expect(page.locator("#shortlist-warning")).toContainText("All 4,653,430 records were checked");
  await expect(page.locator("#shortlist-warning")).toContainText("2,000");
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
  expect(errors).toEqual([]);
});

test("corrupt compressed data disables full-index search instead of silently using the sample", async ({ page, context }) => {
  await context.route("**/ingredients.u16.gz", (route) => route.fulfill({
    status: 200, contentType: "application/gzip", body: Buffer.from("corrupt index fixture"),
  }));
  await page.goto("/");
  await expect(page.locator("#load-index")).toBeVisible();
  await page.locator("#load-index").click();
  await expect(page.locator("#fatal-error")).toContainText("integrity check", { timeout: 30_000 });
  await expect(page.locator("#find-recipes")).toBeDisabled();
  await expect(page.locator(".recipe-card")).toHaveCount(0);
});
