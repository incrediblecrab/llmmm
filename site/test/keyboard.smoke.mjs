/**
 * Can the demo be played without a mouse?
 *
 * Separate from `demo.smoke.mjs` on purpose: that file asks whether the game
 * computes the right thing, this one asks whether it can be reached. They
 * fail for unrelated reasons and a single script that did both would report
 * an arithmetic bug and an accessibility bug as the same red.
 *
 * The failure this exists to catch is specific and was real. Revealing the
 * result hides the panel holding focus, and a browser responds by dropping
 * focus to the document body — so "Next recipe" went from one tab away to
 * unreachable within any reasonable number of presses, while the page looked
 * completely correct. Nothing in a screenshot, a build, or the metric smoke
 * test would show it.
 *
 *   npm run build && npm run keyboard
 */

import { chromium } from "playwright";
import { createServer } from "http";
import { readFile } from "fs/promises";
import { extname, join } from "path";

const ROUNDS = Number(process.env.ROUNDS ?? 3);
const PORT = 8097;
const BASE = (process.env.SITE_BASE ?? "/llmmm").replace(/\/+$/, "");

const TYPES = {
  ".html": "text/html", ".css": "text/css", ".js": "text/javascript",
  ".json": "application/json", ".bin": "application/octet-stream",
  ".svg": "image/svg+xml", ".woff2": "font/woff2",
};

const serve = () =>
  createServer(async (req, res) => {
    let p = join("dist", decodeURIComponent(req.url.split("?")[0])
      .replace(new RegExp(`^${BASE}`), ""));
    if (p.endsWith("/")) p += "index.html";
    try {
      const body = await readFile(p).catch(() => readFile(join(p, "index.html")));
      res.writeHead(200, { "Content-Type": TYPES[extname(p)] ?? "text/plain" });
      res.end(body);
    } catch {
      res.writeHead(404);
      res.end("not found");
    }
  });

let failures = 0;
const check = (ok, msg) => {
  if (!ok) failures++;
  console.log(`  ${ok ? "ok  " : "FAIL"}  ${msg}`);
};

const server = serve();
await new Promise((r) => server.listen(PORT, r));
const browser = await chromium.launch();

try {
  // A reader who has asked for less motion should still get a working game,
  // so the same page is exercised under that preference rather than a
  // separate one.
  const ctx = await browser.newContext({ reducedMotion: "reduce" });
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));

  await page.goto(`http://localhost:${PORT}${BASE}/`, { waitUntil: "networkidle" });
  await page.waitForSelector("[data-panel='play']:not([hidden])", { timeout: 30_000 });

  const dur = await page.evaluate(() => {
    const el = document.querySelector(".sb b");
    return el ? getComputedStyle(el).transitionDuration : "0s";
  });
  check(parseFloat(dur) < 0.05,
    `prefers-reduced-motion collapses transitions (${dur})`);

  let hops = 0;
  let onInput = false;
  while (hops++ < 30 && !onInput) {
    await page.keyboard.press("Tab");
    onInput = await page.evaluate(() =>
      document.activeElement?.id === "guess-input");
  }
  check(onInput, `guess input is reachable by Tab (${hops} presses)`);

  for (let round = 1; round <= ROUNDS; round++) {
    await page.keyboard.type("salt");
    await page.keyboard.press("Enter");
    await page.waitForSelector("[data-panel='result']:not([hidden])", { timeout: 10_000 });
    check(
      await page.evaluate(() => document.activeElement?.hasAttribute("data-again")),
      `round ${round}: focus follows the reveal to "Next recipe"`);

    await page.keyboard.press("Enter");
    await page.waitForSelector("[data-panel='play']:not([hidden])", { timeout: 10_000 });
    check(
      await page.evaluate(() => document.activeElement?.id === "guess-input"),
      `round ${round}: focus returns to the guess input`);
  }

  const rounds = await page.textContent(".sb.rounds b");
  check(Number(rounds) >= ROUNDS,
    `the scoreboard counted every keyboard round (${rounds})`);
  check(
    (await page.getAttribute(".scoreboard", "aria-live")) === "polite",
    "the scoreboard announces changes to a screen reader");
  check(errors.length === 0,
    `no page errors (${errors.join("; ") || "none"})`);
} finally {
  await browser.close();
  server.close();
}

console.log(failures
  ? `\n${failures} check(s) failed.`
  : `\nOK — ${ROUNDS} rounds played with the keyboard alone.`);
process.exit(failures ? 1 : 0);
