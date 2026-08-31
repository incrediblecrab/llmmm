/**
 * Does the thing in the browser still behave like the thing that was measured?
 *
 * The demo is the only interactive claim on the site, and its failure mode is
 * quiet: a broken fetch, a bad offset into the matrix, or a stale export would
 * all still render a playable game that simply scores wrongly. Nothing about
 * the page would look wrong.
 *
 * So this drives the real page in a real browser and plays it, then compares
 * the rates it observes against the numbers the evaluation harness recorded.
 * It is a smoke test, not a proof: at a few hundred rounds the sampling error
 * is a few points, so it is checking that the demo is doing roughly the right
 * arithmetic, not reproducing the metric to four decimals. The exact
 * reproduction is `scripts/export_site.py --verify`, which re-scores the same
 * exported bytes through the harness itself.
 *
 * One expected discrepancy, now measured rather than assumed: the demo draws
 * recipes of 4 to 9 ingredients while the harness scores everything with 3 or
 * more, so the two are not measuring the same population. Rather than compare
 * across that gap, `export_site.py` re-scores the demo's own 400 recipes
 * through the harness and writes the result into demo_recipes.json, which is
 * what this file reads. On the current export the subpopulation is slightly
 * harder for both players — ease 0.5650 against 0.5872, popularity 0.3300
 * against 0.3650 — though at 400 instances that gap is itself within about one
 * standard error, so it is reported, not claimed.
 *
 *   npm run build && npm run smoke
 */

import { chromium } from "playwright";
import { createServer } from "http";
import { readFile } from "fs/promises";
import { extname, join } from "path";

const ROUNDS = Number(process.env.ROUNDS ?? 300);
const PORT = 8099;

/* How many standard errors of disagreement to tolerate. A hand-picked
 * tolerance would be arbitrary and would silently get looser as ROUNDS falls;
 * a threshold in units of sigma stays calibrated at any number of rounds. At
 * 3.5 a correct demo flakes about one run in two thousand per metric, while a
 * wiring bug — a stale matrix, a bad offset, the wrong scorer — misses by tens
 * of points, which is tens of sigma. */
const SIGMAS = 3.5;

/* Mirror the deployment layout rather than serving dist/ at the root. A base
 * path is exactly the kind of thing that works locally and 404s in
 * production, so the test exercises the prefixed case by default — the same
 * default astro.config.mjs uses. */
const BASE = (process.env.SITE_BASE ?? "/llmmm").replace(/\/+$/, "");

/* Not hardcoded: this is what the harness scores on the demo's own recipe
 * pool, written by scripts/export_site.py. If the export changes, the
 * expectation changes with it. */
const demo = JSON.parse(await readFile("dist/data/demo_recipes.json", "utf8"));
const EXPECTED = demo.expected;

const TYPES = {
  ".html": "text/html", ".js": "text/javascript", ".css": "text/css",
  ".json": "application/json", ".bin": "application/octet-stream",
  ".svg": "image/svg+xml",
};

const server = createServer(async (req, res) => {
  let url = decodeURI(req.url.split("?")[0]);
  if (BASE && !url.startsWith(BASE + "/") && url !== BASE) {
    res.writeHead(404);
    res.end("outside base");
    return;
  }
  let path = join("dist", url.slice(BASE.length) || "/");
  if (path.endsWith("/") || path === "dist") path = join(path, "index.html");
  try {
    const body = await readFile(path);
    res.writeHead(200, {
      "content-type": TYPES[extname(path)] ?? "text/plain",
      "content-length": body.length,
    });
    res.end(body);
  } catch {
    res.writeHead(404);
    res.end("not found");
  }
});
await new Promise((r) => server.listen(PORT, r));

const browser = await chromium.launch();
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
/* A 404 on the matrix is the failure this test exists to catch, and it would
 * otherwise surface only as a hung loading panel. */
page.on("response", (r) => {
  if (r.status() >= 400) errors.push(`HTTP ${r.status()} ${r.url()}`);
});

await page.goto(`http://localhost:${PORT}${BASE}/`, { waitUntil: "networkidle" });
await page.waitForSelector('[data-panel="play"]:not([hidden])', {
  timeout: 60_000,
});

let pop = 0, model = 0;
for (let i = 0; i < ROUNDS; i++) {
  /* Submitted empty, so the human column is always wrong and only the two
   * machine columns are under test. */
  await page.click('button[type="submit"]');
  await page.waitForSelector('[data-panel="result"]:not([hidden])');
  if (await page.getAttribute("[data-pop-mark]", "data-ok") === "true") pop++;
  if (await page.getAttribute("[data-model-mark]", "data-ok") === "true") model++;
  await page.click("[data-again]");
  await page.waitForSelector('[data-panel="play"]:not([hidden])');
}

const board = await page.evaluate(() => ({
  pop: Number(document.querySelector('[data-score="pop"]').textContent),
  model: Number(document.querySelector('[data-score="model"]').textContent),
  round: Number(document.querySelector('[data-score="round"]').textContent),
}));

await browser.close();
server.close();

const observed = { ease: model / ROUNDS, popularity: pop / ROUNDS };
const failures = [];

for (const k of ["ease", "popularity"]) {
  const expected = EXPECTED[k];
  const got = observed[k];
  const off = Math.abs(got - expected);
  /* Both sides are estimates: the browser from ROUNDS plays, the expectation
   * from the export's own draw over demo.expected.n recipes. Neither is the
   * truth, so the comparison carries the error of both. */
  const v = expected * (1 - expected);
  const sigma = Math.sqrt(v / ROUNDS + v / demo.expected.n);
  const limit = SIGMAS * sigma;
  console.log(
    `  ${k.padEnd(11)} browser ${got.toFixed(4)}   expected ${expected.toFixed(4)}` +
    `   Δ ${off.toFixed(4)}   ${(off / sigma).toFixed(1)}σ of ${SIGMAS}`);
  if (off > limit) {
    failures.push(`${k} off by ${off.toFixed(4)} (${(off / sigma).toFixed(1)}σ)`);
  }
}

if (observed.ease <= observed.popularity) {
  failures.push("EASE did not beat the popularity baseline");
}
if (board.round !== ROUNDS || board.pop !== pop || board.model !== model) {
  failures.push(`scoreboard disagrees with observed results: ${JSON.stringify(board)}`);
}
if (errors.length) failures.push(`javascript errors:\n  ${errors.join("\n  ")}`);

if (failures.length) {
  console.error(`\nFAIL (${ROUNDS} rounds)\n  ${failures.join("\n  ")}`);
  process.exit(1);
}
console.log(`\nOK — ${ROUNDS} rounds, no javascript errors, EASE ahead of the baseline.`);
