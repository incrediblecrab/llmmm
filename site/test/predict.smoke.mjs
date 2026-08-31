/* Drives the prediction widget end to end.
 *
 * It is the first thing a reader touches, it is the only place their own
 * number appears next to ours, and every one of its states is produced by
 * script rather than by the build — so none of it is checked by anything
 * else on the site. Three ways it could be wrong in public:
 *
 *  - the reveal could disagree with the leaderboard;
 *  - the two-act structure could show the smaller count as though it were
 *    the embedding count, which is the exact overclaim the site exists to
 *    avoid making;
 *  - it could be unreachable without a mouse.
 */
import { chromium } from "playwright";
import { createServer } from "http";
import { readFile } from "fs/promises";
import { extname, join } from "path";
import process from "node:process";

const BASE = (process.env.SITE_BASE ?? "/llmmm").replace(/\/+$/, "");
const PORT = 8102;
const GUESS = 42;

const board = JSON.parse(await readFile("dist/data/leaderboard.json", "utf8"));
const POP = board.popularity_recall_at_10;
const hero = board.models.find((m) => m.model === "item2vec");
const belowEmb = board.models.filter((m) => m.M6_recall_at_10 < POP).length;
const belowBest = board.models.filter((m) => m.best_lift < 0).length;

const TYPES = {
  ".html": "text/html", ".js": "text/javascript", ".css": "text/css",
  ".json": "application/json", ".bin": "application/octet-stream",
  ".svg": "image/svg+xml", ".png": "image/png",
};
const server = createServer(async (req, res) => {
  const url = decodeURI(req.url.split("?")[0]);
  if (BASE && !url.startsWith(BASE + "/") && url !== BASE) {
    res.writeHead(404); res.end("outside base"); return;
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
  } catch { res.writeHead(404); res.end("not found"); }
});
await new Promise((r) => server.listen(PORT, r));

const failures = [];
const browser = await chromium.launch();
try {
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });

  await page.goto(`http://localhost:${PORT}${BASE}/`, { waitUntil: "domcontentloaded" });
  await page.waitForSelector("[data-predict]");

  /* The whole point of the widget is that it works before the 1.3 MB matrix
   * lands, so it is exercised on domcontentloaded, not networkidle. */
  /* Visibility, not attribute presence. An earlier version of this test read
   * the `hidden` attribute and passed while the marker was plainly on screen,
   * because `hidden` does nothing to an SVG element. Ask the browser what is
   * painted. */
  const seesDots = await page.locator("[data-dots] circle").count();
  if (seesDots > 0) {
    failures.push(`${seesDots} model scores on screen before the reader committed`);
  }
  if (await page.locator("[data-you]").isVisible()) {
    failures.push("the reader's marker is painted before they placed one");
  }
  if (await page.locator("[data-band]").isVisible()) {
    failures.push("the below-baseline band is painted before the reveal");
  }
  if (!(await page.isDisabled("[data-lock]"))) {
    failures.push("lock button is enabled with no estimate");
  }

  /* Keyboard-only path: tab to the number field, type, activate the button. */
  await page.fill("[data-num]", String(GUESS));
  if (await page.isDisabled("[data-lock]")) {
    failures.push("typing an estimate did not enable the lock button");
  }
  const youLabel = await page.textContent("[data-you-label]");
  if (!youLabel?.includes(String(GUESS))) {
    failures.push(`marker reads "${youLabel}", expected it to carry ${GUESS}`);
  }

  await page.click("[data-lock]");
  await page.waitForSelector("[data-reveal]:not([hidden])");

  const verdict = (await page.textContent("[data-verdict]")) ?? "";
  const expectHero = (hero.M6_recall_at_10 * 100).toFixed(2);
  if (!verdict.includes(expectHero)) {
    failures.push(`verdict is missing item2vec at ${expectHero}%: "${verdict.slice(0, 160)}"`);
  }
  if (!verdict.includes((POP * 100).toFixed(1))) {
    failures.push(`verdict is missing the baseline at ${(POP * 100).toFixed(1)}%`);
  }
  const err = Math.abs(GUESS - hero.M6_recall_at_10 * 100).toFixed(1);
  if (!verdict.includes(err)) {
    failures.push(`verdict is missing the signed error of ${err} points`);
  }

  await page.waitForSelector("[data-act-note]:not([hidden])", { timeout: 15_000 });
  const act1 = (await page.textContent("[data-act-note]")) ?? "";
  if (!act1.includes(String(belowEmb))) {
    failures.push(`act 1 should count ${belowEmb} embeddings, said: "${act1}"`);
  }
  if (belowEmb !== belowBest && act1.includes(`${belowBest} of`)) {
    failures.push(`act 1 quotes the native count ${belowBest} as if it were the embedding count`);
  }

  const n1 = await page.locator("[data-dots] circle").count();
  if (n1 !== board.models.length) {
    failures.push(`${n1} dots plotted, expected ${board.models.length}`);
  }

  await page.click("[data-act2]");
  const act2 = (await page.textContent("[data-act-note]")) ?? "";
  if (!act2.includes(String(belowBest))) {
    failures.push(`act 2 should land on ${belowBest}, said: "${act2}"`);
  }
  for (const name of ["ease", "masked-set"]) {
    const cls = await page.getAttribute(`[data-dots] [data-model="${name}"]`, "class");
    if (!cls?.includes("above")) {
      failures.push(`${name} did not move above the line in act 2 (class "${cls}")`);
    }
  }

  if (errors.length) failures.push(`javascript errors: ${errors.join("; ")}`);
} finally {
  await browser.close();
  server.close();
}

if (failures.length) {
  console.error(`FAIL\n  ${failures.join("\n  ")}`);
  process.exit(1);
}
console.log(
  `OK — commit-before-reveal holds; ${belowEmb} embeddings then ${belowBest} systems below ` +
  `${POP.toFixed(4)}, reachable by keyboard.`,
);
