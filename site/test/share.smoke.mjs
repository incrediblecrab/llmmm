/* The share grid has to agree with the scoreboard.
 *
 * It is the one number on the site a reader carries somewhere else, so it is
 * also the one that can be wrong in public. The scoreboard and the grid are
 * built from different data — running counters versus a per-round history —
 * so nothing but a test keeps them equal. Two ways they could drift:
 *
 *  - a round that updates the counter but not the history, or the reverse;
 *  - the ten-round cap silently truncating the tally as well as the grid.
 *
 * So: play more rounds than the cap, then check every square against the
 * board.
 */
import { chromium } from "playwright";
import { createServer } from "http";
import { readFile } from "fs/promises";
import { extname, join } from "path";
import process from "node:process";

const BASE = (process.env.SITE_BASE ?? "/llmmm").replace(/\/+$/, "");
const PORT = 8101;
const ROUNDS = Number(process.env.ROUNDS ?? 14);
const CAP = 10;
const HIT = "\u{1F7E9}";
const MISS = "\u2B1C";

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
const url = `http://localhost:${PORT}${BASE}/`;
const stop = () => { try { server.close(); } catch {} };

const failures = [];
let browser;
try {
  browser = await chromium.launch();
  const ctx = await browser.newContext();
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));

  await page.goto(url, { waitUntil: "networkidle" });
  await page.waitForSelector('[data-panel="play"]:not([hidden])', { timeout: 60_000 });

  for (let i = 0; i < ROUNDS; i++) {
    await page.fill("[data-input]", i % 3 === 0 ? "salt" : "zzz-not-an-ingredient");
    await page.press("[data-input]", "Enter");
    await page.waitForSelector('[data-panel="result"]:not([hidden])');
    if (i < ROUNDS - 1) {
      await page.click("[data-again]");
      await page.waitForSelector('[data-panel="play"]:not([hidden])');
    }
  }

  const board = await page.evaluate(() => {
    const n = (k) => Number(document.querySelector(`[data-score="${k}"]`).textContent);
    return { you: n("you"), pop: n("pop"), model: n("model"), round: n("round") };
  });
  const text = await page.evaluate(() => window.__share());

  if (board.round !== ROUNDS) {
    failures.push(`played ${ROUNDS} rounds, board says ${board.round}`);
  }

  const lines = text.split("\n");
  const rows = { You: "you", Counting: "pop", EASE: "model" };
  for (const [label, key] of Object.entries(rows)) {
    const line = lines.find((l) => l.startsWith(label));
    if (!line) { failures.push(`no "${label}" row in the share text`); continue; }

    const squares = [...line].filter((c) => c === HIT || c === MISS);
    if (squares.length !== Math.min(ROUNDS, CAP)) {
      failures.push(
        `${label}: ${squares.length} squares, expected ${Math.min(ROUNDS, CAP)}`);
    }

    const m = line.match(/(\d+)\/(\d+)\s*$/);
    if (!m) { failures.push(`${label}: no n/total tally`); continue; }
    const [, got, total] = m.map(Number);
    if (got !== board[key]) {
      failures.push(`${label}: grid says ${got}, scoreboard says ${board[key]}`);
    }
    /* The tally is over every round played, not just the ten shown. */
    if (total !== board.round) {
      failures.push(`${label}: total ${total}, rounds played ${board.round}`);
    }
  }

  if (!text.includes("llmmm")) failures.push("share text carries no link back");
  if (errors.length) failures.push(`javascript errors: ${errors.join("; ")}`);

  const copied = await page.evaluate(async () => {
    document.querySelector("[data-share]").click();
    await new Promise((r) => setTimeout(r, 300));
    return document.querySelector("[data-share-status]").textContent;
  });
  if (!copied) failures.push("share button left no status for the reader");

  console.log(`\n${text}\n`);
} finally {
  if (browser) await browser.close();
  stop();
}

if (failures.length) {
  console.error(`FAIL (${ROUNDS} rounds)\n  ${failures.join("\n  ")}`);
  process.exit(1);
}
console.log(`OK — share grid matches the scoreboard over ${ROUNDS} rounds.`);
