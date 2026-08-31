/* Renders the social card from the same artefact the site reads.
 *
 * For most people the card *is* the site: it is what a link resolves to in a
 * timeline, and a bare URL gets scrolled past. Drawing it by hand would put
 * another uncheckable copy of the numbers somewhere, so it reads
 * leaderboard.json and applies the same two rules the pages do —
 * `best_recall_at_10` for the score, `best_lift < 0` for below-the-line.
 *
 * Regenerate with `npm run og`.
 */
import { chromium } from "playwright";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const read = (f) =>
  JSON.parse(fs.readFileSync(path.join(root, "public/data", f), "utf8"));
const board = read("leaderboard.json");
/* The corpus size comes from the generation record the site reads, not from a
 * literal here. A second copy of "4.65M" is a second thing to get wrong. */
const recipes = read("corpus.json").generation.recipes;
const recipesM = (recipes / 1e6).toFixed(2).replace(/0$/, "");

const POP = board.popularity_recall_at_10;
const models = board.models
  .map((m) => ({ name: m.model, v: m.best_recall_at_10, below: m.best_lift < 0 }))
  .sort((a, b) => a.v - b.v);

const W = 1200, H = 630;
const PAD = 76, AX_Y = 498, AX_W = W - PAD * 2;
const R = 13, ROW = 2 * R + 5;
const max = Math.max(...models.map((m) => m.v), POP) * 1.05;
const X = (v) => PAD + (v / max) * AX_W;

/* Dodge upward so no two dots overlap; same idea as the on-page swarm. */
const lastInRow = [];
const placed = models.map((m) => {
  const px = X(m.v);
  let row = 0;
  while (lastInRow[row] !== undefined && px - lastInRow[row] < 2 * R + 1) row++;
  lastInRow[row] = px;
  return { ...m, px, row };
});
const nBelow = placed.filter((m) => m.below).length;
if (nBelow !== board.n_below_popularity) {
  throw new Error(`below-count ${nBelow} != artefact ${board.n_below_popularity}`);
}
/* From `placed`, not `models` — the dodge pass returns new objects, so the
 * originals carry no px and every coordinate off them lands at zero. */
const worst = placed[0];
const best = placed[placed.length - 1];

const yOf = (row) => AX_Y - R - 5 - row * ROW;
const topY = yOf(Math.max(...placed.map((p) => p.row))) - R;
const basePx = X(POP);

const dots = placed
  .map((p) => `<circle cx="${p.px.toFixed(1)}" cy="${yOf(p.row).toFixed(1)}" r="${R}"
      fill="${p.below ? "#b23b2e" : "#1d6b93"}" stroke="#fff" stroke-width="2.5"/>`)
  .join("");

const axTicks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
  .filter((t) => t <= max)
  .map((t) => `<text x="${X(t).toFixed(1)}" y="${AX_Y + 33}" font-family="ui-monospace,Menlo,monospace"
      font-size="20" fill="#5b636c" text-anchor="middle">${t.toFixed(2)}</text>`)
  .join("");

const labelY = topY - 16;
const boot = board.bootstrap;
/* The proof line only earns its place if the interval is actually a point. */
const bootExact = boot &&
  boot.n_below_popularity_ci95[0] === boot.n_below_popularity_ci95[1] &&
  boot.n_below_popularity_ci95[0] === nBelow;

const html = `<!doctype html><html><head><meta charset="utf-8"><style>
  html,body{margin:0;padding:0;width:${W}px;height:${H}px;background:#fbfaf8;}
  .eyebrow{position:absolute;left:${PAD}px;top:50px;
    font:600 19px/1 "Helvetica Neue",Helvetica,sans-serif;
    letter-spacing:.11em;text-transform:uppercase;color:#6d757e;}
  h1{position:absolute;left:${PAD}px;top:84px;margin:0;width:${W - PAD * 2}px;
    font:700 80px/0.98 "Helvetica Neue",Helvetica,sans-serif;
    letter-spacing:-.028em;color:#14181c;}
  h1 em{font-style:normal;color:#b23b2e;}
  .sub{position:absolute;left:${PAD}px;top:262px;margin:0;
    font:400 26px/1.3 "Iowan Old Style","Palatino Linotype",Georgia,serif;color:#4d545c;}
  svg{position:absolute;left:0;top:0;}
  .proof{position:absolute;left:${PAD}px;bottom:28px;
    font:400 19px/1 ui-monospace,Menlo,monospace;color:#6d757e;}
  .url{position:absolute;right:${PAD}px;bottom:28px;
    font:400 19px/1 ui-monospace,Menlo,monospace;color:#6d757e;}
</style></head><body>
  <div class="eyebrow">A held-out evaluation of ingredient recommendation</div>
  <h1><em>${nBelow} of ${models.length} models</em><br/>lost to counting.</h1>
  <p class="sub">${recipesM}M recipes \u00b7 recall@10 on recipes no model had seen</p>
  <svg width="${W}" height="${H}">
    <rect x="${PAD}" y="${topY - 6}" width="${(basePx - PAD).toFixed(1)}"
      height="${(AX_Y - topY + 6).toFixed(1)}" fill="#b23b2e" opacity="0.07"/>
    <line x1="${PAD}" x2="${W - PAD}" y1="${AX_Y}" y2="${AX_Y}" stroke="#d3d7dc" stroke-width="2"/>
    <line x1="${basePx}" x2="${basePx}" y1="${(labelY + 8).toFixed(1)}" y2="${AX_Y}"
      stroke="#14181c" stroke-width="3"/>
    ${dots}${axTicks}
    <text x="${(worst.px - 2).toFixed(1)}" y="${labelY}" font-family="ui-monospace,Menlo,monospace"
      font-size="20" fill="#b23b2e">${worst.name} ${worst.v.toFixed(3)}</text>
    <text x="${(basePx - 12).toFixed(1)}" y="${labelY}" font-family="ui-monospace,Menlo,monospace"
      font-size="20" font-weight="700" fill="#14181c" text-anchor="end">counting ${POP.toFixed(3)}</text>
    <text x="${(best.px + 2).toFixed(1)}" y="${labelY}" font-family="ui-monospace,Menlo,monospace"
      font-size="20" fill="#1d6b93" text-anchor="end">${best.name} ${best.v.toFixed(3)}</text>
  </svg>
  <div class="proof">${bootExact
    ? `${nBelow} below in all ${boot.n_boot.toLocaleString("en-US")} bootstrap resamples`
    : `${boot ? boot.n_boot.toLocaleString("en-US") + " bootstrap resamples" : "held-out evaluation"}`}</div>
  <div class="url">incrediblecrab.github.io/llmmm</div>
</body></html>`;

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: W, height: H } });
await page.setContent(html, { waitUntil: "load" });
const out = path.join(root, "public/og.png");
await page.screenshot({ path: out });
await browser.close();

console.log(
  `og.png ${W}x${H} ${(fs.statSync(out).size / 1024).toFixed(0)} KB · ` +
  `${nBelow}/${models.length} below ${POP.toFixed(4)} · ` +
  `worst ${worst.name} ${worst.v.toFixed(4)} · best ${best.name} ${best.v.toFixed(4)}`,
);
