/* Contrast and small-screen overflow, on every page, in both colour schemes.
 *
 * These two properties were checked by hand four times during the design
 * work, and each pass found something: --ink-faint failing at 4.37 against
 * the sunk panel, and three tables overflowing 390px before they were put in
 * scrolling figures. A property that has broken twice and is invisible in
 * review is one that belongs in the suite rather than in someone's memory.
 *
 * The subtlety worth keeping: a throwaway version of this check read
 * getComputedStyle(el).color and reported zero failures on pages whose charts
 * were entirely unaudited. SVG <text> takes its colour from `fill`, and its
 * `color` is whatever it inherited, so an SVG label can be any contrast at
 * all while the audit reports the ink colour of its ancestor. Both properties
 * are read here, and `fill` wins where the element is inside an <svg>.
 *
 * Serves dist/ itself, in the same manner as demo.smoke.mjs, so the test does
 * not depend on a preview server someone remembered to start.
 */
import { chromium } from "playwright";
import { createServer } from "http";
import { readFile } from "fs/promises";
import { extname, join } from "path";

const PORT = 8101;
const BASE = (process.env.SITE_BASE ?? "/ingredients").replace(/\/+$/, "");

/* The nine pages of the argument. `dist/data/` sits beside them but is the
 * public asset directory, not a route — an earlier version of this list
 * included it and spent every run auditing a 404 page. */
const PAGES = ["/", "/models/", "/native/", "/metrics/", "/trust/",
               "/robustness/", "/bugs/", "/limits/", "/methods/"];

/* WCAG 2.1 AA: 4.5:1 for body text, 3:1 for large text, where large means
 * 24px, or 18.66px when bold. */
const AA_BODY = 4.5;
const AA_LARGE = 3;
const MOBILE = 390;

const TYPES = {
  ".html": "text/html", ".js": "text/javascript", ".css": "text/css",
  ".json": "application/json", ".bin": "application/octet-stream",
  ".svg": "image/svg+xml", ".xml": "application/xml",
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
    res.writeHead(200, { "content-type": TYPES[extname(path)] ?? "text/plain" });
    res.end(body);
  } catch { res.writeHead(404); res.end("not found"); }
});
await new Promise((r) => server.listen(PORT, r));

const relLum = ([r, g, b]) => {
  const f = (c) => { c /= 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
  return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
};
const contrast = (fg, bg) => {
  const [hi, lo] = [relLum(fg), relLum(bg)].sort((a, b) => b - a);
  return (hi + 0.05) / (lo + 0.05);
};

/* Collect every element that owns visible text, with the colour it actually
 * paints in and the first opaque background behind it. */
const collect = (page) => page.evaluate(() => {
  const rgb = (c) => (c.match(/[\d.]+/g) || []).slice(0, 3).map(Number);
  const opaque = (c) => {
    const p = c.match(/[\d.]+/g) || [];
    return c !== "transparent" && p.length && (p[3] === undefined || Number(p[3]) > 0);
  };
  /* SVG has no background of its own, so keep walking out of it into the
     page element that actually paints behind the chart. */
  const bgBehind = (el) => {
    for (let n = el; n; n = n.parentElement) {
      const bg = getComputedStyle(n).backgroundColor;
      if (opaque(bg)) return bg;
    }
    return getComputedStyle(document.body).backgroundColor || "rgb(255,255,255)";
  };

  const out = [];
  for (const el of document.querySelectorAll("body *")) {
    const own = [...el.childNodes]
      .filter((n) => n.nodeType === Node.TEXT_NODE)
      .map((n) => n.textContent.trim()).join("");
    if (own.length < 2) continue;

    const s = getComputedStyle(el);
    if (s.visibility === "hidden" || s.display === "none" || Number(s.opacity) === 0) continue;
    if (!el.getClientRects().length) continue;

    const inSvg = el.ownerSVGElement != null;
    const paint = inSvg && opaque(s.fill) ? s.fill : s.color;
    const size = parseFloat(s.fontSize);
    const weight = parseInt(s.fontWeight, 10) || 400;

    out.push({
      fg: rgb(paint),
      bg: rgb(bgBehind(el)),
      large: size >= 24 || (size >= 18.66 && weight >= 700),
      where: el.tagName.toLowerCase() +
             (el.getAttribute("class") ? "." + el.getAttribute("class").split(/\s+/)[0] : ""),
      text: own.slice(0, 40),
    });
  }
  return out;
});

const browser = await chromium.launch();
const fails = [];
let audited = 0;

for (const scheme of ["light", "dark"]) {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  const page = await ctx.newPage();
  /* Dark is now a stored choice rather than an OS preference, so emulating
     colorScheme would silently test the light palette twice. Set the same
     key the masthead control sets, before the page loads. */
  await ctx.addInitScript((s) => {
    try { window.localStorage.setItem("theme", s); } catch (e) {}
  }, scheme);
  for (const path of PAGES) {
    await page.goto(`http://localhost:${PORT}${BASE}${path}`, { waitUntil: "networkidle" });
    const applied = await page.evaluate(() => document.documentElement.dataset.theme ?? "light");
    if (applied !== scheme) {
      fails.push(`${scheme} ${path} — theme did not apply (got "${applied}")`);
      continue;
    }
    for (const node of await collect(page)) {
      audited++;
      const need = node.large ? AA_LARGE : AA_BODY;
      const got = contrast(node.fg, node.bg);
      /* 0.01 of slack: the ratio is computed from 8-bit channels, and a
         token sitting exactly on 4.50 should not fail on a rounding tail. */
      if (got < need - 0.01) {
        fails.push(`${scheme} ${path} ${node.where} ${got.toFixed(2)} < ${need} — "${node.text}"`);
      }
    }
  }
  await ctx.close();
}

const overflows = [];
const ctx = await browser.newContext({ viewport: { width: MOBILE, height: 844 } });
const page = await ctx.newPage();
for (const path of PAGES) {
  await page.goto(`http://localhost:${PORT}${BASE}${path}`, { waitUntil: "networkidle" });
  const w = await page.evaluate(() => document.documentElement.scrollWidth);
  if (w > MOBILE) overflows.push(`${path} scrollWidth ${w} > ${MOBILE}`);
}
await ctx.close();
await browser.close();
server.close();

const report = (label, list) => {
  if (list.length === 0) { console.log(`  ok    ${label}`); return; }
  console.log(`  FAIL  ${label}`);
  list.slice(0, 20).forEach((f) => console.log(`          ${f}`));
  if (list.length > 20) console.log(`          … and ${list.length - 20} more`);
};

console.log();
report(`contrast meets WCAG AA (${audited} text nodes, light and dark)`, fails);
report(`no horizontal overflow at ${MOBILE}px`, overflows);

if (fails.length || overflows.length) {
  console.log(`\nFAILED — ${fails.length} contrast, ${overflows.length} overflow.`);
  process.exit(1);
}
console.log(`\nOK — ${PAGES.length} pages legible in both schemes and on a phone.`);
