import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { gunzipSync } from "node:zlib";
import { createHash } from "node:crypto";
import { ARRAY_FORMAT, prepareIngredientCatalog, searchIngredientCatalog } from "../ingredient-catalog.js";

const input = JSON.parse(readFileSync(0, "utf8"));
const root = resolve(input.directory);
const metadata = JSON.parse(readFileSync(resolve(root, "ingredient-index.json"), "utf8"));
const arrays = {};
const loadStart = performance.now();
for (const [name, [filename, , Type]] of Object.entries(ARRAY_FORMAT)) {
  const bytes = readFileSync(resolve(root, filename));
  const expected = metadata.arrays[name];
  if (bytes.length !== expected.bytes || createHash("sha256").update(bytes).digest("hex") !== expected.sha256) {
    throw new Error(`Compressed ${name} differs from the manifest.`);
  }
  const raw = gunzipSync(bytes);
  if (raw.length !== expected.raw_bytes || createHash("sha256").update(raw).digest("hex") !== expected.raw_sha256) {
    throw new Error(`Decompressed ${name} differs from the manifest.`);
  }
  arrays[name] = new Type(raw.buffer, raw.byteOffset, raw.length / Type.BYTES_PER_ELEMENT);
}
const catalog = prepareIngredientCatalog(metadata, arrays);
const loadMilliseconds = performance.now() - loadStart;
const results = [];
for (const query of input.queries) {
  for (const ranking of ["learned", "heuristic"]) {
    const result = await searchIngredientCatalog(catalog, input.policy, query, ranking, {
      maxCandidates: input.max_candidates ?? 2000, yieldToEvents: false,
    });
    results.push({ query, ...result });
  }
}
process.stdout.write(JSON.stringify({
  load_ms: loadMilliseconds, records: catalog.n_recipes, slots: catalog.n_slots,
  memory: process.memoryUsage(), results,
}));
