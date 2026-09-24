import { ARRAY_FORMAT, clickableLink, ingredientCatalogMetadata, prepareIngredientCatalog } from "./ingredient-catalog.js";
import { sourceUrl } from "./search.js";

export async function checkedBytes(url, expected, {
  maximum = 128 * 1024 * 1024, progress = null, retryDelays = [500, 1_500],
} = {}) {
  if (!expected || !Number.isSafeInteger(expected.bytes) || expected.bytes < 1 || expected.bytes > maximum
      || !/^[a-f0-9]{64}$/.test(expected.sha256)) {
    throw new Error("An index asset has an invalid size or checksum declaration.");
  }
  for (let attempt = 0; ; attempt += 1) {
    try {
      return await downloadChecked(url, expected, progress);
    } catch (error) {
      // fetch() and body reads reject with TypeError on network failure; HTTP, size and checksum failures are final.
      if (!(error instanceof TypeError)) throw error;
      if (attempt >= retryDelays.length) {
        throw new Error(`Download failed after ${attempt + 1} attempts (network error): ${url.pathname}`);
      }
      await new Promise((resolve) => setTimeout(resolve, retryDelays[attempt]));
    }
  }
}

async function downloadChecked(url, expected, progress) {
  const response = await fetch(url, {
    credentials: "omit", cache: "force-cache", signal: AbortSignal.timeout(120_000),
  });
  if (!response.ok) throw new Error(`Index download failed (HTTP ${response.status}): ${url.pathname}`);
  const reader = response.body.getReader();
  const data = new Uint8Array(expected.bytes);
  let position = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      if (position + value.length > data.length) throw new Error("Index download exceeded its declared byte length.");
      data.set(value, position);
      position += value.length;
      if (progress) progress(position, data.length);
    }
  } finally {
    await reader.cancel();
  }
  const hash = [...new Uint8Array(await crypto.subtle.digest("SHA-256", data))]
    .map((value) => value.toString(16).padStart(2, "0")).join("");
  if (position !== data.length || hash !== expected.sha256) {
    throw new Error("Index asset integrity check failed; no partial index has been loaded.");
  }
  return data;
}

export async function checkedDecompression(compressed, expected) {
  if (typeof DecompressionStream === "undefined") {
    throw new Error("This browser does not support gzip decompression. Use a current browser.");
  }
  if (!Number.isSafeInteger(expected.raw_bytes) || expected.raw_bytes < 1
      || expected.raw_bytes > 1024 * 1024 * 1024
      || !/^[a-f0-9]{64}$/.test(expected.raw_sha256)) {
    throw new Error("An uncompressed index asset has an invalid length or digest.");
  }
  const reader = new Blob([compressed]).stream().pipeThrough(new DecompressionStream("gzip")).getReader();
  const result = new Uint8Array(expected.raw_bytes);
  let position = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      if (position + value.length > result.length) throw new Error("Index decompression exceeded its declared size.");
      result.set(value, position);
      position += value.length;
    }
  } finally {
    await reader.cancel();
  }
  const digest = [...new Uint8Array(await crypto.subtle.digest("SHA-256", result))]
    .map((value) => value.toString(16).padStart(2, "0")).join("");
  if (position !== result.length || digest !== expected.raw_sha256) {
    throw new Error("Uncompressed index integrity check failed.");
  }
  return result.buffer;
}

export async function loadIngredientCatalog(indexUrl, indexRecord, progress = null) {
  const littleEndian = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
  if (!littleEndian) throw new Error("This ingredient index requires a little-endian browser.");
  const bytes = await checkedBytes(indexUrl, indexRecord, { maximum: 4 * 1024 * 1024 });
  const metadata = ingredientCatalogMetadata(JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes)));
  let downloaded = 0;
  const total = Object.values(metadata.arrays).reduce((sum, record) => sum + record.bytes, 0);
  const arrays = {};
  for (const [name, [, , Type]] of Object.entries(ARRAY_FORMAT)) {
    const record = metadata.arrays[name];
    const raw = await checkedBytes(new URL(record.file, indexUrl), record, {
      progress: (current) => { if (progress) progress({ phase: "download", bytes: downloaded + current, total }); },
    });
    const buffer = await checkedDecompression(raw, record);
    arrays[name] = new Type(buffer);
    downloaded += record.bytes;
  }
  if (progress) progress({ phase: "verifying", bytes: downloaded, total });
  return prepareIngredientCatalog(metadata, arrays);
}

export async function loadResultLinks(indexUrl, catalog, matches) {
  const shardIds = [...new Set(matches.map((match) => Math.floor(match.id / catalog.rows_per_url_shard)))];
  const links = new Map();
  async function loadShard(id) {
    const record = catalog.link_shards[id];
    const compressed = await checkedBytes(new URL(record.file, indexUrl), record, { maximum: 16 * 1024 * 1024 });
    const buffer = await checkedDecompression(compressed, record);
    const shard = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(buffer));
    if (shard.first_id !== record.first_id || !Array.isArray(shard.links) || shard.links.length !== record.rows) {
      throw new Error("Recipe links are not aligned with their declared ingredient records.");
    }
    for (const match of matches) {
      const offset = match.id - shard.first_id;
      if (offset < 0 || offset >= shard.links.length) continue;
      const link = shard.links[offset];
      if ((link !== null) !== clickableLink(catalog.data.link_status[match.id])) {
        throw new Error("A result's recipe link differs from its link status.");
      }
      if (link !== null) {
        if (typeof link !== "string" || link.length > 8192) throw new Error("Invalid recipe link value.");
        sourceUrl(link);
      }
      links.set(match.id, link);
    }
  }
  for (let first = 0; first < shardIds.length; first += 4) {
    await Promise.all(shardIds.slice(first, first + 4).map(loadShard));
  }
  if (links.size !== matches.length) throw new Error("Some result links are missing from their expected shards.");
  return matches.map((match) => ({ ...match, link: links.get(match.id) }));
}

const TEXT_SHARD_FIELDS = JSON.stringify(["bytes", "compression", "file", "first_id", "raw_bytes", "raw_sha256", "rows", "sha256"]);
const CONTROL_CHARACTER = /[\u0000-\u001f\u007f-\u009f]/;

function boundedText(value) {
  return typeof value === "string" && value.length > 0 && !CONTROL_CHARACTER.test(value)
    && new TextEncoder().encode(value).length <= 16_384;
}

async function checkedJson(indexUrl, record, compressedLimit, rawLimit) {
  if (!Number.isSafeInteger(record.raw_bytes) || record.raw_bytes > rawLimit) {
    throw new Error("A recipe text file declares an invalid size.");
  }
  const compressed = await checkedBytes(new URL(record.file, indexUrl), record, { maximum: compressedLimit });
  return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(await checkedDecompression(compressed, record)));
}

export async function loadTextManifest(indexUrl, catalog) {
  const record = catalog.text_manifest;
  const manifest = await checkedJson(indexUrl, record, 16 * 1024 * 1024, 64 * 1024 * 1024);
  const size = catalog.rows_per_text_shard;
  if (!manifest || JSON.stringify(Object.keys(manifest)) !== '["shards"]' || !Array.isArray(manifest.shards)
      || manifest.shards.length !== record.shards) {
    throw new Error("The recipe text manifest has an unexpected structure.");
  }
  manifest.shards.forEach((shard, index) => {
    const first = index * size;
    if (!shard || JSON.stringify(Object.keys(shard).sort()) !== TEXT_SHARD_FIELDS
        || shard.file !== `text/${String(index).padStart(4, "0")}.json.gz` || shard.first_id !== first
        || shard.rows !== Math.min(size, catalog.n_recipes - first) || shard.compression !== "gzip") {
      throw new Error("Recipe text shards are not contiguous, complete and safely named.");
    }
  });
  return manifest.shards;
}

export async function loadResultText(indexUrl, catalog, shards, matches) {
  const text = new Map();
  async function loadShard(position) {
    const record = shards[position];
    const shard = await checkedJson(indexUrl, record, 16 * 1024 * 1024, 256 * 1024 * 1024);
    if (!shard || JSON.stringify(Object.keys(shard).sort()) !== '["first_id","ingredient_lines","titles"]'
        || shard.first_id !== record.first_id || !Array.isArray(shard.titles) || shard.titles.length !== record.rows
        || !Array.isArray(shard.ingredient_lines) || shard.ingredient_lines.length !== record.rows) {
      throw new Error("Recipe text is not aligned with its declared ingredient records.");
    }
    for (const match of matches) {
      const offset = match.id - shard.first_id;
      if (offset < 0 || offset >= record.rows) continue;
      const title = shard.titles[offset];
      const lines = shard.ingredient_lines[offset];
      if ((title !== null && !boundedText(title)) || (lines !== null && (!Array.isArray(lines)
          || lines.length < 1 || lines.length > 2_048 || !lines.every(boundedText)))) {
        throw new Error("A result's recipe title or ingredient lines are invalid.");
      }
      text.set(match.id, { title, ingredient_lines: lines });
    }
  }
  const positions = [...new Set(matches.map((match) => Math.floor(match.id / catalog.rows_per_text_shard)))];
  for (let first = 0; first < positions.length; first += 4) {
    await Promise.all(positions.slice(first, first + 4).map(loadShard));
  }
  if (text.size !== matches.length) throw new Error("Some result recipe text is missing from its expected shards.");
  return matches.map((match) => ({ ...match, ...text.get(match.id) }));
}
