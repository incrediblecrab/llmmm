import { ARRAY_FORMAT, ingredientCatalogMetadata, prepareIngredientCatalog } from "./ingredient-catalog.js";

export async function checkedBytes(url, expected, { maximum = 128 * 1024 * 1024, progress = null } = {}) {
  if (!expected || !Number.isSafeInteger(expected.bytes) || expected.bytes < 1 || expected.bytes > maximum
      || !/^[a-f0-9]{64}$/.test(expected.sha256)) {
    throw new Error("An index asset has an invalid size or checksum declaration.");
  }
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

export async function loadResultUrls(indexUrl, catalog, matches) {
  const shardIds = [...new Set(matches.map((match) => Math.floor(match.id / catalog.rows_per_url_shard)))];
  const urls = new Map();
  async function loadShard(id) {
    const record = catalog.url_shards[id];
    const compressed = await checkedBytes(new URL(record.file, indexUrl), record, { maximum: 16 * 1024 * 1024 });
    const buffer = await checkedDecompression(compressed, record);
    const shard = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(buffer));
    if (shard.first_id !== record.first_id || !Array.isArray(shard.urls) || shard.urls.length !== record.rows) {
      throw new Error("Source URLs are not aligned with their declared ingredient records.");
    }
    for (const match of matches) {
      const offset = match.id - shard.first_id;
      if (offset < 0 || offset >= shard.urls.length) continue;
      const url = shard.urls[offset];
      if (catalog.data.has_source_url[match.id] !== Number(url !== null)) {
        throw new Error("A result's source link differs from its availability flag.");
      }
      if (url !== null) {
        if (typeof url !== "string" || url.length > 4096) throw new Error("Invalid source URL value.");
        const address = new URL(url);
        if (!["http:", "https:"].includes(address.protocol) || address.username || address.password) {
          throw new Error("Source URL is not an ordinary HTTP(S) link.");
        }
      }
      urls.set(match.id, url);
    }
  }
  for (let first = 0; first < shardIds.length; first += 4) {
    await Promise.all(shardIds.slice(first, first + 4).map(loadShard));
  }
  if (urls.size !== matches.length) throw new Error("Some result source URLs are missing from their expected shards.");
  return matches.map((match) => ({ ...match, source_url: urls.get(match.id) }));
}
