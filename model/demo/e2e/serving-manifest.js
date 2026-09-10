import { expect } from "@playwright/test";
import { createHash } from "node:crypto";

export function bindServingManifest(test) {
  let servedManifestHash;
  let hostingBootstrapRemoved = false;

  function originalHtml(bytes) {
    const html = bytes.toString("utf8");
    const bootstrap = /<head><script>window\.huggingface=\{variables:(\{[^\n]*?\})\};<\/script>/;
    const match = html.match(bootstrap);
    if (!match) return bytes;
    const variables = JSON.parse(match[1]);
    expect(Object.values(variables).every((value) => typeof value === "string" && !value.includes("<"))).toBe(true);
    hostingBootstrapRemoved = true;
    return Buffer.from(html.replace(bootstrap, "<head>"), "utf8");
  }

  test.beforeAll(async ({ request }) => {
    const response = await request.get("manifest.json");
    expect(response.ok()).toBe(true);
    const bytes = await response.body();
    servedManifestHash = createHash("sha256").update(bytes).digest("hex");
    const manifest = JSON.parse(bytes.toString("utf8"));
    await Promise.all(Object.entries(manifest.files).map(async ([name, expected]) => {
      const asset = await request.get(name);
      expect(asset.ok(), name).toBe(true);
      const served = await asset.body();
      const body = name === "index.html" ? originalHtml(served) : served;
      expect(body.length, name).toBe(expected.bytes);
      expect(createHash("sha256").update(body).digest("hex"), name).toBe(expected.sha256);
    }));
  });

  test.beforeEach(async ({}, info) => {
    info.annotations.push({ type: "demo-manifest-sha256", description: servedManifestHash });
    info.annotations.push({ type: "static-html-bootstrap-removed", description: String(hostingBootstrapRemoved) });
  });
}
