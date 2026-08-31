// @ts-check
import { defineConfig } from "astro/config";

import sitemap from "@astrojs/sitemap";

/**
 * A GitHub project page is served from `https://<user>.github.io/<repo>/`, so
 * every absolute path on the site has to carry that prefix. A user page or a
 * custom domain serves from the root and must not. Getting this wrong does not
 * fail the build — it produces a site whose links 404 and whose demo silently
 * cannot fetch its own 6 MB matrix, which is the worst kind of bug to ship
 * because the page still renders.
 *
 * So it is one environment variable, read in one place, and everything else
 * derives from `import.meta.env.BASE_URL` through the `href()` helper in
 * `src/lib/data.ts`. Local builds default to the project-page layout, because
 * that is the case with a prefix and therefore the case that breaks if it is
 * never exercised.
 *
 *   SITE_BASE=/llmmm        default — project page
 *   SITE_BASE=/              user page or custom domain
 */
const base = process.env.SITE_BASE ?? "/llmmm";

export default defineConfig({
  base,
  site: process.env.SITE_URL ?? "https://incrediblecrab.github.io/llmmm",

  build: {
    // Directory routes, so /models/ is a real path rather than /models.html.
    format: "directory",
  },

  vite: {
    // Nine largely static documents. Inlining the small stylesheets saves a
    // round trip that matters more here than separate caching would.
    build: { assetsInlineLimit: 4096 },
  },

  integrations: [sitemap()],
});