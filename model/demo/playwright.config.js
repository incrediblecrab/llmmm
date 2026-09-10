import { defineConfig } from "@playwright/test";
import { basename, dirname, extname, join } from "node:path";

const report = process.env.LLMMM_DEMO_REPORT;

export default defineConfig({
  testDir: "./e2e",
  testMatch: "**/demo.spec.js",
  outputDir: report
    ? join(dirname(report), `${basename(report, extname(report))}-artifacts`) : "test-results",
  timeout: 30_000,
  workers: 1,
  retries: 0,
  reporter: [
    ["list"],
    ["json", { outputFile: process.env.LLMMM_DEMO_REPORT || "test-results/browser-report.json" }],
  ],
  use: {
    baseURL: process.env.LLMMM_DEMO_URL || "http://127.0.0.1:7860",
    browserName: "chromium",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    { name: "desktop", use: { viewport: { width: 1360, height: 1000 } } },
    { name: "mobile", use: { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true } },
  ],
});
