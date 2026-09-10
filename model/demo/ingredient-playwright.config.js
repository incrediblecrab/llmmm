import { defineConfig } from "@playwright/test";
import config from "./playwright.config.js";

export default defineConfig({
  ...config,
  testMatch: "**/ingredient-demo.spec.js",
  timeout: 120_000,
});
