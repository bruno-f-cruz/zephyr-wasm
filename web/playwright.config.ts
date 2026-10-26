import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "tests",
  timeout: 15 * 60_000,
  use: {
    channel: "msedge", // H.264 decoding (Playwright's own Chromium has none)
    headless: !process.env.HEADED,
    launchOptions: { args: ["--enable-unsafe-webgpu", "--enable-features=WebGPU"] },
    acceptDownloads: true,
  },
  webServer: { command: "npx vite --port 5179 --strictPort", port: 5179, reuseExistingServer: true },
});
