// Upload a clip, place the box zephyr used, predict, and keep the parquet for
// tools/compare.py (VIDEO, RATE, BOX, OUT from the environment; defaults: the test clip).
import { expect, test } from "@playwright/test";
import { mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";

const video = resolve(process.env.VIDEO ?? "../testdata/face-60s.mp4");
const rate = process.env.RATE ?? "240.3846";
const [x, y, size] = (process.env.BOX ?? "450,215,192").split(",");
const out = resolve(process.env.OUT ?? "../testdata/web/face-60s.prediction.parquet");
const cpu = process.env.CPU === "1";

test("a video is predicted and downloaded", async ({ page }) => {
  page.on("console", (m) => console.log(`[browser] ${m.type()}: ${m.text()}`));
  await page.goto("http://localhost:5179/");
  await page.setInputFiles("#file", video);
  await expect(page.locator("#step-crop")).toBeVisible({ timeout: 120_000 });
  await page.fill("#rate", rate);
  for (const [id, value] of [["#box-size", size], ["#box-x", x], ["#box-y", y]]) {
    await page.fill(id, value);
    await page.dispatchEvent(id, "change");
  }
  if (cpu) await page.check("#cpu");
  await page.click("#run");
  const progress = setInterval(async () => {
    console.log(await page.locator("#status").textContent().catch(() => ""));
  }, 10_000);
  await expect(page.locator("#step-result")).toBeVisible({ timeout: 14 * 60_000 });
  clearInterval(progress);
  console.log("summary:", await page.locator("#summary").textContent());
  console.log("geometry:", await page.locator("#geometry").textContent());
  const [download] = await Promise.all([page.waitForEvent("download"), page.click("#save")]);
  mkdirSync(dirname(out), { recursive: true });
  await download.saveAs(out);
  await page.screenshot({ path: out.replace(/\.prediction\.parquet$/, ".png"), fullPage: true });
});
