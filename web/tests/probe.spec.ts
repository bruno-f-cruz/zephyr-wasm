import { test } from "@playwright/test";
import { resolve } from "node:path";
test("probe", async ({ page }) => {
  page.on("response", (r) => r.status() >= 400 && console.log("BAD", r.status(), r.url()));
  page.on("console", (m) => console.log(`[browser] ${m.type()}: ${m.text()}`));
  await page.goto(process.env.URL!);
  await page.setInputFiles("#file", resolve("../testdata/face-60s.mp4"));
  await page.waitForTimeout(15000);
});
