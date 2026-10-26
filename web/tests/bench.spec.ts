// Where the browser's time goes on a clip: decoding (all frames, or skipping those no
// frame refers to), and copying the kept frames' luma (one at a time, or concurrently).
import { test } from "@playwright/test";
import { resolve } from "node:path";

const video = resolve(process.env.VIDEO ?? "../testdata/face-60s.mp4");

test("decode throughput", async ({ page }) => {
  page.on("console", (m) => m.type() !== "debug" && console.log(`[browser] ${m.text()}`));
  await page.goto("http://localhost:5179/");
  await page.setInputFiles("#file", video);
  await page.waitForFunction(() => !document.getElementById("step-crop")!.hidden, null, { timeout: 120_000 });
  const report = await page.evaluate(async () => {
    const { openVideo, decode, presentationIndex, unreferenced } = await import("/src/video.ts");
    const file = (document.getElementById("file") as HTMLInputElement).files![0];
    const v = await openVideo(file);
    const n = v.samples.length;
    const { bySample, byTimestamp } = presentationIndex(v);
    const skip = unreferenced(v)!;
    const wanted = (s: number, d: Uint8Array) => bySample[s] % 4 === 0 || !skip(d);
    const rect = { x: 450, y: 216, width: 192, height: 192 };
    const out: Record<string, number> = {};
    const time = async (name: string, copy: "none" | "serial" | "parallel", skipping: boolean, acceleration: HardwareAcceleration = "prefer-hardware") => {
      const t = performance.now();
      let chain: Promise<unknown> = Promise.resolve();
      let busy = 0;
      await decode(v, (f) => {
        const i = byTimestamp.get(f.timestamp)!;
        if (copy === "none" || i % 4) return f.close();
        const buf = new Uint8Array(f.allocationSize({ rect }));
        if (copy === "serial") chain = chain.then(() => f.copyTo(buf, { rect })).finally(() => f.close());
        else {
          busy++;
          f.copyTo(buf, { rect }).finally(() => { f.close(); busy--; });
        }
      }, { wanted: skipping ? wanted : undefined, acceleration });
      await chain;
      while (busy) await new Promise((r) => setTimeout(r, 5));
      out[name] = Math.round(n / ((performance.now() - t) / 1000));
    };
    await time("decode all", "none", false);
    await time("decode skipping", "none", true);
    await time("skipping + serial copies", "serial", true);
    await time("skipping + parallel copies", "parallel", true);
    await time("software: decode skipping", "none", true, "prefer-software");
    await time("software: skipping + parallel copies", "parallel", true, "prefer-software");
    // Several decoders side by side, each on a run of samples from a keyframe
    const keys = v.samples.map((x, k) => (x.sync ? k : -1)).filter((k) => k >= 0);
    for (const pieces of [2, 4, 8]) {
      const cuts = [0];
      for (let q = 1; q < pieces; q++) cuts.push(keys[Math.floor((q * keys.length) / pieces)]);
      cuts.push(n);
      const t = performance.now();
      let busy = 0;
      await Promise.all(cuts.slice(0, -1).map((first, q) => decode(v, (f) => {
        const i = byTimestamp.get(f.timestamp)!;
        if (i % 4) return f.close();
        const buf = new Uint8Array(f.allocationSize({ rect }));
        busy++;
        f.copyTo(buf, { rect }).finally(() => { f.close(); busy--; });
      }, { first, stop: cuts[q + 1], wanted })));
      while (busy) await new Promise((r) => setTimeout(r, 5));
      out[`${pieces} decoders, skipping + copies`] = Math.round(n / ((performance.now() - t) / 1000));
    }
    return out;
  });
  console.log(JSON.stringify(report));
});
