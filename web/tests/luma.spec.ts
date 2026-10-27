// Experiment: what gray does WebGPU's importExternalTexture give for frame 0 of a clip?
// Writes the R, G and B bytes of the whole frame to OUT (raw, W*H*3) for tools/check_gpu_luma.py.
import { test } from "@playwright/test";
import { writeFileSync } from "node:fs";
import { resolve } from "node:path";

const video = resolve(process.env.VIDEO ?? "../testdata/face-60s.mp4");

test("external texture luma", async ({ page }) => {
  page.on("console", (m) => console.log(`[browser] ${m.text()}`));
  await page.route("**/clip.mp4", (r) => r.fulfill({ path: video, contentType: "video/mp4" }));
  await page.goto("http://localhost:5179/");
  const b64 = await page.evaluate(async () => {
    const v = document.createElement("video");
    v.muted = true;
    v.src = "/clip.mp4";
    await new Promise((r) => (v.onloadeddata = r));
    const frame = new VideoFrame(v);
    const { codedWidth: W, codedHeight: H } = frame;
    console.log("frame", frame.format, W, H, JSON.stringify(frame.colorSpace));
    const adapter = (await navigator.gpu.requestAdapter())!;
    const device = await adapter.requestDevice();
    const tex = device.importExternalTexture({ source: frame });
    const module = device.createShaderModule({
      code: `
      @group(0) @binding(0) var t: texture_external;
      @group(0) @binding(1) var<storage, read_write> o: array<u32>;
      @compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3u) {
        if (id.x >= ${W}u || id.y >= ${H}u) { return; }
        let c = textureLoad(t, vec2u(id.x, id.y));
        let q = vec3u(clamp(c.rgb * 255.0 + 0.5, vec3f(0), vec3f(255)));
        o[id.y * ${W}u + id.x] = q.r | (q.g << 8u) | (q.b << 16u);
      }`,
    });
    const pipe = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
    const size = W * H * 4;
    const buf = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const bind = device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: tex }, { binding: 1, resource: { buffer: buf } }],
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe); pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8));
    pass.end();
    enc.copyBufferToBuffer(buf, 0, read, 0, size);
    device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const px = new Uint32Array(read.getMappedRange().slice(0));
    const out = new Uint8Array(W * H * 3);
    for (let i = 0; i < px.length; i++) { out[3*i] = px[i] & 255; out[3*i+1] = (px[i] >> 8) & 255; out[3*i+2] = (px[i] >> 16) & 255; }
    let s = "";
    for (let i = 0; i < out.length; i += 0x8000) s += String.fromCharCode(...out.subarray(i, i + 0x8000));
    return btoa(s);
  });
  writeFileSync(resolve("../testdata/web/gpu-frame0.rgb"), Buffer.from(b64, "base64"));
});
