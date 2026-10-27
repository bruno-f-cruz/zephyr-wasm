// The crop on the GPU: the decoded frame is imported as an external texture (the browser
// turns its YUV into RGB; for a gray video that is the 16..255 range expanded, to within
// one gray level of zephyr's), a compute shader area-averages the box onto the crop and
// normalises it, and the encoder is handed GPU buffers, so no pixel returns to the CPU.
import type { Cropper } from "./crop";

const SHADER = /* wgsl */ `
struct P { crop: u32, tx: u32, ty: u32, mean: f32, scale: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> xi: array<u32>;
@group(0) @binding(2) var<storage, read> xw: array<f32>;
@group(0) @binding(3) var<storage, read> yi: array<u32>;
@group(0) @binding(4) var<storage, read> yw: array<f32>;
@group(1) @binding(0) var t: texture_external;
@group(1) @binding(1) var<storage, read_write> o: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= p.crop || id.y >= p.crop) { return; }
  var v = 0.0;
  for (var b = 0u; b < p.ty; b++) {
    let wy = yw[id.y * p.ty + b];
    if (wy == 0.0) { continue; }
    let row = yi[id.y * p.ty + b];
    for (var a = 0u; a < p.tx; a++) {
      let wx = xw[id.x * p.tx + a];
      if (wx == 0.0) { continue; }
      let c = textureLoad(t, vec2u(xi[id.x * p.tx + a], row)).rgb;
      v += wy * wx * dot(c, vec3f(0.299, 0.587, 0.114));
    }
  }
  let gray = clamp(round(v * 255.0), 0.0, 255.0);
  o[id.y * p.crop + id.x] = (gray - p.mean) / p.scale;
}`;

function upload(device: GPUDevice, data: Float32Array | Uint32Array | Uint32Array, usage: number): GPUBuffer {
  const buffer = device.createBuffer({ size: Math.max(16, data.byteLength), usage: usage | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buffer, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
  return buffer;
}

export class GpuCrops {
  readonly bytes: number;
  readonly zeros: GPUBuffer;
  private pipeline: GPUComputePipeline;
  private fixed: GPUBindGroup;
  private pool: GPUBuffer[] = [];
  private crop: number;

  constructor(readonly device: GPUDevice, cropper: Cropper, mean: number, std: number) {
    const { x, y, g } = cropper;
    this.crop = g.crop;
    this.bytes = g.crop * g.crop * 4;
    this.pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: SHADER }), entryPoint: "main" },
    });
    const storage = GPUBufferUsage.STORAGE;
    const absolute = (a: typeof x) => Uint32Array.from(a.index, (i) => i + a.lo);
    const params = new ArrayBuffer(32);
    new Uint32Array(params, 0, 3).set([g.crop, x.taps, y.taps]);
    new Float32Array(params, 12, 2).set([mean, std]);
    this.fixed = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        upload(device, new Uint32Array(params), GPUBufferUsage.UNIFORM),
        upload(device, absolute(x), storage),
        upload(device, x.weight, storage),
        upload(device, absolute(y), storage),
        upload(device, y.weight, storage),
      ].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    this.zeros = this.take();
    // normalised black: what zephyr pads the halo with is zero, in normalised units
    device.queue.writeBuffer(this.zeros, 0, new Float32Array(this.crop * this.crop));
  }

  private take(): GPUBuffer {
    return (
      this.pool.pop() ??
      this.device.createBuffer({
        size: this.bytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      })
    );
  }

  /** The normalised crop of *frame* into a buffer; the frame is free once this returns' work is done. */
  async cropFrame(frame: VideoFrame): Promise<GPUBuffer> {
    const out = this.take();
    const texture = this.device.importExternalTexture({ source: frame });
    const pass_ = this.device.createCommandEncoder();
    const pass = pass_.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.fixed);
    pass.setBindGroup(
      1,
      this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(1),
        entries: [
          { binding: 0, resource: texture },
          { binding: 1, resource: { buffer: out } },
        ],
      }),
    );
    pass.dispatchWorkgroups(Math.ceil(this.crop / 8), Math.ceil(this.crop / 8));
    pass.end();
    this.device.queue.submit([pass_.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    return out;
  }

  /** The crops, in order, as one buffer for the encoder's input. */
  chunk(frames: GPUBuffer[]): GPUBuffer {
    const buffer = this.device.createBuffer({
      size: frames.length * this.bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const enc = this.device.createCommandEncoder();
    frames.forEach((f, i) => enc.copyBufferToBuffer(f, 0, buffer, i * this.bytes, this.bytes));
    this.device.queue.submit([enc.finish()]);
    return buffer;
  }

  release(frames: GPUBuffer[]) {
    for (const f of frames) if (f !== this.zeros) this.pool.push(f);
  }
}
