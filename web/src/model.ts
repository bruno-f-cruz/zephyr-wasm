// The network in the browser: two ONNX graphs (tools/export_onnx.py) run by ONNX
// Runtime on the GPU (WebGPU) or, without one, on the CPU (WebAssembly). The encoder
// takes chunks of crops as they are decoded; the temporal network runs once every
// embedding is in, in overlapping windows, as zephyr.infer.predict_clip does.
import * as ort from "onnxruntime-web/webgpu";

export interface Meta {
  crop: [number, number];
  target_size: [number, number];
  select_fs: number;
  output_fs: number;
  mean: number;
  std: number;
  halo: number;
  embed: number;
  chunk: number;
  receptive_field: number;
  window: number;
  head_threshold: number;
  head_min_distance_s: number;
}

export interface Network {
  meta: Meta;
  encoder: ort.InferenceSession;
  temporal: ort.InferenceSession;
  provider: string;
}

export async function loadNetwork(url: string, prefer: "webgpu" | "wasm" = "webgpu"): Promise<Network> {
  const meta: Meta = await (await fetch(`${url}/meta.json`)).json();
  const tries = prefer === "webgpu" && "gpu" in navigator ? ["webgpu", "wasm"] : ["wasm"];
  let last: unknown;
  for (const provider of tries) {
    try {
      const options = { executionProviders: [provider], graphOptimizationLevel: "all" as const };
      const encoder = await ort.InferenceSession.create(`${url}/encoder.onnx`, options);
      const temporal = await ort.InferenceSession.create(`${url}/temporal.onnx`, options);
      return { meta, encoder, temporal, provider };
    } catch (e) {
      last = e;
    }
  }
  throw last;
}

/** Runs the encoder on crops as they come, each chunk with its halo. */
export class Encoder {
  private frames: Float32Array[] = [];
  private zeros: Float32Array;
  private out: Float32Array;
  count = 0; // embeddings made
  private started = false;

  constructor(private net: Network, capacity: number) {
    const { crop, embed, halo } = net.meta;
    this.zeros = new Float32Array(crop[0] * crop[1]);
    this.out = new Float32Array(capacity * embed);
    for (let i = 0; i < halo; i++) this.frames.push(this.zeros);
  }

  get pending(): number {
    return this.frames.length;
  }

  push(crop: Float32Array) {
    this.frames.push(crop);
    this.started = true;
  }

  /** Encode every full chunk waiting (all of them, padded with zeros, when *last*). */
  async drain(last = false): Promise<void> {
    const { chunk, halo } = this.net.meta;
    if (last && this.started) for (let i = 0; i < halo; i++) this.frames.push(this.zeros);
    while (this.frames.length >= chunk + 2 * halo || (last && this.frames.length > 2 * halo)) {
      const take = Math.min(chunk, this.frames.length - 2 * halo);
      await this.run(this.frames.slice(0, take + 2 * halo));
      this.frames = this.frames.slice(take);
    }
  }

  private async run(frames: Float32Array[]) {
    const { crop, embed } = this.net.meta;
    const size = crop[0] * crop[1];
    const data = new Float32Array(frames.length * size);
    frames.forEach((f, i) => data.set(f, i * size));
    const input = new ort.Tensor("float32", data, [1, 1, frames.length, crop[1], crop[0]]);
    const { embeddings } = await this.net.encoder.run({ frames: input });
    const values = embeddings.data as Float32Array;
    this.out.set(values, this.count * embed);
    this.count += values.length / embed;
    input.dispose?.();
    embeddings.dispose?.();
  }

  embeddings(): Float32Array {
    return this.out.subarray(0, this.count * this.net.meta.embed);
  }
}

export interface Prediction {
  time: Float64Array;
  signal: Float32Array;
  onset: Float32Array;
  inhale: Uint8Array;
}

/** Embeddings at *times* (selection grid) -> the trace on the output grid. */
export async function predictTrace(
  net: Network,
  embeddings: Float32Array,
  times: Float64Array,
  onWindow?: (done: number, total: number) => void,
): Promise<Prediction> {
  const { embed, output_fs, receptive_field } = net.meta;
  const n = times.length;
  const nOut = Math.floor((times[n - 1] - times[0]) * output_fs) + 1;
  const grid = new Float64Array(nOut);
  for (let i = 0; i < nOut; i++) grid[i] = times[0] + i / output_fs;

  // Linear resampling onto the output grid (zephyr.model.resample_embeddings)
  const resampled = new Float32Array(nOut * embed);
  let right = 1;
  for (let i = 0; i < nOut; i++) {
    while (right < n - 1 && times[right] < grid[i]) right++;
    const left = right - 1;
    const span = Math.max(times[right] - times[left], 1e-9);
    const w = Math.min(1, Math.max(0, (grid[i] - times[left]) / span));
    for (let e = 0; e < embed; e++) {
      const lo = embeddings[left * embed + e];
      resampled[i * embed + e] = lo + (embeddings[right * embed + e] - lo) * w;
    }
  }

  // The temporal network in windows, each window's padded edges trimmed
  const window = Math.min(net.meta.window, nOut);
  const margin = Math.min(Math.floor(receptive_field / 2), Math.floor((window - 1) / 2));
  const hop = Math.max(1, window - 2 * margin);
  const starts: number[] = [];
  if (nOut <= window) starts.push(0);
  else {
    for (let s = 0; s <= nOut - window; s += hop) starts.push(s);
    if (starts[starts.length - 1] !== nOut - window) starts.push(nOut - window);
  }
  const signal = new Float32Array(nOut);
  const onset = new Float32Array(nOut);
  for (const [k, s] of starts.entries()) {
    const input = new ort.Tensor(
      "float32",
      resampled.slice(s * embed, (s + window) * embed),
      [1, window, embed],
    );
    const out = await net.temporal.run({ embeddings: input });
    const sig = out.signal.data as Float32Array;
    const ons = out.onset.data as Float32Array;
    const lo = s + (s > 0 ? margin : 0);
    const hi = s + window - (s + window < nOut ? margin : 0);
    for (let i = lo; i < hi; i++) {
      signal[i] = sig[i - s];
      onset[i] = 1 / (1 + Math.exp(-ons[i - s]));
    }
    input.dispose?.();
    out.signal.dispose?.();
    out.onset.dispose?.();
    onWindow?.(k + 1, starts.length);
  }
  zscore(signal);
  zscore(signal); // zephyr predict z-scores the stitched trace again

  const distance = Math.max(1, Math.round(net.meta.head_min_distance_s * output_fs));
  const inhale = new Uint8Array(nOut);
  for (const p of findPeaks(onset, net.meta.head_threshold, distance)) inhale[p] = 1;
  return { time: grid, signal, onset, inhale };
}

function zscore(x: Float32Array) {
  let mean = 0;
  for (const v of x) mean += v;
  mean /= x.length;
  let sd = 0;
  for (const v of x) sd += (v - mean) ** 2;
  sd = Math.sqrt(sd / x.length) || 1;
  for (let i = 0; i < x.length; i++) x[i] = (x[i] - mean) / sd;
}

/** scipy.signal.find_peaks(p, height=, distance=) on a 1-D array. */
export function findPeaks(p: ArrayLike<number>, height: number, distance: number): number[] {
  const peaks: number[] = [];
  const n = p.length;
  let i = 1;
  while (i < n - 1) {
    if (p[i - 1] < p[i]) {
      let ahead = i + 1;
      while (ahead < n - 1 && p[ahead] === p[i]) ahead++;
      if (p[ahead] < p[i]) {
        peaks.push(Math.floor((i + ahead - 1) / 2));
        i = ahead;
      }
    }
    i++;
  }
  const tall = peaks.filter((k) => p[k] >= height);
  const keep = new Array(tall.length).fill(true);
  const order = tall.map((_, k) => k).sort((a, b) => p[tall[b]] - p[tall[a]] || b - a);
  for (const k of order) {
    if (!keep[k]) continue;
    for (let j = k - 1; j >= 0 && tall[k] - tall[j] < distance; j--) keep[j] = false;
    for (let j = k + 1; j < tall.length && tall[j] - tall[k] < distance; j++) keep[j] = false;
  }
  return tall.filter((_, k) => keep[k]);
}
