// The gray crop zephyr's preprocessing makes, from a decoded frame: zephyr scales the
// whole frame to its working size (area), crops the box, and keeps full-range luma.
// Here only the box's native pixels are read, area-averaged straight onto the crop.
import type { Meta } from "./model";

export interface Geometry {
  native: [number, number];
  frame: [number, number]; // zephyr's working frame: boxSize native px become the crop
  box: [number, number]; // the crop's corner on that frame
  boxSize: number; // native pixels
  crop: number;
}

/** The share of the frame the network's crops covered, in this video's pixels. */
export function defaultBoxSize(meta: Meta, native: [number, number]): number {
  return Math.round((meta.crop[0] * native[0]) / meta.target_size[0]);
}

/** zephyr predict --video's frame and box for a box at native *corner* of *boxSize*. */
export function geometry(
  meta: Meta,
  native: [number, number],
  corner: [number, number],
  boxSize: number,
): Geometry {
  const crop = meta.crop[0];
  const scale = crop / boxSize;
  const frame: [number, number] = [Math.round(native[0] * scale), Math.round(native[1] * scale)];
  const s = frame[0] / native[0];
  const clamp = (v: number, hi: number) => Math.min(Math.max(Math.round(v * s), 0), hi);
  return {
    native,
    frame,
    box: [clamp(corner[0], frame[0] - crop), clamp(corner[1], frame[1] - crop)],
    boxSize,
    crop,
  };
}

/** Per output pixel, the native pixels it averages and their weights (area). */
interface Axis {
  lo: number; // first native pixel read
  n: number; // native pixels read
  index: Int32Array; // per output pixel and tap: native pixel - lo
  weight: Float32Array;
  taps: number;
}

function axis(start: number, n: number, ratio: number, limit: number): Axis {
  const spans = [];
  let taps = 0;
  for (let i = 0; i < n; i++) {
    const a = (start + i) * ratio;
    const b = (start + i + 1) * ratio;
    const pixels: [number, number][] = [];
    for (let j = Math.floor(a); j < Math.min(Math.ceil(b), limit); j++) {
      const w = Math.min(b, j + 1) - Math.max(a, j);
      if (w > 0) pixels.push([j, w]);
    }
    const total = pixels.reduce((t, [, w]) => t + w, 0);
    spans.push(pixels.map(([j, w]) => [j, w / total] as [number, number]));
    taps = Math.max(taps, pixels.length);
  }
  const lo = Math.min(...spans.map((p) => p[0][0]));
  const hi = Math.max(...spans.map((p) => p[p.length - 1][0]));
  const index = new Int32Array(n * taps);
  const weight = new Float32Array(n * taps);
  spans.forEach((pixels, i) =>
    pixels.forEach(([j, w], t) => {
      index[i * taps + t] = j - lo;
      weight[i * taps + t] = w;
    }),
  );
  return { lo, n: hi - lo + 1, index, weight, taps };
}

const LIMITED = [255 / 219, (-16 * 255) / 219];

export class Cropper {
  readonly x: Axis;
  readonly y: Axis;
  readonly rect: DOMRectInit; // even-aligned native rectangle to copy
  private buffer: Uint8Array | null = null;
  private rows: Float32Array;
  private canvas: OffscreenCanvas | null = null;

  constructor(readonly g: Geometry, readonly mean: number, readonly std: number) {
    const rx = g.native[0] / g.frame[0];
    const ry = g.native[1] / g.frame[1];
    this.x = axis(g.box[0], g.crop, rx, g.native[0]);
    this.y = axis(g.box[1], g.crop, ry, g.native[1]);
    const x0 = this.x.lo & ~1;
    const y0 = this.y.lo & ~1;
    const x1 = Math.min(g.native[0], (this.x.lo + this.x.n + 1) & ~1);
    const y1 = Math.min(g.native[1], (this.y.lo + this.y.n + 1) & ~1);
    this.rect = { x: x0, y: y0, width: x1 - x0, height: y1 - y0 };
    this.rows = new Float32Array(this.y.n * g.crop);
  }

  /**
   * The crop of *frame*, gray 0..255 (into *out*, crop x crop), from its luma when the
   * frame can be read as YUV, else from its RGB as the browser draws it.
   */
  async gray(frame: VideoFrame, out: Float32Array): Promise<void> {
    const r = this.rect as Required<DOMRectInit>;
    let luma: Uint8Array | Uint8ClampedArray;
    let stride: number;
    let step = 1;
    let full = false;
    if (frame.format && /^(I4|NV12|I420)/.test(frame.format)) {
      const size = frame.allocationSize({ rect: r });
      if (!this.buffer || this.buffer.length < size) this.buffer = new Uint8Array(size);
      const layout = await frame.copyTo(this.buffer, { rect: r });
      luma = this.buffer.subarray(layout[0].offset);
      stride = layout[0].stride;
      full = frame.colorSpace?.fullRange === true;
    } else {
      this.canvas ??= new OffscreenCanvas(r.width, r.height);
      const ctx = this.canvas.getContext("2d", { willReadFrequently: true })!;
      ctx.drawImage(frame, r.x, r.y, r.width, r.height, 0, 0, r.width, r.height);
      const rgba = ctx.getImageData(0, 0, r.width, r.height).data;
      const y = new Uint8ClampedArray(r.width * r.height);
      for (let i = 0; i < y.length; i++)
        y[i] = 0.299 * rgba[4 * i] + 0.587 * rgba[4 * i + 1] + 0.114 * rgba[4 * i + 2];
      luma = y;
      stride = r.width;
      step = 1;
      full = true;
    }
    this.resample(luma, stride, step, full, out);
  }

  /** Area-average the copied luma onto the crop, to full range, then normalise. */
  resample(luma: ArrayLike<number>, stride: number, step: number, full: boolean, out: Float32Array) {
    const { x, y, rows } = this;
    const crop = this.g.crop;
    const dx = x.lo - (this.rect.x as number);
    const dy = y.lo - (this.rect.y as number);
    for (let j = 0; j < y.n; j++) {
      const base = (j + dy) * stride;
      for (let i = 0; i < crop; i++) {
        let v = 0;
        for (let t = 0; t < x.taps; t++) {
          const w = x.weight[i * x.taps + t];
          if (w) v += w * luma[base + (x.index[i * x.taps + t] + dx) * step];
        }
        rows[j * crop + i] = v;
      }
    }
    const [a, b] = full ? [1, 0] : LIMITED;
    for (let k = 0; k < crop; k++) {
      for (let i = 0; i < crop; i++) {
        let v = 0;
        for (let t = 0; t < y.taps; t++) {
          const w = y.weight[k * y.taps + t];
          if (w) v += w * rows[y.index[k * y.taps + t] * crop + i];
        }
        const gray = Math.min(255, Math.max(0, Math.round(a * v + b)));
        out[k * crop + i] = (gray - this.mean) / this.std;
      }
    }
  }
}
