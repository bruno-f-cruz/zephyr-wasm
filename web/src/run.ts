// One video, start to end: decode every frame, keep those on the network's selection
// grid (frame i at i / frameRate, as zephyr predict --frame-rate), crop and encode them
// as they come, then the temporal network over the whole.
import { Cropper, type Geometry } from "./crop";
import { Encoder, predictTrace, type Network, type Prediction } from "./model";
import { decode, type VideoInfo } from "./video";

export interface Progress {
  stage: "decoding" | "predicting";
  done: number;
  total: number;
}

export interface Result extends Prediction {
  frame: Int32Array;
  seconds: { decode: number; predict: number };
}

/** Frames zephyr keeps: nearest each k / select_fs, over floor(duration * fs) + 1. */
export function anchors(frames: number, frameRate: number, selectFs: number): Int32Array {
  const duration = (frames - 1) / frameRate;
  if (selectFs > (frames - 1) / duration) {
    throw new Error(`the video runs below the network's ${selectFs} Hz`);
  }
  const n = Math.floor(duration * selectFs) + 1;
  const out = new Int32Array(n);
  const step = frameRate / selectFs;
  for (let k = 0; k < n; k++) out[k] = Math.min(frames - 1, Math.round(k * step));
  return out;
}

export async function predictVideo(
  video: VideoInfo,
  net: Network,
  g: Geometry,
  frameRate: number,
  onProgress: (p: Progress) => void,
  signal?: AbortSignal,
): Promise<Result> {
  const total = video.samples.length;
  const picks = anchors(total, frameRate, net.meta.select_fs);
  const cropper = new Cropper(g, net.meta.mean, net.meta.std);
  const encoder = new Encoder(net, picks.length);
  const size = g.crop * g.crop;
  const started = performance.now();

  // Frames come out in presentation order; reading a crop is asynchronous, so the
  // crops are chained to keep that order, and the encoder drains between reads.
  let index = 0;
  let next = 0;
  let chain: Promise<void> = Promise.resolve();
  let failure: unknown = null;
  let lastShown = 0;
  const onFrame = (frame: VideoFrame) => {
    const i = index++;
    if (next < picks.length && i === picks[next]) {
      next++;
      const out = new Float32Array(size);
      chain = chain
        .then(() => cropper.gray(frame, out))
        .then(() => encoder.push(out))
        .catch((e) => (failure ??= e))
        .finally(() => frame.close());
    } else frame.close();
    const now = performance.now();
    if (now - lastShown > 200) {
      lastShown = now;
      onProgress({ stage: "decoding", done: index, total });
    }
  };
  const decoding = decode(video, onFrame, 0, total, signal);
  // Encode while decoding: wake every 100 ms to drain the chunks that filled
  let finished = false;
  decoding.finally(() => (finished = true)).catch(() => {});
  while (!finished) {
    await new Promise((r) => setTimeout(r, 100));
    if (failure) throw failure;
    await encoder.drain();
  }
  await decoding;
  await chain;
  if (failure) throw failure;
  await encoder.drain(true);
  onProgress({ stage: "decoding", done: index, total });
  if (index !== total) throw new Error(`decoded ${index} frames, the index lists ${total}`);
  if (encoder.count !== picks.length) {
    throw new Error(`encoded ${encoder.count} of ${picks.length} frames`);
  }
  const decoded = performance.now();

  const times = new Float64Array(picks.length);
  picks.forEach((p, k) => (times[k] = p / frameRate));
  const prediction = await predictTrace(net, encoder.embeddings(), times, (done, all) =>
    onProgress({ stage: "predicting", done, total: all }),
  );
  const frame = new Int32Array(prediction.time.length);
  prediction.time.forEach((t, k) => (frame[k] = Math.round(t * frameRate)));
  return {
    ...prediction,
    frame,
    seconds: { decode: (decoded - started) / 1000, predict: (performance.now() - decoded) / 1000 },
  };
}
