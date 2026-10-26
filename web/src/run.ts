// One video, start to end: decode every frame, keep those on the network's selection
// grid (frame i at i / frameRate, as zephyr predict --frame-rate), crop and encode them
// as they come, then the temporal network over the whole.
import { Cropper, type Geometry } from "./crop";
import { Encoder, predictTrace, type Network, type Prediction } from "./model";
import { decode, presentationIndex, unreferenced, type VideoInfo } from "./video";

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
  acceleration?: HardwareAcceleration,
): Promise<Result> {
  const total = video.samples.length;
  const picks = anchors(total, frameRate, net.meta.select_fs);
  const cropper = new Cropper(g, net.meta.mean, net.meta.std);
  const encoder = new Encoder(net, picks.length);
  const size = g.crop * g.crop;
  const started = performance.now();

  // A frame no other frame refers to is decoded only if it is kept (most of a fast
  // camera's B-frames are skipped). Frames are told apart by their timestamps, so
  // skipped ones leave no gap, and their crops are copied several at a time, then
  // handed to the encoder in order.
  const { bySample, byTimestamp } = presentationIndex(video);
  const kept = new Map<number, number>(); // presentation index -> its pick
  picks.forEach((p, k) => kept.set(p, k));
  const skippable = unreferenced(video);
  const wanted = skippable ? (s: number, data: Uint8Array) => kept.has(bySample[s]) || !skippable(data) : undefined;
  const ready = new Map<number, Float32Array>();
  let next = 0; // the next pick to give the encoder
  let copying = 0; // crops being copied
  let failure: unknown = null;
  let lastShown = 0;
  const onFrame = (frame: VideoFrame) => {
    const i = byTimestamp.get(frame.timestamp);
    const k = i === undefined ? undefined : kept.get(i);
    if (k === undefined) {
      if (i === undefined) failure ??= new Error(`a frame at ${frame.timestamp} us is in no sample`);
      frame.close();
    } else {
      const out = new Float32Array(size);
      copying++;
      cropper
        .gray(frame, out)
        .then(() => {
          ready.set(k, out);
          while (ready.has(next)) {
            encoder.push(ready.get(next)!);
            ready.delete(next++);
          }
        })
        .catch((e) => (failure ??= e))
        .finally(() => {
          frame.close();
          copying--;
        });
    }
    const now = performance.now();
    if (now - lastShown > 200) {
      lastShown = now;
      onProgress({ stage: "decoding", done: next > 0 ? picks[next - 1] + 1 : 0, total });
    }
  };
  const decoding = decode(video, onFrame, { signal, wanted, acceleration });
  // Encode while decoding: wake every 100 ms to drain the chunks that filled
  let finished = false;
  decoding.finally(() => (finished = true)).catch(() => {});
  while (!finished) {
    await new Promise((r) => setTimeout(r, 100));
    if (failure) throw failure;
    await encoder.drain();
  }
  await decoding;
  while (copying > 0) await new Promise((r) => setTimeout(r, 10));
  if (failure) throw failure;
  await encoder.drain(true);
  onProgress({ stage: "decoding", done: total, total });
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
