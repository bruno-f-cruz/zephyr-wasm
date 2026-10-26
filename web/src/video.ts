// Reading an mp4 in the browser: mp4box parses only its index (the moov box); each
// frame's bytes are then read straight from the File and decoded with WebCodecs, so a
// video of many gigabytes never sits in memory.
import { createFile, DataStream, Endianness } from "mp4box";

export interface Sample {
  offset: number;
  size: number;
  cts: number; // presentation time, in the track's timescale
  dts: number;
  duration: number;
  sync: boolean;
}

export interface VideoInfo {
  file: File;
  width: number;
  height: number;
  codec: string;
  description: Uint8Array | undefined;
  timescale: number;
  samples: Sample[]; // in decode order
  nominalRate: number; // frames per second the container says
}

const HEADER = 16;

/** Offset and size of each top-level box, read from their headers alone. */
async function topBoxes(file: File): Promise<{ type: string; offset: number; size: number }[]> {
  const boxes = [];
  let offset = 0;
  while (offset + 8 <= file.size) {
    const head = new DataView(await file.slice(offset, offset + HEADER).arrayBuffer());
    let size = head.getUint32(0);
    const type = String.fromCharCode(...[4, 5, 6, 7].map((i) => head.getUint8(i)));
    if (size === 1) size = Number(head.getBigUint64(8));
    if (size === 0) size = file.size - offset;
    if (size < 8) throw new Error(`broken mp4 box at byte ${offset}`);
    boxes.push({ type, offset, size });
    offset += size;
  }
  return boxes;
}

/** The video track's codec, size and sample table, from the file's moov box. */
export async function openVideo(file: File): Promise<VideoInfo> {
  const boxes = await topBoxes(file);
  const moov = boxes.find((b) => b.type === "moov");
  if (!moov) throw new Error(`${file.name}: no moov box, so not an mp4 that can be read`);
  const mp4 = createFile();
  let info: any;
  mp4.onReady = (i: any) => (info = i);
  mp4.onError = (e: any) => {
    throw new Error(String(e));
  };
  // The index boxes whole, every other box only its header, so mp4box steps over them
  for (const box of boxes) {
    const whole = box.type === "ftyp" || box.type === "moov";
    const end = whole ? box.offset + box.size : Math.min(box.offset + HEADER, box.offset + box.size);
    const buffer: any = await file.slice(box.offset, end).arrayBuffer();
    buffer.fileStart = box.offset;
    mp4.appendBuffer(buffer);
    if (info) break;
  }
  mp4.flush();
  if (!info) throw new Error(`${file.name}: its index could not be read`);
  const track = info.videoTracks[0];
  if (!track) throw new Error(`${file.name} has no video track`);
  const trak: any = mp4.getTrackById(track.id);
  const samples: Sample[] = trak.samples.map((s: any) => ({
    offset: s.offset,
    size: s.size,
    cts: s.cts,
    dts: s.dts,
    duration: s.duration,
    sync: s.is_sync,
  }));
  const span = samples.reduce((a, s) => Math.max(a, s.cts + s.duration), 0) / track.timescale;
  return {
    file,
    width: track.video.width,
    height: track.video.height,
    codec: track.codec,
    description: description(trak),
    timescale: track.timescale,
    samples,
    nominalRate: samples.length / span,
  };
}

function description(trak: any): Uint8Array | undefined {
  for (const entry of trak.mdia.minf.stbl.stsd.entries) {
    const box = entry.avcC || entry.hvcC || entry.vpcC || entry.av1C;
    if (box) {
      const stream = new DataStream(undefined, 0, Endianness.BIG_ENDIAN);
      box.write(stream);
      return new Uint8Array(stream.buffer, 8); // without the box header
    }
  }
  return undefined;
}

export function decoderConfig(video: VideoInfo): VideoDecoderConfig {
  return {
    codec: video.codec,
    codedWidth: video.width,
    codedHeight: video.height,
    description: video.description,
    hardwareAcceleration: "prefer-hardware",
    optimizeForLatency: false,
  };
}

const READ_BYTES = 8 << 20;
const QUEUE = 48;

/**
 * Decode samples [first, stop) (decode order) and hand every output frame, in
 * presentation order, to *onFrame*, which must close it. Reads the file in runs of
 * about READ_BYTES and keeps at most QUEUE chunks in the decoder.
 */
export async function decode(
  video: VideoInfo,
  onFrame: (frame: VideoFrame) => void,
  first = 0,
  stop = video.samples.length,
  signal?: AbortSignal,
): Promise<void> {
  let failure: unknown = null;
  const decoder = new VideoDecoder({
    output: onFrame,
    error: (e) => (failure = e),
  });
  decoder.configure(decoderConfig(video));
  const { samples, file, timescale } = video;
  let i = first;
  while (i < stop) {
    if (signal?.aborted) throw new DOMException("stopped", "AbortError");
    if (failure) throw failure;
    // One read covering the next run of samples (they lie in order in the file)
    let j = i;
    const start = samples[i].offset;
    let end = start;
    while (j < stop && samples[j].offset - start < READ_BYTES) {
      end = Math.max(end, samples[j].offset + samples[j].size);
      j++;
    }
    const bytes = new Uint8Array(await file.slice(start, end).arrayBuffer());
    for (let k = i; k < j; k++) {
      const s = samples[k];
      while (decoder.decodeQueueSize > QUEUE) {
        await new Promise((resolve) => decoder.addEventListener("dequeue", resolve, { once: true }));
        if (failure) throw failure;
      }
      decoder.decode(
        new EncodedVideoChunk({
          type: s.sync || k === first ? "key" : "delta",
          timestamp: Math.round((1e6 * s.cts) / timescale),
          duration: Math.round((1e6 * s.duration) / timescale),
          data: bytes.subarray(s.offset - start, s.offset - start + s.size),
        }),
      );
    }
    i = j;
  }
  await decoder.flush();
  decoder.close();
  if (failure) throw failure;
}

/** One frame near *seconds* (the keyframe before it, decoded on to it), for the crop. */
export async function frameAt(video: VideoInfo, seconds: number): Promise<VideoFrame> {
  const { samples, timescale } = video;
  const target = seconds * timescale;
  let key = 0;
  for (let k = 0; k < samples.length; k++) {
    if (samples[k].sync && samples[k].cts <= target) key = k;
    if (samples[k].dts > target) break;
  }
  let stop = key + 1;
  while (stop < samples.length && samples[stop].dts <= target + timescale) stop++;
  let best: VideoFrame | null = null;
  await decode(video, (frame) => {
    if (best === null || Math.abs(frame.timestamp - seconds * 1e6) < Math.abs(best.timestamp - seconds * 1e6)) {
      best?.close();
      best = frame;
    } else frame.close();
  }, key, stop);
  if (best === null) throw new Error("no frame decoded");
  return best;
}
