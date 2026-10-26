import { Cropper, defaultBoxSize, geometry, type Geometry } from "./crop";
import { loadNetwork, type Network } from "./model";
import { download, Plots, rates, toParquet } from "./output";
import { predictVideo, type Result } from "./run";
import { frameAt, openVideo, type VideoInfo } from "./video";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const show = (id: string, on = true) => ($(id).hidden = !on);

let video: VideoInfo | null = null;
let preview: VideoFrame | null = null;
let corner: [number, number] = [0, 0];
let result: Result | null = null;
let stopper: AbortController | null = null;
const networks = new Map<string, Network>();
const plots = new Plots($("overview"), $("detail"));
const BASE = import.meta.env.BASE_URL;

function fail(e: unknown) {
  const text = e instanceof Error ? e.message : String(e);
  $("error").textContent = text;
  show("error");
  console.error(e);
}

function clock(seconds: number): string {
  const m = Math.floor(seconds / 60);
  return `${m}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
}

async function listModels() {
  const names: string[] = await (await fetch(`${BASE}models/index.json`)).json();
  $<HTMLSelectElement>("model").replaceChildren(
    ...names.map((n) => Object.assign(document.createElement("option"), { value: n, textContent: n })),
  );
}

async function network(): Promise<Network> {
  await modelsListed;
  const name = $<HTMLSelectElement>("model").value;
  const cpu = $<HTMLInputElement>("cpu").checked;
  const key = `${name}:${cpu}`;
  if (!networks.has(key)) networks.set(key, await loadNetwork(`${BASE}models/${name}`, cpu ? "wasm" : "webgpu"));
  return networks.get(key)!;
}

async function choose(file: File) {
  show("error", false);
  show("step-result", false);
  $("drop-text").textContent = `Reading ${file.name}…`;
  try {
    video = await openVideo(file);
  } catch (e) {
    $("drop-text").textContent = "Choose an mp4, or drop it here";
    return fail(e);
  }
  const support = await VideoDecoder.isConfigSupported({
    codec: video.codec,
    codedWidth: video.width,
    codedHeight: video.height,
    description: video.description,
  });
  if (!support.supported) return fail(`this browser cannot decode ${video.codec}`);
  const frames = video.samples.length;
  const duration = frames / video.nominalRate;
  $("drop-text").textContent = file.name;
  const facts = {
    Size: `${video.width} × ${video.height}`,
    Frames: frames.toLocaleString(),
    "Container rate": `${video.nominalRate.toFixed(3)} Hz`,
    Length: clock(duration),
    Codec: video.codec,
    File: `${(file.size / 1e9).toFixed(2)} GB`,
  };
  $("video-info").replaceChildren(
    ...Object.entries(facts).map(([k, v]) => {
      const div = document.createElement("div");
      div.innerHTML = `<dt>${k}</dt><dd></dd>`;
      div.querySelector("dd")!.textContent = v;
      return div;
    }),
  );
  show("video-info");
  show("rate-row");
  $<HTMLInputElement>("rate").value = String(+video.nominalRate.toFixed(4));

  const meta = (await network()).meta;
  const size = defaultBoxSize(meta, [video.width, video.height]);
  $<HTMLInputElement>("box-size").value = String(size);
  corner = [Math.round((video.width - size) / 2), Math.round((video.height - size) / 2)];
  preview?.close();
  preview = await frameAt(video, Math.min(150, duration / 2));
  const canvas = $<HTMLCanvasElement>("frame");
  canvas.width = video.width;
  canvas.height = video.height;
  show("step-crop");
  show("step-run");
  await drawCrop();
}

function currentGeometry(net: Network): Geometry {
  const size = Number($<HTMLInputElement>("box-size").value);
  return geometry(net.meta, [video!.width, video!.height], corner, size);
}

async function drawCrop() {
  if (!video || !preview) return;
  const net = await network();
  const size = Math.max(8, Math.min(Number($<HTMLInputElement>("box-size").value), video.width, video.height));
  corner = [
    Math.min(Math.max(0, corner[0]), video.width - size),
    Math.min(Math.max(0, corner[1]), video.height - size),
  ];
  $<HTMLInputElement>("box-x").value = String(corner[0]);
  $<HTMLInputElement>("box-y").value = String(corner[1]);
  const g = currentGeometry(net);
  const ctx = $<HTMLCanvasElement>("frame").getContext("2d")!;
  ctx.drawImage(preview, 0, 0);
  ctx.strokeStyle = "#ffcf33";
  ctx.lineWidth = Math.max(2, video.width / 320);
  ctx.strokeRect(corner[0], corner[1], size, size);
  $("geometry").textContent =
    `The video is scaled to ${g.frame[0]} × ${g.frame[1]} and cropped at ` +
    `(${g.box[0]}, ${g.box[1]}): ${size} px become the network's ${g.crop} px.`;

  const crop = new Float32Array(g.crop * g.crop);
  await new Cropper(g, 0, 1).gray(preview, crop); // unnormalised: gray 0..255
  const image = new ImageData(g.crop, g.crop);
  crop.forEach((v, i) => image.data.set([v, v, v, 255], 4 * i));
  const canvas = $<HTMLCanvasElement>("crop-preview");
  canvas.width = canvas.height = g.crop;
  canvas.getContext("2d")!.putImageData(image, 0, 0);
}

async function run() {
  if (!video) return;
  show("error", false);
  show("step-result", false);
  $<HTMLButtonElement>("run").disabled = true;
  show("stop");
  show("progress");
  stopper = new AbortController();
  const bar = $<HTMLProgressElement>("bar");
  const status = $("status");
  try {
    const net = await network();
    const rate = Number($<HTMLInputElement>("rate").value);
    if (!(rate > 0)) throw new Error("give the video's frame rate");
    const started = performance.now();
    let stageStart = started;
    let stage = "";
    result = await predictVideo(video, net, currentGeometry(net), rate, (p) => {
      const now = performance.now();
      if (p.stage !== stage) {
        stage = p.stage;
        stageStart = now;
      }
      const elapsed = (now - stageStart) / 1000;
      const speed = p.done / Math.max(elapsed, 1e-3);
      const left = (p.total - p.done) / Math.max(speed, 1e-9);
      bar.value = p.done / p.total;
      const unit = p.stage === "decoding" ? "frames" : "windows";
      status.textContent =
        `${p.stage === "decoding" ? "Decoding and encoding" : "Predicting"} on ${net.provider === "webgpu" ? "the GPU (WebGPU)" : "the CPU (WebAssembly)"}: ` +
        `${p.done.toLocaleString()} / ${p.total.toLocaleString()} ${unit} · ` +
        `${Math.round(speed).toLocaleString()} ${unit}/s · ${clock(elapsed)} elapsed, ${clock(left)} left`;
    }, stopper.signal);
    const total = (performance.now() - started) / 1000;
    const { hz } = rates(result);
    const median = hz.length ? [...hz].sort((a, b) => a - b)[Math.floor(hz.length / 2)] : NaN;
    const inhales = result.inhale.reduce((a, v) => a + v, 0);
    $("summary").textContent =
      `${inhales.toLocaleString()} inhales over ${clock(result.time[result.time.length - 1])}, ` +
      `median ${median.toFixed(2)} breaths/s · ${clock(total)} on ${net.provider} ` +
      `(decode ${clock(result.seconds.decode)}, network ${clock(result.seconds.predict)})`;
    status.textContent = `Done in ${clock(total)}.`;
    bar.value = 1;
    show("step-result");
    plots.show(result);
    $<HTMLInputElement>("scrub").max = String(result.time[result.time.length - 1]);
  } catch (e) {
    if ((e as Error).name === "AbortError") status.textContent = "Stopped.";
    else fail(e);
  } finally {
    $<HTMLButtonElement>("run").disabled = false;
    show("stop", false);
    stopper = null;
  }
}

async function save() {
  if (!result || !video) return;
  const net = $<HTMLSelectElement>("model").value;
  const g = currentGeometry(await network());
  const buffer = toParquet(result, {
    video: video.file.name,
    network: net,
    frame_rate: $<HTMLInputElement>("rate").value,
    box_native: JSON.stringify([...corner, g.boxSize]),
    frame: JSON.stringify(g.frame),
    box: JSON.stringify([...g.box, g.crop, g.crop]),
    made_by: "zephyr-wasm",
  });
  const stem = video.file.name.replace(/\.[^.]+$/, "");
  download(buffer, `${stem}.prediction.parquet`);
}

$<HTMLInputElement>("file").addEventListener("change", (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (file) choose(file);
});
const drop = $("drop");
drop.addEventListener("dragover", (e) => {
  e.preventDefault();
  drop.classList.add("over");
});
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  const file = e.dataTransfer?.files[0];
  if (file) choose(file);
});
$<HTMLCanvasElement>("frame").addEventListener("click", (e) => {
  if (!video) return;
  const canvas = e.target as HTMLCanvasElement;
  const r = canvas.getBoundingClientRect();
  const size = Number($<HTMLInputElement>("box-size").value);
  const x = ((e.clientX - r.left) / r.width) * video.width;
  const y = ((e.clientY - r.top) / r.height) * video.height;
  corner = [Math.round(x - size / 2), Math.round(y - size / 2)];
  drawCrop();
});
for (const id of ["box-size", "box-x", "box-y"]) {
  $<HTMLInputElement>(id).addEventListener("change", () => {
    if (id !== "box-size") corner = [Number($<HTMLInputElement>("box-x").value), Number($<HTMLInputElement>("box-y").value)];
    drawCrop();
  });
}
$("run").addEventListener("click", run);
$("stop").addEventListener("click", () => stopper?.abort());
$("save").addEventListener("click", save);
$<HTMLInputElement>("scrub").addEventListener("input", (e) => plots.goTo(Number((e.target as HTMLInputElement).value)));
$<HTMLSelectElement>("span").addEventListener("change", (e) => {
  plots.span = Number((e.target as HTMLSelectElement).value);
  plots.goTo(Number($<HTMLInputElement>("scrub").value));
});
$<HTMLSelectElement>("model").addEventListener("change", () => drawCrop());

const modelsListed = listModels();
modelsListed.catch(fail);
if (!("VideoDecoder" in window)) fail("this browser has no WebCodecs: use a recent Chrome or Edge");
