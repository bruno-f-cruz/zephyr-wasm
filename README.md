# zephyr-wasm

A prototype: [zephyr](https://github.com/AllenNeuralDynamics/zephyr)'s breathing network
running in the browser, on the computer's own GPU. Choose an mp4, place the crop, and get
the predicted breathing as a plot and as a parquet with `zephyr predict`'s columns
(`time`, `frame`, `signal`, `onset`, `inhale`). The video never leaves the machine.

Needs a browser with WebCodecs and WebGPU (a recent Chrome or Edge); without a GPU it
runs on the CPU through WebAssembly (**CPU only**), much more slowly.

## Running it

```bash
cd web
npm install
npm run dev          # http://localhost:5173
```

`npm run build` writes a static site to `web/dist/`; any static host serves it, as long
as it sends the two cross-origin isolation headers in `vite.config.ts` (for threaded
WebAssembly; WebGPU works without them).

## How it works

| Step | zephyr (`zephyr predict --video`) | Here |
| --- | --- | --- |
| Read the video | ffmpeg | mp4box.js reads the index; each frame's bytes are read from the file and decoded with WebCodecs (hardware) |
| Frame times | `--frame-rate` (frame *i* at *i* / rate s) | the same; the container's rate is filled in |
| Pick frames | nearest each *k* / 60 s | the same |
| Crop | scale the frame (area) so the box becomes 96 px, crop, full-range gray | area-average only the box's luma onto 96x96, video range to full range |
| Network | PyTorch | two ONNX graphs on ONNX Runtime Web (WebGPU): the frame encoder on chunks of crops as they decode, then the temporal network in overlapping 1024-sample windows |
| Between the two | `resample_embeddings` | the same linear interpolation, in JavaScript |
| Inhales | the onset head's peaks above the checkpoint's threshold, 50 ms apart | the same (`findPeaks` mirrors `scipy.signal.find_peaks`) |

Only one 128-number embedding per kept frame is held, so an hour of video takes about
130 MB, not the 2.4 GB of crops.

### Matching zephyr

On a 60-second clip of a 240 Hz face video, against `zephyr predict --video` with the same
box and frame rate:

- signal correlation 0.99986;
- 195 of 200 inhales at the same sample, 199 within one.

The rest is the crop's luma: within one gray level of ffmpeg's everywhere (69% exact),
since ffmpeg's area scaling and range conversion round differently.

## The network

For this demo the network is in the repository, `web/public/models/full-gray-all-cv-v4-cv_0`
(zephyr's `full-gray-all-cv-v4` run, fold `cv_0`, 3.4 MB): a cross-validation model, with
its test animals held out, not one trained for use. Later it should be published with
zephyr and loaded from there.

## Adding a network

```bash
uv run --project <zephyr> --with onnx --with onnxruntime python tools/export_onnx.py \
    --checkpoint <run>/best.pt --out web/public/models/<name>
```

writes `encoder.onnx`, `temporal.onnx` and `meta.json` (crop, recipe, normalisation,
onset threshold), checks both graphs against PyTorch, and adds the network to
`web/public/models/index.json`.

## Checking it

- `tools/pipeline.py`: the app's algorithm in numpy and ONNX Runtime, compared with a
  zephyr prediction.
- `tools/check_luma.py`: the crop's gray against ffmpeg's.
- `web/tests/predict.spec.ts` (`npm test`, in Edge): uploads a clip, places the box,
  predicts and downloads the parquet. `VIDEO`, `RATE`, `BOX=x,y,size` and `OUT` choose
  another video.
- `tools/compare.py`: that parquet against zephyr's.
