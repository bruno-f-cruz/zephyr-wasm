"""The web app's algorithm in numpy and ONNX Runtime, to check it against zephyr.

Every step is what ``web/src`` does, written plainly: the luma of each decoded frame
(ffmpeg stands in for WebCodecs), frames picked on a 60 Hz grid by frame rate, the
box's native pixels area-averaged onto the crop and turned to full range, the encoder
in chunks with their halo, linear resampling onto the output grid, the temporal
network in overlapping windows, a z-score, and the onset head's peaks.

    uv run --project <zephyr> --with onnxruntime python \
        tools/pipeline.py testdata/face-60s.mp4 --frame-rate 240.3846 --box 450 215 \
        --models web/public/models/full-gray-all-cv-v4-cv_0 --compare <zephyr.parquet>
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

LIMITED = (255 / 219, -16 * 255 / 219)
"""Video-range luma (16..235) to full range, as ffmpeg's gray conversion does."""


def probe(video: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout
    w, h = out.strip().split(",")
    return int(w), int(h)


def luma_frames(video: Path, size: tuple[int, int]):
    """Each decoded frame's Y plane, in presentation order."""
    w, h = size
    frame = w * h * 3 // 2
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(video), "-fps_mode", "passthrough",
         "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
        stdout=subprocess.PIPE,
    )
    while chunk := proc.stdout.read(frame):
        yield np.frombuffer(chunk, np.uint8)[: w * h].reshape(h, w)
    proc.wait()


def geometry(native, meta, box, box_size):
    """zephyr's working frame and its box: *box_size* native pixels become the crop."""
    crop, target = meta["crop"][0], meta["target_size"]
    size = box_size or round(crop * native[0] / target[0])
    scale = crop / size
    frame = (round(native[0] * scale), round(native[1] * scale))
    s = frame[0] / native[0]
    x = min(max(round(box[0] * s), 0), frame[0] - crop)
    y = min(max(round(box[1] * s), 0), frame[1] - crop)
    return frame, (x, y)


def area_weights(start: int, n: int, ratio: float, limit: int) -> np.ndarray:
    """(n, limit) weights: output pixel i averages native [(start+i)r, (start+i+1)r)."""
    weights = np.zeros((n, limit))
    for i in range(n):
        a, b = (start + i) * ratio, (start + i + 1) * ratio
        for j in range(int(np.floor(a)), min(int(np.ceil(b)), limit)):
            weights[i, j] = max(0.0, min(b, j + 1) - max(a, j))
        weights[i] /= weights[i].sum()
    return weights


def find_peaks(p: np.ndarray, height: float, distance: int) -> np.ndarray:
    """scipy.signal.find_peaks(p, height=, distance=) for a 1-D array."""
    peaks, i, n = [], 1, len(p)
    while i < n - 1:  # local maxima, the middle of a flat top
        if p[i - 1] < p[i]:
            ahead = i + 1
            while ahead < n - 1 and p[ahead] == p[i]:
                ahead += 1
            if p[ahead] < p[i]:
                peaks.append((i + ahead - 1) // 2)
                i = ahead
        i += 1
    peaks = np.array([k for k in peaks if p[k] >= height], dtype=int)
    keep = np.ones(len(peaks), bool)
    for k in np.argsort(p[peaks], kind="stable")[::-1]:  # highest first
        if not keep[k]:
            continue
        j = k - 1
        while j >= 0 and peaks[k] - peaks[j] < distance:
            keep[j] = False
            j -= 1
        j = k + 1
        while j < len(peaks) and peaks[j] - peaks[k] < distance:
            keep[j] = False
            j += 1
    return peaks[keep]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--frame-rate", type=float, required=True)
    parser.add_argument("--box", type=int, nargs=2, required=True)
    parser.add_argument("--box-size", type=int)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    meta = json.loads((args.models / "meta.json").read_text())
    encoder = ort.InferenceSession(str(args.models / "encoder.onnx"))
    temporal = ort.InferenceSession(str(args.models / "temporal.onnx"))

    native = probe(args.video)
    frame, (bx, by) = geometry(native, meta, args.box, args.box_size)
    crop = meta["crop"][0]
    rx, ry = native[0] / frame[0], native[1] / frame[1]
    wx = area_weights(bx, crop, rx, native[0])
    wy = area_weights(by, crop, ry, native[1])
    cols = np.nonzero(wx.any(0))[0]
    rows = np.nonzero(wy.any(0))[0]
    wx, wy = wx[:, cols[0] : cols[-1] + 1], wy[:, rows[0] : rows[-1] + 1]

    # Frames on the selection grid, by frame rate: anchor k is the frame nearest k / fs
    step = args.frame_rate / meta["select_fs"]
    crops, times, k = [], [], 0
    for i, y in enumerate(luma_frames(args.video, native)):
        if i != round(k * step):
            continue
        roi = y[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1].astype(np.float64)
        gray = np.clip(np.rint(LIMITED[0] * (wy @ roi @ wx.T) + LIMITED[1]), 0, 255)
        crops.append(((gray - meta["mean"]) / meta["std"]).astype(np.float32))
        times.append(i / args.frame_rate)
        k += 1
    crops, times = np.stack(crops), np.array(times)
    n = len(crops)

    # Encoder in chunks, each with its halo (zeros past the ends)
    halo, chunk = meta["halo"], meta["chunk"]
    padded = np.concatenate([np.zeros((halo, crop, crop), np.float32), crops,
                             np.zeros((halo, crop, crop), np.float32)])
    embeddings = np.concatenate([
        encoder.run(None, {"frames": padded[s : min(s + chunk, n) + 2 * halo][None, None]})[0]
        for s in range(0, n, chunk)
    ])

    # Output grid and linear resampling onto it
    out_fs = meta["output_fs"]
    grid = times[0] + np.arange(int(np.floor((times[-1] - times[0]) * out_fs)) + 1) / out_fs
    right = np.clip(np.searchsorted(times, grid), 1, n - 1)
    left = right - 1
    w = np.clip((grid - times[left]) / np.maximum(times[right] - times[left], 1e-9), 0, 1)
    resampled = (embeddings[left] + (embeddings[right] - embeddings[left]) * w[:, None])
    resampled = resampled.astype(np.float32)

    # Temporal network in windows, trimming each window's padded edges
    n_out, window = len(grid), min(meta["window"], len(grid))
    margin = min(meta["receptive_field"] // 2, (window - 1) // 2)
    hop = max(1, window - 2 * margin)
    starts = [0] if n_out <= window else list(range(0, n_out - window + 1, hop))
    if starts[-1] != n_out - window:
        starts.append(n_out - window)
    signal, onset = np.zeros(n_out, np.float32), np.zeros(n_out, np.float32)
    for s in starts:
        sig, ons = temporal.run(None, {"embeddings": resampled[None, s : s + window]})
        lo = s + (margin if s > 0 else 0)
        hi = s + window - (margin if s + window < n_out else 0)
        signal[lo:hi] = sig[0, lo - s : hi - s]
        onset[lo:hi] = 1 / (1 + np.exp(-ons[0, lo - s : hi - s]))
    signal = (signal - signal.mean()) / (signal.std() or 1.0)
    signal = (signal - signal.mean()) / (signal.std() or 1.0)  # as predict does again

    distance = max(1, round(meta["head_min_distance_s"] * out_fs))
    inhale = np.zeros(n_out, bool)
    inhale[find_peaks(onset, meta["head_threshold"], distance)] = True
    ours = pd.DataFrame({"time": grid, "frame": np.rint(grid * args.frame_rate).astype(int),
                         "signal": signal, "onset": onset, "inhale": inhale})
    print(f"{n} frames on the selection grid, {n_out} output samples, {inhale.sum()} inhales")
    if args.compare:
        theirs = pd.read_parquet(args.compare)
        print("rows", len(ours), len(theirs))
        for column in ("time", "frame", "signal", "onset"):
            d = np.abs(ours[column].to_numpy(float) - theirs[column].to_numpy(float))
            print(f"  {column:7s} max |diff| {d.max():.2e}  mean {d.mean():.2e}")
        print(f"  signal corr {np.corrcoef(ours['signal'], theirs['signal'])[0, 1]:.6f}")
        a, b = np.nonzero(ours["inhale"])[0], np.nonzero(theirs["inhale"])[0]
        print(f"  inhales {len(a)} vs {len(b)}, the same {len(np.intersect1d(a, b))}")


if __name__ == "__main__":
    main()
