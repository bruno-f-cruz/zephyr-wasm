"""Does a 2x2 mean of the decoded Y plane give zephyr's crops (scale area, crop, gray)?"""
import subprocess, sys
import numpy as np
from zephyr.video import iter_frames

video, n = sys.argv[1], 600
box = (225, 108, 96, 96)  # on 360x270: native 450, 216, 192 px
ref = np.concatenate(list(iter_frames(video, scale_to=(360, 270), crop=box, frames=n)))
raw = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-frames:v", str(n), "-f", "rawvideo",
                      "-pix_fmt", "yuv420p", "-"], capture_output=True, check=True).stdout
fb = 720 * 540 * 3 // 2
y = np.frombuffer(raw, np.uint8).reshape(n, fb)[:, : 720 * 540].reshape(n, 540, 720)
x, yy = 2 * box[0], 2 * box[1]
roi = y[:, yy : yy + 192, x : x + 192].astype(np.int32)
s = roi.reshape(n, 96, 2, 96, 2).sum(axis=(2, 4))
for name, est in {"round": (s + 2) // 4, "floor": s // 4, "float": s / 4}.items():
    d = est.astype(np.float64) - ref
    print(f"{name:6s} max |diff| {np.abs(d).max():.2f}  mean |diff| {np.abs(d).mean():.4f}  exact {np.mean(d == 0):.4f}")

m = s / 4
a, b = np.polyfit(m.ravel(), ref.ravel().astype(np.float64), 1)
print(f"fit: ref = {a:.4f} * mean + {b:.3f}   (limited->full would be {255/219:.4f}, {-16*255/219:.3f})")
for name, est in {
    "full-range round": np.clip(np.rint((m - 16) * 255 / 219), 0, 255),
    "full-range of rounded mean": np.clip(np.rint(((s + 2) // 4 - 16) * 255 / 219), 0, 255),
}.items():
    d = est - ref
    print(f"{name:28s} max {np.abs(d).max():.1f} mean {np.abs(d).mean():.4f} exact {np.mean(d == 0):.4f} within1 {np.mean(np.abs(d) <= 1):.4f}")

full = (m - 16) * 255 / 219
for name, est in {"floor": np.floor(full), "round-0.5": np.rint(full - 0.5), "floor+0.25": np.floor(full + 0.25), "fit": np.rint(a * m + b)}.items():
    est = np.clip(est, 0, 255); d = est - ref
    print(f"{name:12s} max {np.abs(d).max():.1f} mean {np.abs(d).mean():.4f} exact {np.mean(d == 0):.4f}")
