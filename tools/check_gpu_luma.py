"""How does WebGPU's importExternalTexture gray (tests/luma.spec.ts) relate to the decoded Y plane?
usage: check_gpu_luma.py clip.mp4 gpu-frame0.rgb"""
import subprocess, sys
import numpy as np

video, rgb_path = sys.argv[1:3]
W, H = 720, 540
raw = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-frames:v", "1", "-f", "rawvideo",
                      "-pix_fmt", "yuv420p", "-"], capture_output=True, check=True).stdout
y = np.frombuffer(raw, np.uint8)[: W * H].reshape(H, W).astype(int)
u = np.frombuffer(raw, np.uint8)[W * H : W * H * 5 // 4]
v = np.frombuffer(raw, np.uint8)[W * H * 5 // 4 :]
print(f"chroma: U {u.min()}..{u.max()}, V {v.min()}..{v.max()}")
rgb = np.fromfile(rgb_path, np.uint8).reshape(-1, W, 3)[:H].astype(int)
print("channels differ:", {c: int(np.abs(rgb[..., c] - rgb[..., 1]).max()) for c in (0, 2)})
ref = np.clip(np.rint((y - 16) * 255 / 219), 0, 255)
d = rgb[..., 1] - ref
print(f"G vs rint(expand(Y)): max {np.abs(d).max():.0f} mean {np.abs(d).mean():.4f} exact {np.mean(d == 0):.4f} within1 {np.mean(np.abs(d) <= 1):.4f}")
for lv in range(16, 236, 20):
    m = y == lv
    if m.any(): print(f"  Y={lv:3d} -> G mean {rgb[..., 1][m].mean():6.2f} (linear {(lv-16)*255/219:6.2f}) n={m.sum()}")
