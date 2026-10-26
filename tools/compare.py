"""Compare the web app's prediction parquet with zephyr predict's for the same video.

    uv run --project C:/git/AllenNeuralDynamics/zephyr python tools/compare.py \
        testdata/web/face-60s.prediction.parquet testdata/zephyr/face-60s.prediction.parquet
"""

import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ours, theirs = (pd.read_parquet(p) for p in sys.argv[1:3])
print("metadata:", {k.decode(): v.decode() for k, v in (pq.read_metadata(sys.argv[1]).metadata or {}).items()})
print("columns", list(ours.columns), "rows", len(ours), "vs", len(theirs))
n = min(len(ours), len(theirs))
for column in ("time", "frame", "signal", "onset"):
    d = np.abs(ours[column].to_numpy(float)[:n] - theirs[column].to_numpy(float)[:n])
    print(f"  {column:7s} max |diff| {d.max():.2e}  mean {d.mean():.2e}")
print(f"  signal correlation {np.corrcoef(ours['signal'][:n], theirs['signal'][:n])[0, 1]:.6f}")
a = np.nonzero(ours["inhale"].to_numpy()[:n])[0]
b = np.nonzero(theirs["inhale"].to_numpy()[:n])[0]
near = sum(np.min(np.abs(b - k)) <= 1 for k in a) if len(b) else 0
print(f"  inhales {len(a)} vs {len(b)}: {len(np.intersect1d(a, b))} at the same sample, {near} within one")
