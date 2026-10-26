"""Export a zephyr checkpoint as the two ONNX graphs the web app runs, plus meta.json.

``encoder.onnx``: normalised gray crops ``(1, 1, T + 2h, H, W)`` (``h`` frames of halo
either side, zeros at a video's ends) -> embeddings ``(T, E)``. ``temporal.onnx``:
embeddings on the output grid ``(1, T_out, E)`` -> signal and onset logits
``(1, T_out)``. The linear resampling between the two grids
(``zephyr.model.resample_embeddings``) runs in the browser.

    uv run --project C:/git/AllenNeuralDynamics/zephyr --with onnx --with onnxruntime \
        python tools/export_onnx.py --checkpoint best.pt --out web/public/models/<name>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from zephyr.evaluate import checkpoint_head_threshold, load_checkpoint
from zephyr.predict import crop_size, read_checkpoint

CHUNK = 256
"""Frames the encoder takes per call (the halo comes on top)."""


class Encoder(nn.Module):
    """The frame encoder on one chunk with its halo, as ``encode_sequence`` runs it."""

    def __init__(self, encoder, crop: tuple[int, int]) -> None:
        super().__init__()
        self.encoder = encoder
        self.halo = encoder.temporal_kernel // 2
        with torch.no_grad():  # the fixed sizes the graph keeps, so pooling is static
            probe = torch.zeros(1, 1, 1 + 2 * self.halo, crop[1], crop[0])
            stem = probe[0].transpose(0, 1) if encoder.stem is None else encoder.stem(probe)
            self.shape = tuple(stem.shape[1:2]) + tuple(stem.shape[-2:])
            trunk = encoder.trunk(torch.zeros(1, *self.shape))
        pool = encoder.pool.output_size
        pool = (pool, pool) if isinstance(pool, int) else pool
        k = (trunk.shape[-2] // pool[0], trunk.shape[-1] // pool[1])
        assert trunk.shape[-2] == k[0] * pool[0] and trunk.shape[-1] == k[1] * pool[1]
        self.pool = nn.AvgPool2d(k, k)  # what AdaptiveAvgPool2d does at this size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.encoder
        if e.stem is None:
            frames = x[0].transpose(0, 1).reshape(-1, *self.shape)  # T, 1, H, W
        else:
            stem = e.stem(x)  # 1, W0, T + 2h, h', w'
            if self.halo:
                stem = stem[:, :, self.halo : -self.halo]
            frames = stem.transpose(1, 2).reshape(-1, *self.shape)
        return e.head(self.pool(e.trunk(frames)))


class Temporal(nn.Module):
    def __init__(self, temporal) -> None:
        super().__init__()
        self.temporal = temporal

    def forward(self, embeddings: torch.Tensor):
        signal, onset = self.temporal(embeddings)[:2]
        return signal, onset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    model, mean, std, _ = load_checkpoint(args.checkpoint, torch.device("cpu"))
    state = read_checkpoint(args.checkpoint)
    crop = crop_size(state)
    recipe = state["feature_config"]
    halo = model.encoder.temporal_kernel // 2
    embed = model.encoder.embed
    args.out.mkdir(parents=True, exist_ok=True)

    encoder = Encoder(model.encoder, crop).eval()
    temporal = Temporal(model.temporal).eval()
    frames = torch.randn(1, 1, CHUNK + 2 * halo, crop[1], crop[0])
    sequence = torch.randn(1, 1024, embed)
    with torch.no_grad():
        torch.onnx.export(
            encoder,
            (frames,),
            args.out / "encoder.onnx",
            input_names=["frames"],
            output_names=["embeddings"],
            dynamic_axes={"frames": {2: "frames"}, "embeddings": {0: "frames"}},
            opset_version=17,
            dynamo=False,
        )
        torch.onnx.export(
            temporal,
            (sequence,),
            args.out / "temporal.onnx",
            input_names=["embeddings"],
            output_names=["signal", "onset"],
            dynamic_axes={
                "embeddings": {1: "samples"},
                "signal": {1: "samples"},
                "onset": {1: "samples"},
            },
            opset_version=17,
            dynamo=False,
        )

        # The graphs give what PyTorch gives, at another length than they were traced at
        x = torch.randn(1, 1, 100 + 2 * halo, crop[1], crop[0])
        s = torch.randn(1, 700, embed)
        want_e = encoder(x).numpy()
        want_s, want_o = (t.numpy() for t in temporal(s))
    got_e = ort.InferenceSession(str(args.out / "encoder.onnx")).run(None, {"frames": x.numpy()})[0]
    got_s, got_o = ort.InferenceSession(str(args.out / "temporal.onnx")).run(
        None, {"embeddings": s.numpy()}
    )
    for name, a, b in (("embeddings", got_e, want_e), ("signal", got_s, want_s), ("onset", got_o, want_o)):
        print(f"{name}: shape {a.shape}, max |onnx - torch| {np.abs(a - b).max():.2e}")

    meta = {
        "checkpoint": f"{args.checkpoint.parent.name}/{args.checkpoint.name}",
        "crop": list(crop),
        "target_size": list(recipe["target_size"]),
        "select_fs": float(recipe["select_fs"]),
        "output_fs": float(recipe["output_fs"]),
        "mean": mean,
        "std": std,
        "halo": halo,
        "embed": embed,
        "chunk": CHUNK,
        "receptive_field": model.receptive_field,
        "window": 1024,
        "head_threshold": checkpoint_head_threshold(state),
        "head_min_distance_s": 0.05,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    index = args.out.parent / "index.json"  # the networks the app offers
    names = json.loads(index.read_text()) if index.is_file() else []
    index.write_text(json.dumps(sorted({*names, args.out.name}), indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
