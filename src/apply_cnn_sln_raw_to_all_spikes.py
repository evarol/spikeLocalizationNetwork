"""Apply a trained CNN-SLN (RelativeXYZNet) to all 2.48M spikes and save the
raw predicted localizations (global = anchor + model output). NO motion
correction, NO z override — for apples-to-apples comparison with the
Transformer-SLN raw application.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from relative_xyz_common import RelativeXYZNet


def load_cnn(ckpt_path: Path, device: str = "cpu") -> tuple[RelativeXYZNet, float]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model = RelativeXYZNet()
    model.load_state_dict(state)
    return model.to(device).eval(), float(ckpt["waveform_scale"])


@torch.no_grad()
def predict_batched(model, waveforms_mm, scale: float, device: str,
                    batch: int = 1024) -> np.ndarray:
    n = len(waveforms_mm)
    out = np.zeros((n, 3), dtype=np.float32)
    for i in tqdm(range(0, n, batch), desc="CNN-SLN inference"):
        j = min(i + batch, n)
        W = torch.as_tensor(np.asarray(waveforms_mm[i:j]),
                            dtype=torch.float32, device=device)
        out[i:j] = model(W / scale).cpu().numpy()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--waveforms", default="results/all_spike_waveforms/waveforms_all.npy", type=Path)
    p.add_argument("--anchors", default="results/spike_anchors.npy", type=Path)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=1024)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    wf = np.load(args.waveforms, mmap_mode="r")
    anchors = np.load(args.anchors).astype(np.float32)
    assert len(wf) == len(anchors), f"length mismatch: wf={len(wf)} anchors={len(anchors)}"
    print(f"N={len(wf):,} spikes, waveform shape={wf.shape}, device={args.device}")

    model, scale = load_cnn(args.ckpt, args.device)
    print(f"waveform_scale = {scale:.4f}")

    rel = predict_batched(model, wf, scale, args.device, args.batch)
    print(f"  relative offsets: x∈[{rel[:,0].min():.1f},{rel[:,0].max():.1f}], "
          f"y∈[{rel[:,1].min():.1f},{rel[:,1].max():.1f}], "
          f"z∈[{rel[:,2].min():.1f},{rel[:,2].max():.1f}]")

    GL = anchors.copy()
    GL[:, 0] = anchors[:, 0] + rel[:, 0]
    GL[:, 1] = anchors[:, 1] + rel[:, 1]
    GL[:, 2] = rel[:, 2]    # z anchor=0; raw CNN z prediction used directly (no MP override)

    np.save(args.out_dir / "GL_pre.npy", GL.astype(np.float32))
    np.save(args.out_dir / "x.npy", GL[:, 0].astype(np.float32))
    np.save(args.out_dir / "y.npy", GL[:, 1].astype(np.float32))
    np.save(args.out_dir / "z.npy", GL[:, 2].astype(np.float32))
    print(f"wrote {args.out_dir}/{{GL_pre, x, y, z}}.npy  in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
