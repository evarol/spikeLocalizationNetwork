"""Apply a trained SLN checkpoint + DREDge motion estimate to all spikes
whose waveforms are cached. Method-agnostic — pass in checkpoint, motion file,
and waveform cache via CLI.

Outputs two numpy arrays of shape (N, 3) under --out_dir:
  GL_pre_dredge.npy     : SLN forward output + anchor, uncorrected
  GL_post_dredge.npy    : after subtracting the motion field at each spike's (t, y)
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


def load_sln(ckpt_path: Path, device: str = "cpu") -> RelativeXYZNet:
    model = RelativeXYZNet()
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def predict_sln_batched(model, waveforms_mm, scale: float, device: str,
                        batch: int = 1024) -> np.ndarray:
    """waveforms_mm is a memory-mapped (N, 10, 90) float32 array."""
    n = len(waveforms_mm)
    out = np.zeros((n, 3), dtype=np.float32)
    for i in tqdm(range(0, n, batch), desc="SLN inference"):
        j = min(i + batch, n)
        W = torch.as_tensor(np.asarray(waveforms_mm[i:j]),
                            dtype=torch.float32, device=device)
        out[i:j] = model(W / scale).cpu().numpy()
    return out


def apply_motion(GL: np.ndarray, frame_ids: np.ndarray, disp: np.ndarray,
                 y_anchors: np.ndarray, dx_per_bin: np.ndarray | None) -> np.ndarray:
    """Vectorized bilinear interpolation of disp[(t_bin, y)] over y_anchors,
    minus optional 1-D x drift per t_bin."""
    GL = GL.copy()
    T = disp.shape[0]
    fids = np.clip(frame_ids.astype(np.int64), 0, T - 1)

    # Depth-dependent Δy per spike.
    y = GL[:, 1]
    if disp.shape[1] == 1:
        dy = disp[fids, 0]
    else:
        # bilinear in y, lookup in t
        ya = y_anchors
        yi = np.searchsorted(ya, y) - 1
        yi = np.clip(yi, 0, len(ya) - 2)
        ya0 = ya[yi]; ya1 = ya[yi + 1]
        wy = np.clip((y - ya0) / np.maximum(ya1 - ya0, 1e-9), 0.0, 1.0)
        dy = (1 - wy) * disp[fids, yi] + wy * disp[fids, yi + 1]

    GL[:, 1] -= dy

    if dx_per_bin is not None and dx_per_bin.size:
        GL[:, 0] -= dx_per_bin[fids]

    return GL


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path,
                   help="SLN checkpoint .pt (state_dict or wrapped)")
    p.add_argument("--motion", required=True, type=Path,
                   help="DREDge motion .npz with disp, y_anchors, Δx_per_bin")
    p.add_argument("--waveforms", default="results/all_spike_waveforms/waveforms_all.npy", type=Path,
                   help="Cached (N, 10, 90) all-spike waveforms")
    p.add_argument("--channels", default="results/spike_channels.npy", type=Path)
    p.add_argument("--ch_locs", default="results/channel_locations.npy", type=Path)
    p.add_argument("--spike_times", default="results/spike_times.npy", type=Path)
    p.add_argument("--fs_path", default="results/fs.npy", type=Path)
    p.add_argument("--manifest", required=True, type=Path,
                   help="manifest.json with 'waveform_scale' (e.g. dataset_dir/manifest.json)")
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--device", default="cpu",
                   help="cpu | mps | cuda. CPU works fine for batched inference.")
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--monopolar_z", default=None, type=Path,
                   help="If provided, override z column with values from this (N,) "
                        ".npy file (e.g. results/tpca_monopolar_full/spike_locs_z.npy).")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    with open(args.manifest) as f:
        manifest = json.load(f)
    scale = float(manifest["waveform_scale"])
    print(f"waveform_scale = {scale:.4f}  (from {args.manifest})")

    wf = np.load(args.waveforms, mmap_mode="r")
    spike_channels = np.load(args.channels)
    ch_locs = np.load(args.ch_locs).astype(np.float32)
    spike_times = np.load(args.spike_times)
    fs = float(np.load(args.fs_path)[0])

    assert len(wf) == len(spike_channels) == len(spike_times), "length mismatch"
    print(f"N={len(wf):,} spikes, waveform shape={wf.shape}, device={args.device}")

    model = load_sln(args.ckpt, args.device)
    rel = predict_sln_batched(model, wf, scale, args.device, args.batch)

    # Anchors = mean position of 10 nearest channels (precomputed in dataset prep);
    # simpler: just channel position of peak channel. The training used anchors of
    # mean(10 nearest channels) in (x, y, 0). Re-derive to match.
    from relative_xyz_common import build_neighborhood_lookup
    _, anchor_lookup = build_neighborhood_lookup(ch_locs)
    anchors = anchor_lookup[spike_channels.astype(np.int64)]  # (N, 3)
    GL_pre = anchors + rel
    if args.monopolar_z is not None:
        mp_z = np.load(args.monopolar_z).astype(np.float32)
        assert len(mp_z) == len(GL_pre), f"monopolar_z length {len(mp_z)} ≠ N={len(GL_pre)}"
        GL_pre[:, 2] = mp_z
        print(f"  overrode z with monopolar from {args.monopolar_z}")
    print(f"SLN forward done in {time.time()-t0:.1f}s. "
          f"GL_pre y∈[{GL_pre[:,1].min():.1f},{GL_pre[:,1].max():.1f}]")

    motion = dict(np.load(args.motion))
    disp = motion["disp"]
    y_anchors = motion["y_anchors"]
    dx_per_bin = motion.get("Δx_per_bin", motion.get("dx_per_bin"))
    print(f"motion disp shape={disp.shape}, y_anchors={len(y_anchors)}, "
          f"|Δx_per_bin| {'present' if dx_per_bin is not None else 'absent'}")

    frame_ids = np.clip((spike_times.astype(np.float64) / fs).astype(np.int64),
                        0, disp.shape[0] - 1)
    GL_post = apply_motion(GL_pre, frame_ids, disp, y_anchors, dx_per_bin)

    np.save(args.out_dir / "GL_pre_dredge.npy", GL_pre.astype(np.float32))
    np.save(args.out_dir / "GL_post_dredge.npy", GL_post.astype(np.float32))
    # Per-axis 1-D arrays so the method-agnostic visualization drivers can
    # consume them via their --x_path / --y_path / --z_path flags.
    np.save(args.out_dir / "x.npy", GL_post[:, 0].astype(np.float32))
    np.save(args.out_dir / "y.npy", GL_post[:, 1].astype(np.float32))
    np.save(args.out_dir / "z.npy", GL_post[:, 2].astype(np.float32))
    print(f"wrote {args.out_dir}/{{GL_pre_dredge, GL_post_dredge, x, y, z}}.npy "
          f"in total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
