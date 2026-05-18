"""Generate a single depth-raster visualization (method-agnostic).

Inputs:
  --t_path      spike times (samples), int array       [default: results/spike_times.npy]
  --y_path      per-spike depth y (μm)                 [default: results/spike_locs_y.npy]
  --alpha_path  per-spike amplitude α (μV)             [default: results/spike_locs_alpha.npy]
  --fs_path     sampling rate (Hz)                     [default: results/fs.npy]
  --ch_path     channel locations (C, 2) in μm         [default: results/channel_locations.npy]

The visualization is method-agnostic: pass the y array for whatever method you
want (raw monopolar, MP+DREDge corrected, SLN+DREDge corrected, etc.).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from depth_raster import DepthRasterStyle, percentile_clip, save_single_panel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--t_path", default="results/spike_times.npy", type=Path)
    p.add_argument("--y_path", default="results/spike_locs_y.npy", type=Path)
    p.add_argument("--alpha_path", default="results/spike_locs_alpha.npy", type=Path)
    p.add_argument("--fs_path", default="results/fs.npy", type=Path)
    p.add_argument("--ch_path", default="results/channel_locations.npy", type=Path)
    p.add_argument("--out", default="figures/depth_raster.png", type=Path)
    p.add_argument("--label", default="depth-raster", help="title prefix")
    p.add_argument("--vmin", default=None, type=float)
    p.add_argument("--vmax", default=None, type=float)
    p.add_argument("--ylim", default=None, help="y_lo,y_hi (μm); default = probe span + slack")
    args = p.parse_args()

    spike_times = np.load(args.t_path)
    y_um = np.load(args.y_path).astype(np.float32)
    alpha = np.load(args.alpha_path).astype(np.float32)
    fs = float(np.load(args.fs_path)[0])
    ch = np.load(args.ch_path)

    assert len(y_um) == len(spike_times) == len(alpha), \
        f"length mismatch: y={len(y_um)} t={len(spike_times)} α={len(alpha)}"

    t_s = spike_times.astype(np.float64) / fs
    log_alpha = np.log10(np.clip(alpha, 1.0, None))

    auto_lo, auto_hi = percentile_clip(log_alpha)
    vmin = args.vmin if args.vmin is not None else auto_lo
    vmax = args.vmax if args.vmax is not None else auto_hi

    if args.ylim is not None:
        y_lo, y_hi = (float(s) for s in args.ylim.split(","))
    else:
        y_lo = float(ch[:, 1].min()) - 30.0
        y_hi = float(ch[:, 1].max()) + 60.0

    xlim = (0.0, float(t_s.max()))
    title = (f"{args.label}  ·  N={len(spike_times):,}  ·  "
             f"t∈[0,{t_s.max():.0f}]s,  y∈[{ch[:,1].min():.0f},{ch[:,1].max():.0f}]μm  "
             f"·  log₁₀α∈[{vmin:.2f},{vmax:.2f}]")

    save_single_panel(
        args.out, t_s, y_um, log_alpha,
        vmin=vmin, vmax=vmax, title=title,
        style=DepthRasterStyle(),
        xlim=xlim, ylim=(y_lo, y_hi),
    )
    print(f"wrote {args.out}  ({args.out.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
