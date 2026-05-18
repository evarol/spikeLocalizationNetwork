"""Depth raster colored by the SLN-predicted z coordinate (μm) instead of
log₁₀ α. Same y(t) layout, just a different per-spike color channel.

Useful for asking "do the z predictions look stable across time per depth
band?" — visible drift in color = unstable z estimates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from depth_raster import DepthRasterStyle, save_single_panel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--t_path", default="results/spike_times.npy", type=Path)
    p.add_argument("--y_path", required=True, type=Path,
                   help="per-spike y / depth (μm) — y-axis position")
    p.add_argument("--z_path", required=True, type=Path,
                   help="per-spike z (μm) — used as color")
    p.add_argument("--fs_path", default="results/fs.npy", type=Path)
    p.add_argument("--ch_path", default="results/channel_locations.npy", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--label", required=True)
    p.add_argument("--vmin", default=None, type=float,
                   help="z color floor (μm); default = 1st percentile")
    p.add_argument("--vmax", default=None, type=float,
                   help="z color ceil  (μm); default = 99th percentile")
    p.add_argument("--cmap", default="viridis",
                   help="colormap for z (default 'viridis'; 'inferno' for alpha-like)")
    p.add_argument("--ylim", default=None, help="y_lo,y_hi (μm)")
    args = p.parse_args()

    spike_times = np.load(args.t_path)
    y_um = np.load(args.y_path).astype(np.float32)
    z_um = np.load(args.z_path).astype(np.float32)
    fs = float(np.load(args.fs_path)[0])
    ch = np.load(args.ch_path)

    assert len(y_um) == len(spike_times) == len(z_um), \
        f"length mismatch: y={len(y_um)} t={len(spike_times)} z={len(z_um)}"

    t_s = spike_times.astype(np.float64) / fs

    vmin = args.vmin if args.vmin is not None else float(np.percentile(z_um, 1))
    vmax = args.vmax if args.vmax is not None else float(np.percentile(z_um, 99))

    if args.ylim is not None:
        y_lo, y_hi = (float(s) for s in args.ylim.split(","))
    else:
        y_lo = float(ch[:, 1].min()) - 30.0
        y_hi = float(ch[:, 1].max()) + 60.0

    xlim = (0.0, float(t_s.max()))
    title = (f"{args.label}  ·  N={len(spike_times):,}  ·  "
             f"t∈[0,{t_s.max():.0f}]s,  y∈[{ch[:,1].min():.0f},{ch[:,1].max():.0f}]μm  "
             f"·  color = z (μm) ∈ [{vmin:.1f}, {vmax:.1f}]")

    # Override style cmap + colorbar label for z
    style = DepthRasterStyle(
        cmap=args.cmap,
        cbar_label="z (μm)",
    )

    save_single_panel(
        args.out, t_s, y_um, z_um,
        vmin=vmin, vmax=vmax, title=title,
        style=style,
        xlim=xlim, ylim=(y_lo, y_hi),
    )
    print(f"wrote {args.out}  ({args.out.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
