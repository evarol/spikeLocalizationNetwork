"""Generate an X-Y pairwise Pearson correlation matrix (method-agnostic).

Pass --x_path and --y_path for whatever method's localizations you want
(raw monopolar by default).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from xy_pairwise_correlation import (
    XYCorrStyle, XYHistGrid,
    build_xy_histograms, pairwise_pearson, save_single_panel,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--t_path", default="results/spike_times.npy", type=Path)
    p.add_argument("--x_path", default="results/spike_locs_x.npy", type=Path)
    p.add_argument("--y_path", default="results/spike_locs_y.npy", type=Path)
    p.add_argument("--fs_path", default="results/fs.npy", type=Path)
    p.add_argument("--out", default="figures/xy_pairwise_corr.png", type=Path)
    p.add_argument("--matrix_out", default="figures/xy_pairwise_corr.npy", type=Path)
    p.add_argument("--label", default="X-Y pairwise Pearson correlation",
                   help="title prefix")
    p.add_argument("--bin_s", default=1.0, type=float, help="time bin size (s)")
    p.add_argument("--x_bin", default=8.0, type=float, help="spatial x bin (μm)")
    p.add_argument("--y_bin", default=8.0, type=float, help="spatial y bin (μm)")
    p.add_argument("--x_lo", default=-80.0, type=float)
    p.add_argument("--x_hi", default=120.0, type=float)
    p.add_argument("--y_lo", default=-40.0, type=float)
    p.add_argument("--y_hi", default=3860.0, type=float)
    p.add_argument("--soft_sigma", default=0.0, type=float,
                   help="Gaussian σ (μm) for soft-histogram scatter. 0 = hard bin.")
    args = p.parse_args()

    t0 = time.time()
    spike_times = np.load(args.t_path)
    x_um = np.load(args.x_path).astype(np.float32)
    y_um = np.load(args.y_path).astype(np.float32)
    fs = float(np.load(args.fs_path)[0])

    assert len(x_um) == len(y_um) == len(spike_times), "length mismatch"

    t_s = spike_times.astype(np.float64) / fs
    T = int(np.ceil(t_s.max() / args.bin_s))
    t_bin = np.clip((t_s / args.bin_s).astype(np.int64), 0, T - 1)

    grid = XYHistGrid(x_lo=args.x_lo, x_hi=args.x_hi, x_bin_um=args.x_bin,
                       y_lo=args.y_lo, y_hi=args.y_hi, y_bin_um=args.y_bin)
    if args.soft_sigma > 0:
        D = len(grid.x_centers) * len(grid.y_centers)
        print(f"  T={T} bins · soft grid xc={len(grid.x_centers)} · yc={len(grid.y_centers)}"
              f" · D={D} · σ={args.soft_sigma}μm")
    else:
        D = (len(grid.x_edges) - 1) * (len(grid.y_edges) - 1)
        print(f"  T={T} bins · hard grid x={len(grid.x_edges)-1} · y={len(grid.y_edges)-1}"
              f" · D={D}")

    H = build_xy_histograms(t_bin, x_um, y_um, T, grid,
                             soft_sigma_um=args.soft_sigma)
    print(f"  built H {H.shape} in {time.time()-t0:.1f}s")

    C = pairwise_pearson(H)
    off_diag = C[~np.eye(T, dtype=bool)]
    print(f"  pairwise Pearson:  C∈[{C.min():.3f},{C.max():.3f}],  "
          f"mean off-diag = {off_diag.mean():.3f}")

    args.matrix_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.matrix_out, C)
    print(f"  saved matrix → {args.matrix_out}")

    title = (f"{args.label}\nT={T}s, grid Δx={args.x_bin:.0f}μm Δy={args.y_bin:.0f}μm,  "
             f"mean off-diag ρ = {off_diag.mean():.3f}")
    save_single_panel(args.out, C, title=title, style=XYCorrStyle())
    print(f"  wrote {args.out}  ({args.out.stat().st_size/1e6:.2f} MB)   "
          f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
