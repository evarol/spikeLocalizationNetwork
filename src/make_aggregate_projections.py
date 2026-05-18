"""Generate an aggregate-projections figure (method-agnostic).

Inputs are per-spike arrays. Defaults render the raw monopolar localizations
on dataset1_p1; override --{x,y,z,alpha,ch}_path for any other method.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aggregate_projections import (
    AggregateProjectionsStyle, render_aggregate_projections,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--x_path", default="results/tpca_monopolar_full/spike_locs_x.npy", type=Path)
    p.add_argument("--y_path", default="results/tpca_monopolar_full/spike_locs_y.npy", type=Path)
    p.add_argument("--z_path", default="results/tpca_monopolar_full/spike_locs_z.npy", type=Path)
    p.add_argument("--alpha_path", default="results/spike_locs_alpha.npy", type=Path)
    p.add_argument("--ch_path", default="results/channel_locations.npy", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--label", default="aggregate-projections", type=str)
    p.add_argument("--z_inches", default=2.0, type=float)
    p.add_argument("--y_zoom", default=None,
                   help="Restrict depth (y) axis to y_lo,y_hi (μm) — zooms the "
                        "x-y and z-y panels to make small drift differences visible.")
    args = p.parse_args()

    x = np.load(args.x_path).astype(np.float32)
    y = np.load(args.y_path).astype(np.float32)
    z = np.load(args.z_path).astype(np.float32)
    alpha = np.load(args.alpha_path).astype(np.float32)
    ch = np.load(args.ch_path).astype(np.float32)
    assert len(x) == len(y) == len(z) == len(alpha), "length mismatch"

    ch_xyz = np.zeros((len(ch), 3), dtype=np.float32)
    ch_xyz[:, 0] = ch[:, 0]; ch_xyz[:, 1] = ch[:, 1]
    log_alpha = np.log10(np.clip(alpha, 1.0, None))

    style = AggregateProjectionsStyle(z_inches=args.z_inches)
    title = f"{args.label}  ·  N = {len(x):,}  ·  inferno cmap on log₁₀ α"
    y_zoom = None
    if args.y_zoom is not None:
        y_zoom = tuple(float(v) for v in args.y_zoom.split(","))
        assert len(y_zoom) == 2, "--y_zoom must be y_lo,y_hi"
        title = f"{title}  ·  y ∈ [{y_zoom[0]:.0f}, {y_zoom[1]:.0f}] μm"
    fig = render_aggregate_projections(x, y, z, log_alpha, ch_xyz,
                                       title=title, style=style, y_zoom=y_zoom)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=style.dpi, facecolor="black")
    plt.close(fig)
    print(f"wrote {args.out}  ({args.out.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
