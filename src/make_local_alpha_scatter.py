"""3-panel local-coordinate scatter, colored by log10 monopolar α.

Panels:
  - top-left:  x-y (local)   x-axis = y_local (μm),  y-axis = x_local (μm)
  - top-right: x-z (local)   x-axis = z (μm),         y-axis = x_local (μm)
  - bottom-left: z-y (local) x-axis = y_local (μm),  y-axis = z (μm)

l_x, l_y are (predicted position − channel-anchor centroid) — the network's
output offset before motion correction. z anchor is 0 so z = predicted z.

This reproduces the figT5_local_<method>_color_alpha.png panel-set previously
generated inline; promoted to a script so all 4 methods can be rendered with
a single command per method.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PANEL_BG = "#0d0d0d"
SPINE = "#5b5b5b"
TEXT = "white"


def styled_axes(ax, xlabel: str, ylabel: str, title: str = ""):
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE)
    ax.tick_params(colors=TEXT, labelsize=7)
    ax.set_xlabel(xlabel, color=TEXT, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT, fontsize=9)
    if title:
        ax.text(0.02, 0.96, title, transform=ax.transAxes,
                color=TEXT, fontsize=9, fontweight="bold",
                ha="left", va="top")
    ax.grid(alpha=0.12, linewidth=0.4)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--GL_pre", required=True, type=Path,
                   help="(N, 3) anchor + SLN-rel, no motion (results/.../GL_pre.npy "
                        "or GL_pre_dredge.npy depending on apply script)")
    p.add_argument("--anchors", default="results/spike_anchors.npy", type=Path)
    p.add_argument("--alpha_path", default="results/spike_locs_alpha.npy", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--label", required=True,
                   help="e.g. 'CNN-SLN_all-spike_post-DREDge_ep10'")
    p.add_argument("--xy_lx_range", default="-100,100",
                   help="x_local range (μm) for top-left + top-right y axes")
    p.add_argument("--xy_ly_range", default="-100,100",
                   help="y_local range (μm) for top-left + bottom x axes")
    p.add_argument("--z_range", default="0,125",
                   help="z range (μm) for top-right x and bottom y axes")
    args = p.parse_args()

    GL_pre = np.load(args.GL_pre).astype(np.float32)
    anchors = np.load(args.anchors).astype(np.float32)
    alpha = np.load(args.alpha_path).astype(np.float32)
    assert len(GL_pre) == len(anchors) == len(alpha), \
        f"length mismatch: GL_pre={len(GL_pre)} anchors={len(anchors)} alpha={len(alpha)}"

    l_x = GL_pre[:, 0] - anchors[:, 0]
    l_y = GL_pre[:, 1] - anchors[:, 1]
    z   = GL_pre[:, 2]    # z anchor = 0
    log_alpha = np.log10(np.clip(alpha, 1.0, None))
    N = len(l_x)

    # Sort by alpha so bright spikes draw on top
    order = np.argsort(log_alpha)
    l_x_o = l_x[order]; l_y_o = l_y[order]; z_o = z[order]; la_o = log_alpha[order]

    xy_lx = tuple(float(v) for v in args.xy_lx_range.split(","))
    xy_ly = tuple(float(v) for v in args.xy_ly_range.split(","))
    zr    = tuple(float(v) for v in args.z_range.split(","))

    # Color norm — match the existing figT5 aesthetic (inferno, ~3–4 log10α)
    vmin, vmax = float(np.percentile(la_o, 1)), float(np.percentile(la_o, 99))
    scatter_kw = dict(s=0.5, alpha=0.25, c=la_o, cmap="inferno",
                       vmin=vmin, vmax=vmax, linewidths=0, rasterized=True)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 11.5), facecolor="black",
                              constrained_layout=True)

    ax = axes[0, 0]
    sc = ax.scatter(l_y_o, l_x_o, **scatter_kw)
    ax.set_xlim(*xy_ly); ax.set_ylim(*xy_lx)
    styled_axes(ax, "y_local (μm)", "x_local (μm)", title="x-y (local)")

    ax = axes[0, 1]
    ax.scatter(z_o, l_x_o, **scatter_kw)
    ax.set_xlim(*zr); ax.set_ylim(*xy_lx)
    styled_axes(ax, "z (μm)", "x_local (μm)", title="x-z (local)")
    # Colorbar inset on the right side of this panel
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("log₁₀ α", color=TEXT, fontsize=8)
    cb.ax.tick_params(colors=TEXT, labelsize=7)
    cb.outline.set_edgecolor(SPINE)

    ax = axes[1, 0]
    ax.scatter(l_y_o, z_o, **scatter_kw)
    ax.set_xlim(*xy_ly); ax.set_ylim(*zr)
    styled_axes(ax, "y_local (μm)", "z (μm)", title="z-y (local)")

    # Hide the bottom-right empty panel
    axes[1, 1].set_visible(False)

    fig.suptitle(f"pos = (l_x,l_y)_{args.label}  ·  color = log₁₀ α  ·  N={N:,}",
                  color=TEXT, fontsize=11, y=1.01)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}  ({args.out.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
