"""Optical-flow-style displacement coloring.

For each spike, compute (Δx, Δy) = (target_x - source_x, target_y - source_y),
then color the spike at its SOURCE position using HSV:
    hue = direction of displacement vector
    saturation = magnitude / max_mag   (zero → white)
    value = 1

Layout uses the aggregate-projections / localization-movie top-block (x-y wide
panel, x-z on right, z-y below x-y). All three panels show the same per-spike
color since the displacement is in 2D x-y.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from aggregate_projections import AggregateProjectionsStyle
from localization_movie import (
    PANEL_BG, SPINE_COLOR, TEXT_COLOR,
    XY_XLIM, XY_YLIM, XZ_XLIM, XZ_YLIM, ZY_XLIM, ZY_YLIM,
    Y_SPAN, X_SPAN, Z_SPAN,
)


def displacement_to_rgb(dx_lateral: np.ndarray, dy_depth: np.ndarray,
                         max_mag: float) -> np.ndarray:
    """Return (N, 3) RGB per spike.

    Panel convention: angle 0° = +depth-y direction (right in panel).
    So hue = atan2(dx_lateral, dy_depth) wrapped to [0, 1].
    """
    mag = np.sqrt(dx_lateral ** 2 + dy_depth ** 2)
    angle = np.arctan2(dx_lateral, dy_depth)   # [-π, π]
    hue = (angle / (2 * np.pi)) % 1.0
    sat = np.clip(mag / max(max_mag, 1e-9), 0, 1)
    val = np.ones_like(sat)
    hsv = np.stack([hue, sat, val], axis=-1).astype(np.float32)
    rgb = mcolors.hsv_to_rgb(hsv)
    return rgb, mag


def draw_colorwheel(ax: plt.Axes, max_mag: float):
    """Render a colorwheel inset with axis labels matching the panel convention."""
    n = 256
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X * X + Y * Y)
    # 0° angle = +depth-y = +X in this inset
    A = np.arctan2(Y, X)
    H = (A / (2 * np.pi)) % 1.0
    S = np.clip(R, 0, 1)
    V = np.ones_like(R)
    hsv = np.stack([H, S, V], axis=-1)
    rgb = mcolors.hsv_to_rgb(hsv)
    rgb[R > 1] = 1.0  # outside disk → white
    ax.imshow(rgb, extent=[-1, 1, -1, 1], origin="lower")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_facecolor("black")
    # Axis arrows / labels (panel convention: +depth-y → right, +lateral-x → up)
    ax.annotate("", xy=(1.0, 0), xytext=(-1.0, 0),
                arrowprops=dict(arrowstyle="->", color="white", lw=0.8))
    ax.annotate("", xy=(0, 1.0), xytext=(0, -1.0),
                arrowprops=dict(arrowstyle="->", color="white", lw=0.8))
    ax.text(1.05, 0, "+depth y", color="white", fontsize=6,
            ha="left", va="center")
    ax.text(0, 1.07, "+lateral x", color="white", fontsize=6,
            ha="center", va="bottom")
    ax.text(0.5, -1.18, f"sat=1 ⇔ |Δ|≥{max_mag:.1f} μm",
            color="white", fontsize=6, ha="center", va="top",
            transform=ax.transAxes)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)


def _panel(ax, h, v, colors, ch_h, ch_v, xlim, ylim, xlabel, ylabel, label_text,
           style):
    ax.set_facecolor(PANEL_BG)
    ax.scatter(ch_h, ch_v, s=style.s_channel, c="white",
               alpha=style.alpha_channel, linewidths=0,
               marker="s", rasterized=True, zorder=1.5)
    if len(h) > 0:
        ax.scatter(h, v, s=style.s_spike, c=colors,
                   alpha=style.alpha_spike, linewidths=0,
                   rasterized=True, zorder=2)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel, color=TEXT_COLOR, fontsize=8)
    ax.set_ylabel(ylabel, color=TEXT_COLOR, fontsize=8)
    ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax.grid(alpha=0.12, linewidth=0.4)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_COLOR)
    ax.text(0.02, 0.96, label_text, transform=ax.transAxes,
            ha="left", va="top", color=TEXT_COLOR, fontsize=8,
            fontweight="bold")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source_x", required=True, type=Path)
    p.add_argument("--source_y", required=True, type=Path)
    p.add_argument("--source_z", required=True, type=Path)
    p.add_argument("--target_x", required=True, type=Path)
    p.add_argument("--target_y", required=True, type=Path)
    p.add_argument("--ch_path", default="results/channel_locations.npy", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--label", required=True)
    p.add_argument("--max_mag_pct", type=float, default=95.0,
                   help="Magnitude percentile used to saturate the colorwheel.")
    p.add_argument("--alpha_spike", type=float, default=0.30,
                   help="Override aggregate-projections alpha (denser shows direction better).")
    p.add_argument("--s_spike", type=float, default=0.35)
    p.add_argument("--y_zoom", default=None)
    args = p.parse_args()

    sx = np.load(args.source_x).astype(np.float32)
    sy = np.load(args.source_y).astype(np.float32)
    sz = np.load(args.source_z).astype(np.float32)
    tx = np.load(args.target_x).astype(np.float32)
    ty = np.load(args.target_y).astype(np.float32)
    assert len(sx) == len(sy) == len(sz) == len(tx) == len(ty), "length mismatch"
    ch = np.load(args.ch_path).astype(np.float32)
    ch_xyz = np.zeros((len(ch), 3), dtype=np.float32)
    ch_xyz[:, 0] = ch[:, 0]; ch_xyz[:, 1] = ch[:, 1]

    dx = tx - sx
    dy = ty - sy
    max_mag = float(np.percentile(np.sqrt(dx * dx + dy * dy), args.max_mag_pct))
    rgb, mag = displacement_to_rgb(dx, dy, max_mag=max_mag)
    print(f"N={len(sx):,}  |Δ| stats: mean={mag.mean():.2f}, "
          f"p50={np.median(mag):.2f}, p95={max_mag:.2f}, max={mag.max():.2f} μm")

    # Plot densest (largest magnitude) on top for visibility.
    order = np.argsort(mag)

    style = AggregateProjectionsStyle(s_spike=args.s_spike,
                                       alpha_spike=args.alpha_spike)

    fig_w = style.fig_w
    Y, X = Y_SPAN, X_SPAN
    usable_w = fig_w - style.left_margin_in - style.right_margin_in
    w_xy = usable_w - style.gap_in - style.z_inches
    h_xy = w_xy * X / Y
    h_zy = style.z_inches
    top_inches = h_xy + style.spacer_inches + h_zy
    fig_h = top_inches + style.top_margin + style.bottom_margin_in

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="black")
    left_c0 = style.left_margin_in / fig_w
    width_c0 = w_xy / fig_w
    left_c1 = (style.left_margin_in + w_xy + style.gap_in) / fig_w
    width_c1 = style.z_inches / fig_w
    top_y = 1.0 - (style.top_margin / fig_h)
    bot_xy_y = top_y - (h_xy / fig_h)
    bot_zy_y = bot_xy_y - (style.spacer_inches + h_zy) / fig_h
    h_xy_frac = h_xy / fig_h
    h_zy_frac = h_zy / fig_h

    ax_xy = fig.add_axes([left_c0, bot_xy_y, width_c0, h_xy_frac])
    ax_xz = fig.add_axes([left_c1, bot_xy_y, width_c1, h_xy_frac])
    ax_zy = fig.add_axes([left_c0, bot_zy_y, width_c0, h_zy_frac])

    if args.y_zoom:
        y_zoom = tuple(float(v) for v in args.y_zoom.split(","))
        xy_xlim = y_zoom; zy_xlim = y_zoom
    else:
        xy_xlim = XY_XLIM; zy_xlim = ZY_XLIM

    sy_o = sy[order]; sx_o = sx[order]; sz_o = sz[order]; rgb_o = rgb[order]

    _panel(ax_xy, sy_o, sx_o, rgb_o, ch_xyz[:, 1], ch_xyz[:, 0],
           xy_xlim, XY_YLIM, "y / depth (μm)", "x (μm)", "x-y", style)
    ax_xy.set_aspect("auto")
    _panel(ax_xz, sz_o, sx_o, rgb_o, ch_xyz[:, 2], ch_xyz[:, 0],
           XZ_XLIM, XZ_YLIM, "z (μm)", "x (μm)", "x-z", style)
    ax_xz.set_aspect("auto")
    _panel(ax_zy, sy_o, sz_o, rgb_o, ch_xyz[:, 1], ch_xyz[:, 2],
           zy_xlim, ZY_YLIM, "y / depth (μm)", "z (μm)", "z-y", style)
    ax_zy.set_aspect("auto")

    # Colorwheel inset in the upper-right corner of the figure.
    cw_size_frac = 0.085
    cw_ax = fig.add_axes([
        1.0 - cw_size_frac - 0.015,
        1.0 - cw_size_frac * (fig_w / fig_h) - 0.015,
        cw_size_frac, cw_size_frac * (fig_w / fig_h),
    ])
    draw_colorwheel(cw_ax, max_mag=max_mag)

    fig.suptitle(
        f"{args.label}  ·  N={len(sx):,}  ·  color = displacement (HSV, "
        f"sat saturates at p{args.max_mag_pct:.0f}={max_mag:.1f} μm)",
        color=TEXT_COLOR, fontsize=11, y=1.0 - 0.18 / fig_h,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=style.dpi, facecolor="black")
    plt.close(fig)
    print(f"wrote {args.out}  ({args.out.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
