"""Bookmarked visualization: aggregate-projections.

Takes drift-corrected spike localizations (one (x, y, z) per spike) and
renders the 3 projection panels (x-y, x-z, z-y) matching the layout and
physical aspect ratios used by the localization-movie's top block. No time
axis — all spikes from the recording are scattered together.

Layout (same explicit-position scheme as localization_movie.render_frame):
  - x-y (col 0, top): depth y on x-axis, lateral x on y-axis, aspect=equal
  - x-z (col 1, top): z on x-axis, lateral x on y-axis, aspect=auto, z stretched
  - z-y (col 0, bottom): depth y on x-axis, z on y-axis, aspect=auto, z stretched
  - The (col 1, bottom) corner is blank by design (alignment guarantees).

Pinned alignment: x-y and z-y share (left, width); x-y and x-z share (bottom,
height).
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from localization_movie import (
    LOG_VMAX, LOG_VMIN, PANEL_BG, SPINE_COLOR, TEXT_COLOR,
    XY_XLIM, XY_YLIM, XZ_XLIM, XZ_YLIM, ZY_XLIM, ZY_YLIM,
    Y_SPAN, X_SPAN, Z_SPAN,
)


def _agg_panel(ax, cloud_h, cloud_v, colors, ch_h, ch_v, xlim, ylim,
               xlabel, ylabel, label_text, style):
    """Aggregate scatter with configurable marker size / alpha (small/dim by
    default for 2.5M-point clouds)."""
    ax.set_facecolor(PANEL_BG)
    ax.scatter(ch_h, ch_v, s=style.s_channel, c="white",
               alpha=style.alpha_channel, linewidths=0,
               marker="s", rasterized=True, zorder=1.5)
    if len(cloud_h) > 0:
        ax.scatter(cloud_h, cloud_v, s=style.s_spike, c=colors,
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


@dataclass(frozen=True)
class AggregateProjectionsStyle:
    fig_w: float = 24.0
    z_inches: float = 2.0
    spacer_inches: float = 0.5
    top_margin: float = 0.6
    left_margin_in: float = 1.2
    right_margin_in: float = 0.2
    gap_in: float = 0.10
    bottom_margin_in: float = 0.55
    dpi: int = 200
    # Scatter style — tuned for aggregate visibility (small marker, low alpha).
    s_spike: float = 0.25
    alpha_spike: float = 0.10
    s_channel: float = 4.0
    alpha_channel: float = 0.18


def render_aggregate_projections(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    log_alpha: np.ndarray, ch_xyz: np.ndarray,
    *,
    title: str = "",
    style: AggregateProjectionsStyle = AggregateProjectionsStyle(),
    y_zoom: tuple[float, float] | None = None,
) -> plt.Figure:
    """Render a single 3-panel aggregate-projections figure. Returns the Figure
    so the caller can save / close. ``x``, ``y``, ``z``, ``log_alpha`` are
    per-spike (N,) arrays; ``ch_xyz`` is (C, 3) channel coords with z=0."""
    Y, X = Y_SPAN, X_SPAN

    usable_w = style.fig_w - style.left_margin_in - style.right_margin_in
    w_xy = usable_w - style.gap_in - style.z_inches
    h_xy = w_xy * X / Y
    h_zy = style.z_inches
    top_inches = h_xy + style.spacer_inches + h_zy
    fig_h = top_inches + style.top_margin + style.bottom_margin_in

    fig = plt.figure(figsize=(style.fig_w, fig_h), facecolor="black")

    left_c0 = style.left_margin_in / style.fig_w
    width_c0 = w_xy / style.fig_w
    left_c1 = (style.left_margin_in + w_xy + style.gap_in) / style.fig_w
    width_c1 = style.z_inches / style.fig_w
    top_y = 1.0 - (style.top_margin / fig_h)
    bot_xy_y = top_y - (h_xy / fig_h)
    bot_zy_y = bot_xy_y - (style.spacer_inches + h_zy) / fig_h
    h_xy_frac = h_xy / fig_h
    h_zy_frac = h_zy / fig_h

    ax_xy = fig.add_axes([left_c0, bot_xy_y, width_c0, h_xy_frac])
    ax_xz = fig.add_axes([left_c1, bot_xy_y, width_c1, h_xy_frac])
    ax_zy = fig.add_axes([left_c0, bot_zy_y, width_c0, h_zy_frac])

    cmap = matplotlib.colormaps["inferno"]
    norm = mcolors.Normalize(vmin=LOG_VMIN, vmax=LOG_VMAX, clip=True)
    # Plot darker-amplitude spikes on top so dense bands stay visible.
    order = np.argsort(log_alpha)
    x_s = x[order]; y_s = y[order]; z_s = z[order]; la = log_alpha[order]
    colors = cmap(norm(la))

    xy_xlim = y_zoom if y_zoom is not None else XY_XLIM
    zy_xlim = y_zoom if y_zoom is not None else ZY_XLIM

    _agg_panel(ax_xy, y_s, x_s, colors, ch_xyz[:, 1], ch_xyz[:, 0],
               xy_xlim, XY_YLIM, "y / depth (μm)", "x (μm)", "x-y", style)
    ax_xy.set_aspect("auto")

    _agg_panel(ax_xz, z_s, x_s, colors, ch_xyz[:, 2], ch_xyz[:, 0],
               XZ_XLIM, XZ_YLIM, "z (μm)", "x (μm)", "x-z", style)
    ax_xz.set_aspect("auto")

    _agg_panel(ax_zy, y_s, z_s, colors, ch_xyz[:, 1], ch_xyz[:, 2],
               zy_xlim, ZY_YLIM, "y / depth (μm)", "z (μm)", "z-y", style)
    ax_zy.set_aspect("auto")

    if title:
        fig.suptitle(title, color=TEXT_COLOR, fontsize=12,
                     y=1.0 - 0.18 / fig_h)
    return fig
