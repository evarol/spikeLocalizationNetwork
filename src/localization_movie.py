"""Bookmarked visualization: localization-movie.

Per-frame layout (one method):
  Top row: x-y, x-z, z-y panels with channel markers (z trimmed to z>0
  for x-z and z-y).
  Bottom row: motion estimate, mean X-Y pairwise correlation, spatial
  entropy — each is a full-session timeseries with a marker at the
  current frame.

Aspect ratios and panel proportions mirror localization_comparison_v2.mp4.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


XY_XLIM = (-10.0, 3850.0)   # depth on x-axis
XY_YLIM = (-42.0, 82.0)     # lateral x on y-axis
XZ_XLIM = (0.0, 65.0)       # z trimmed to z>0
XZ_YLIM = (-42.0, 82.0)
ZY_XLIM = (-10.0, 3850.0)
ZY_YLIM = (0.0, 65.0)       # z trimmed to z>0

Y_SPAN = XY_XLIM[1] - XY_XLIM[0]
X_SPAN = XY_YLIM[1] - XY_YLIM[0]
Z_SPAN = XZ_XLIM[1] - XZ_XLIM[0]

LOG_VMIN = 2.95
LOG_VMAX = 3.88

PANEL_BG = "#0d0d0d"
SPINE_COLOR = "#5b5b5b"
TEXT_COLOR = "white"


@dataclass
class MovieFrameData:
    frame_ids: np.ndarray          # int per spike, in [0, T)
    GL: np.ndarray                  # (N, 3) [x, y, z]
    log_alpha: np.ndarray           # (N,)
    ch_xyz: np.ndarray              # (C, 3) channels with z=0
    motion_trace: np.ndarray        # (T,) μm — DREDge motion estimate
    mean_corr_trace: np.ndarray     # (T,) Pearson row mean
    entropy_trace: np.ndarray       # (T,) nats
    method_label: str = "Raw Monopolar"
    method_color: str = "#ff8a55"
    motion_label: str = "DREDge estimate (rigid) μm"
    corr_label: str = "mean X-Y Pearson ρ"
    entropy_label: str = "spatial entropy (nats)"
    frame_offsets: np.ndarray = None  # cumulative spike-index breaks per frame


def _build_frame_offsets(frame_ids: np.ndarray, T: int) -> np.ndarray:
    """Return (T+1,) breakpoints in a frame-sorted spike index array."""
    counts = np.bincount(frame_ids, minlength=T)
    offsets = np.zeros(T + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    return offsets


def prepare_frame_index(data: MovieFrameData) -> MovieFrameData:
    """Sort spikes by frame_id so we can slice per-frame in O(1)."""
    T = len(data.motion_trace)
    order = np.argsort(data.frame_ids, kind="stable")
    new = MovieFrameData(
        frame_ids=data.frame_ids[order],
        GL=data.GL[order],
        log_alpha=data.log_alpha[order],
        ch_xyz=data.ch_xyz,
        motion_trace=data.motion_trace,
        mean_corr_trace=data.mean_corr_trace,
        entropy_trace=data.entropy_trace,
        method_label=data.method_label,
        method_color=data.method_color,
        motion_label=data.motion_label,
        corr_label=data.corr_label,
        entropy_label=data.entropy_label,
    )
    new.frame_offsets = _build_frame_offsets(new.frame_ids, T)
    return new


def _scatter_panel(ax, cloud_h, cloud_v, colors, ch_h, ch_v, xlim, ylim,
                   xlabel, ylabel, label_text):
    ax.set_facecolor(PANEL_BG)
    ax.scatter(ch_h, ch_v, s=6.0, c="white", alpha=0.16, linewidths=0,
               marker="s", rasterized=True, zorder=1.5)
    if len(cloud_h) > 0:
        ax.scatter(cloud_h, cloud_v, s=6.0, c=colors, alpha=0.58, linewidths=0,
                   rasterized=True, zorder=2)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel, color=TEXT_COLOR, fontsize=8)
    ax.set_ylabel(ylabel, color=TEXT_COLOR, fontsize=8)
    ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax.grid(alpha=0.12, linewidth=0.4)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_COLOR)
    ax.text(0.02, 0.96, label_text, transform=ax.transAxes, ha="left", va="top",
            color=TEXT_COLOR, fontsize=8, fontweight="bold")


def _timeseries_panel(ax, trace, frame_idx, color, ylabel, title):
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_COLOR)
    t_range = np.arange(len(trace))
    ax.plot(t_range, trace, color=color, lw=0.9, alpha=0.9, rasterized=True)
    cur = float(trace[frame_idx]) if 0 <= frame_idx < len(trace) and np.isfinite(trace[frame_idx]) else np.nan
    ax.axvline(frame_idx, color="white", lw=1.0, alpha=0.75, zorder=3)
    if np.isfinite(cur):
        ax.scatter([frame_idx], [cur], s=30, color=color, edgecolors="white",
                   linewidths=0.5, zorder=4)
    finite = np.isfinite(trace)
    if finite.any():
        ylo = float(trace[finite].min()); yhi = float(trace[finite].max())
        pad = 0.08 * max(yhi - ylo, 1e-6)
        ax.set_ylim(ylo - pad, yhi + pad)
    ax.set_xlim(0, len(trace) - 1)
    ax.set_ylabel(ylabel, color=TEXT_COLOR, fontsize=9)
    ax.set_title(title, color=TEXT_COLOR, fontsize=9)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.grid(alpha=0.14, linewidth=0.4)


def render_frame(
    frame_idx: int,
    data: MovieFrameData,
    *,
    fig_w: float = 24.0,
    z_inches: float = 2.0,
    spacer_inches: float = 0.5,
    bottom_inches: float = 3.6,
    top_margin: float = 0.6,
    left_margin_in: float = 1.2,
    right_margin_in: float = 0.2,
    gap_in: float = 0.10,
    middle_margin_in: float = 0.4,
    bottom_margin_in: float = 0.55,
    ts_gap_in: float = 0.20,
) -> plt.Figure:
    """Render one localization-movie frame.

    Layout uses fig.add_axes() with explicit figure-fraction positions:
      - x-y (top, col 0)   ← aspect=auto, cell sized to keep true data scale
      - x-z (top, col 1)   ← aspect=auto, z stretched (col 1 width = z_inches)
      - z-y (bottom, col 0) ← aspect=auto, z stretched (row height = z_inches)
      - 3 timeseries panels span the full top-block width

    Alignment guarantees:
      x-y and z-y share identical (left, width) ⇒ depth-axis pixel-aligned.
      x-y and x-z share identical (bottom, height) ⇒ lateral-x pixel-aligned.
      z stretches identically in x-z (horizontal) and z-y (vertical).
    """
    assert data.frame_offsets is not None, "call prepare_frame_index() first"

    Y, X = Y_SPAN, X_SPAN

    usable_w = fig_w - left_margin_in - right_margin_in
    w_xy = usable_w - gap_in - z_inches
    h_xy = w_xy * X / Y
    h_zy = z_inches

    top_inches = h_xy + spacer_inches + h_zy
    fig_h = top_inches + bottom_inches + top_margin

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="black")

    # Top-row figure-fraction positions.
    left_c0 = left_margin_in / fig_w
    width_c0 = w_xy / fig_w
    left_c1 = (left_margin_in + w_xy + gap_in) / fig_w
    width_c1 = z_inches / fig_w
    full_width = width_c0 + (gap_in + z_inches) / fig_w

    top_y = 1.0 - (top_margin / fig_h)
    bot_xy_y = top_y - (h_xy / fig_h)
    bot_zy_y = bot_xy_y - (spacer_inches + h_zy) / fig_h
    h_xy_frac = h_xy / fig_h
    h_zy_frac = h_zy / fig_h

    ax_xy = fig.add_axes([left_c0, bot_xy_y, width_c0, h_xy_frac])
    ax_xz = fig.add_axes([left_c1, bot_xy_y, width_c1, h_xy_frac])
    ax_zy = fig.add_axes([left_c0, bot_zy_y, width_c0, h_zy_frac])

    cmap = matplotlib.colormaps["inferno"]
    norm = mcolors.Normalize(vmin=LOG_VMIN, vmax=LOG_VMAX, clip=True)

    a, b = int(data.frame_offsets[frame_idx]), int(data.frame_offsets[frame_idx + 1])
    GL_f = data.GL[a:b]
    log_alpha_f = data.log_alpha[a:b]
    colors_f = cmap(norm(log_alpha_f)) if len(log_alpha_f) else np.zeros((0, 4))

    ax_xy.set_title(data.method_label, color=data.method_color, fontsize=11)

    _scatter_panel(ax_xy,
                   GL_f[:, 1] if len(GL_f) else np.empty(0),
                   GL_f[:, 0] if len(GL_f) else np.empty(0),
                   colors_f, data.ch_xyz[:, 1], data.ch_xyz[:, 0],
                   XY_XLIM, XY_YLIM, "y / depth (μm)", "x (μm)", "x-y")
    ax_xy.set_aspect("auto")

    _scatter_panel(ax_xz,
                   GL_f[:, 2] if len(GL_f) else np.empty(0),
                   GL_f[:, 0] if len(GL_f) else np.empty(0),
                   colors_f, data.ch_xyz[:, 2], data.ch_xyz[:, 0],
                   XZ_XLIM, XZ_YLIM, "z (μm)", "x (μm)", "x-z")
    ax_xz.set_aspect("auto")

    _scatter_panel(ax_zy,
                   GL_f[:, 1] if len(GL_f) else np.empty(0),
                   GL_f[:, 2] if len(GL_f) else np.empty(0),
                   colors_f, data.ch_xyz[:, 1], data.ch_xyz[:, 2],
                   ZY_XLIM, ZY_YLIM, "y / depth (μm)", "z (μm)", "z-y")
    ax_zy.set_aspect("auto")

    # Bottom 3 timeseries panels (full width).
    ts_block_top = bot_zy_y - (middle_margin_in / fig_h)
    ts_block_bottom = bottom_margin_in / fig_h
    ts_total_h = ts_block_top - ts_block_bottom
    ts_h_each = (ts_total_h - 2 * ts_gap_in / fig_h) / 3.0

    ax_mot = fig.add_axes([left_c0, ts_block_top - 1 * ts_h_each - 0 * ts_gap_in/fig_h, full_width, ts_h_each])
    ax_corr = fig.add_axes([left_c0, ts_block_top - 2 * ts_h_each - 1 * ts_gap_in/fig_h, full_width, ts_h_each])
    ax_ent = fig.add_axes([left_c0, ts_block_top - 3 * ts_h_each - 2 * ts_gap_in/fig_h, full_width, ts_h_each])

    _timeseries_panel(ax_mot, data.motion_trace, frame_idx, data.method_color,
                      data.motion_label, "Motion (Δy applied)")
    _timeseries_panel(ax_corr, data.mean_corr_trace, frame_idx, data.method_color,
                      data.corr_label, "Mean X-Y Pearson ρ vs all other bins")
    _timeseries_panel(ax_ent, data.entropy_trace, frame_idx, data.method_color,
                      data.entropy_label, "Spatial entropy H(t)")
    ax_ent.set_xlabel("time (s)", color=TEXT_COLOR, fontsize=9)

    fig.suptitle(
        f"localization-movie  ·  {data.method_label}  ·  t = {frame_idx}–{frame_idx + 1} s",
        color=TEXT_COLOR, fontsize=12, y=1.0 - 0.18/fig_h,
    )
    return fig
