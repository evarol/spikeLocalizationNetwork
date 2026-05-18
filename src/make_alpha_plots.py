"""Prototype: alpha-related diagnostic plots.

Three figure types:
  (1) alpha vs axis aggregate scatter: 3-panel (α↔x, α↔y, α↔z).
  (2) alpha variability vs depth: how much does mean log₁₀α at a given depth
      bin fluctuate over time? Lower = better motion correction (a single unit's
      true α is constant; flicker comes from drift moving it across channels).
  (3) per-spike α heatmap over (depth, time): visualize whether each unit's α
      stripe is stable in time.

For SLN-method comparisons, α is always the MP-fitted amplitude (the SLN does
not predict α). What changes between methods is *which depth y* each spike is
assigned to — so the joint (y, α, t) structure differs even though α itself
doesn't change.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
        ax.set_title(title, color=TEXT, fontsize=10)
    ax.grid(alpha=0.14, linewidth=0.4)


def fig_alpha_vs_axis(x, y, z, log_alpha, out_path, title: str,
                       x_range=(-150, 200), y_range=(0, 3840), z_range=(0, 250)):
    """3-panel: x vs α, y vs α, z vs α (axis on x-axis, log α on y-axis)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="black",
                              constrained_layout=True)
    # Sort by mag so brightest on top
    order = np.argsort(log_alpha)
    for ax, vals, lbl, axis_range in [
        (axes[0], x[order], "x (μm)", x_range),
        (axes[1], y[order], "y / depth (μm)", y_range),
        (axes[2], z[order], "z (μm)", z_range),
    ]:
        ax.scatter(vals, log_alpha[order], s=0.4, alpha=0.15,
                   c=log_alpha[order], cmap="inferno",
                   vmin=2.95, vmax=3.88,
                   linewidths=0, rasterized=True)
        ax.set_xlim(*axis_range)
        styled_axes(ax, lbl, "log₁₀ α", title="")
    fig.suptitle(title, color=TEXT, fontsize=11, y=1.04)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def fig_alpha_variability_vs_depth(y, log_alpha, t_bin, out_path, title: str,
                                    y_bin_um: float = 20.0, y_range=(0, 3840),
                                    min_spikes_per_cell: int = 5):
    """For each depth bin, plot the standard deviation of mean(log₁₀α) across
    time bins. Smaller = more biologically plausible (α stable per unit)."""
    yb_edges = np.arange(y_range[0], y_range[1] + y_bin_um, y_bin_um)
    yi = np.clip(np.searchsorted(yb_edges, y, side="right") - 1, 0, len(yb_edges) - 2)
    T = int(t_bin.max()) + 1
    Yn = len(yb_edges) - 1

    # Accumulate sums and counts per (y_bin, t_bin)
    n_idx = yi.astype(np.int64) * T + t_bin.astype(np.int64)
    sums = np.bincount(n_idx, weights=log_alpha.astype(np.float64),
                       minlength=Yn * T).reshape(Yn, T)
    counts = np.bincount(n_idx, minlength=Yn * T).reshape(Yn, T)
    valid = counts >= min_spikes_per_cell
    mean_la = np.where(valid, sums / np.maximum(counts, 1), np.nan)

    # std over time per depth bin
    std_la = np.nanstd(mean_la, axis=1)
    mean_la_overall = np.nanmean(mean_la, axis=1)
    nvalid_t = valid.sum(axis=1)
    keep = nvalid_t >= max(10, T // 20)   # need at least some time coverage

    # 2-panel figure: (top) std vs depth; (bottom) heatmap of mean log α (y × t)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), facecolor="black",
                              gridspec_kw={"height_ratios": [1, 3]},
                              constrained_layout=True)

    yb_centers = 0.5 * (yb_edges[:-1] + yb_edges[1:])
    ax = axes[0]
    ax.plot(yb_centers[keep], std_la[keep], color="#ffcc55", lw=1.0)
    styled_axes(ax, "y / depth (μm)", "std(mean log₁₀α) over t",
                title="α temporal variability per depth bin (lower = better motion correction)")
    ax.set_xlim(*y_range)

    ax = axes[1]
    extent = (0.0, float(T), float(y_range[0]), float(y_range[1]))
    finite = mean_la[np.isfinite(mean_la)]
    if finite.size:
        vmin, vmax = np.percentile(finite, [1, 99])
    else:
        vmin, vmax = 2.95, 3.88
    im = ax.imshow(mean_la, origin="lower", aspect="auto", cmap="inferno",
                    extent=extent, vmin=vmin, vmax=vmax,
                    interpolation="nearest")
    styled_axes(ax, "time (s)", "y / depth (μm)",
                title="mean log₁₀α per (depth bin, 1-s time bin)")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("mean log₁₀α", color=TEXT, fontsize=8)
    cb.ax.tick_params(colors=TEXT, labelsize=7)
    cb.outline.set_edgecolor(SPINE)

    fig.suptitle(title, color=TEXT, fontsize=11, y=1.005)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--x_path", required=True, type=Path)
    p.add_argument("--y_path", required=True, type=Path)
    p.add_argument("--z_path", required=True, type=Path)
    p.add_argument("--alpha_path", default="results/spike_locs_alpha.npy", type=Path)
    p.add_argument("--t_path", default="results/spike_times.npy", type=Path)
    p.add_argument("--fs_path", default="results/fs.npy", type=Path)
    p.add_argument("--out_prefix", required=True,
                   help="figures/<name> — will produce _alpha_vs_axis.png and _alpha_var_vs_depth.png")
    p.add_argument("--label", required=True)
    args = p.parse_args()

    x = np.load(args.x_path).astype(np.float32)
    y = np.load(args.y_path).astype(np.float32)
    z = np.load(args.z_path).astype(np.float32)
    alpha = np.load(args.alpha_path).astype(np.float32)
    spike_times = np.load(args.t_path)
    fs = float(np.load(args.fs_path)[0])
    N = len(x)
    assert len(y) == len(z) == len(alpha) == len(spike_times) == N
    print(f"N = {N:,}")

    log_alpha = np.log10(np.clip(alpha, 1.0, None))
    t_s = spike_times.astype(np.float64) / fs
    t_bin = np.clip(t_s.astype(np.int64), 0, 1957)

    fig1_path = Path(f"{args.out_prefix}_alpha_vs_axis.png")
    fig_alpha_vs_axis(x, y, z, log_alpha, fig1_path,
                       title=f"α vs axis (aggregate) — {args.label}")
    print(f"wrote {fig1_path}  ({fig1_path.stat().st_size/1e6:.2f} MB)")

    fig2_path = Path(f"{args.out_prefix}_alpha_var_vs_depth.png")
    fig_alpha_variability_vs_depth(y, log_alpha, t_bin, fig2_path,
                                    title=f"α temporal variability vs depth — {args.label}")
    print(f"wrote {fig2_path}  ({fig2_path.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
