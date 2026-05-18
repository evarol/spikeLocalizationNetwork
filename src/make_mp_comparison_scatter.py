"""2-row × 3-col scatter of learned-method localizations vs MP+DREDge.

  Top row:    GLOBAL (x, y, z)     — model values vs MP+DREDge values
  Bottom row: LOCAL  (l_x, l_y, z) — model − anchor  vs  MP+DREDge − anchor

For each panel:
    x_axis = MP+DREDge value (global or local)
    y_axis = method's value (global or local)
    diagonal (red dashed) = identity

Why both? The "anchor" (channel-neighborhood centroid) is a per-spike constant
shared by every method's localization. Global scatter Pearson ρ is inflated by
this shared component (anchor variance ≫ localization-offset variance). The
local scatter strips that out and shows the agreement on the actual
offset-from-anchor decision — the part the model actually predicts.

Reports per-axis RMS deviation and Pearson ρ in each panel title.
Both inputs are drift-corrected (so disagreement is localization, not motion).
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
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.set_xlabel(xlabel, color=TEXT, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT, fontsize=10)
    if title:
        ax.set_title(title, color=TEXT, fontsize=10)
    ax.grid(alpha=0.12, linewidth=0.4)


def panel(ax, mp_vals: np.ndarray, m_vals: np.ndarray, axn: str, method: str,
           frame: str, subsample: int = 200_000, seed: int = 0):
    if len(mp_vals) > subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(mp_vals), size=subsample, replace=False)
        idx.sort()
        mp_s = mp_vals[idx]; m_s = m_vals[idx]
    else:
        mp_s = mp_vals; m_s = m_vals

    ax.scatter(mp_s, m_s, s=0.6, alpha=0.10, c="#3b8bff",
                linewidths=0, rasterized=True)
    lo = float(min(mp_s.min(), m_s.min()))
    hi = float(max(mp_s.max(), m_s.max()))
    ax.plot([lo, hi], [lo, hi], color="#cc0000", lw=0.8, ls="--", alpha=0.7, zorder=3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    # Stats on FULL data
    d = m_vals - mp_vals
    rms = float(np.sqrt(np.mean(d * d)))
    bias = float(np.mean(d))
    corr = float(np.corrcoef(mp_vals, m_vals)[0, 1])
    axis_lbl = axn if frame == "global" else f"l_{axn}" if axn != "z" else "z"
    title = f"{frame}  {axis_lbl}    RMS={rms:.2f}μm  bias={bias:+.2f}  ρ={corr:.4f}"
    styled_axes(ax, f"MP+DREDge {axis_lbl} (μm)", f"{method} {axis_lbl} (μm)", title=title)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method_x", required=True, type=Path)
    p.add_argument("--method_y", required=True, type=Path)
    p.add_argument("--method_z", required=True, type=Path)
    p.add_argument("--mp_x", default="results/mp_dredge_all_spikes/x.npy", type=Path)
    p.add_argument("--mp_y", default="results/mp_dredge_all_spikes/y.npy", type=Path)
    p.add_argument("--mp_z", default="results/mp_dredge_all_spikes/z.npy", type=Path)
    p.add_argument("--anchors", default="results/spike_anchors.npy", type=Path,
                   help="(N, 3) per-spike channel-neighborhood centroid; "
                        "anchors[:, 2] is conventionally 0 so z_local = z_global")
    p.add_argument("--method_label", required=True, help="e.g. 'CNN all-spike ep20'")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n_plot", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # Load globals (drift-corrected) for both methods
    mp_x = np.load(args.mp_x).astype(np.float32)
    mp_y = np.load(args.mp_y).astype(np.float32)
    mp_z = np.load(args.mp_z).astype(np.float32)
    m_x = np.load(args.method_x).astype(np.float32)
    m_y = np.load(args.method_y).astype(np.float32)
    m_z = np.load(args.method_z).astype(np.float32)

    # Anchors (per-spike): the channel-neighborhood centroid common to both methods
    anchors = np.load(args.anchors).astype(np.float32)   # (N, 3)
    assert len(mp_x) == len(m_x) == len(anchors), "length mismatch"
    N = len(mp_x)

    # Local frame = global − anchor (per-spike; anchor z is 0 by convention so z is unchanged)
    a_x, a_y, a_z = anchors[:, 0], anchors[:, 1], anchors[:, 2]
    mp_lx = mp_x - a_x;  m_lx = m_x - a_x
    mp_ly = mp_y - a_y;  m_ly = m_y - a_y
    mp_lz = mp_z - a_z;  m_lz = m_z - a_z

    fig, axes = plt.subplots(2, 3, figsize=(18, 12.4), facecolor="black",
                              constrained_layout=True)
    # Top row: global
    panel(axes[0, 0], mp_x, m_x, "x", args.method_label, "global", subsample=args.n_plot, seed=args.seed)
    panel(axes[0, 1], mp_y, m_y, "y", args.method_label, "global", subsample=args.n_plot, seed=args.seed)
    panel(axes[0, 2], mp_z, m_z, "z", args.method_label, "global", subsample=args.n_plot, seed=args.seed)
    # Bottom row: local
    panel(axes[1, 0], mp_lx, m_lx, "x", args.method_label, "local", subsample=args.n_plot, seed=args.seed)
    panel(axes[1, 1], mp_ly, m_ly, "y", args.method_label, "local", subsample=args.n_plot, seed=args.seed)
    panel(axes[1, 2], mp_lz, m_lz, "z", args.method_label, "local", subsample=args.n_plot, seed=args.seed)

    fig.suptitle(
        f"mp-comparison-scatter   ·   {args.method_label}  vs  MP+DREDge canonical baseline\n"
        f"top: global (x, y, z)   ·   bottom: local (l_x, l_y, z) = global − anchor   ·   "
        f"N={N:,} (plotted: min({args.n_plot:,}, N))   ·   dataset1_p1",
        color=TEXT, fontsize=12, y=1.015,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}  ({args.out.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
