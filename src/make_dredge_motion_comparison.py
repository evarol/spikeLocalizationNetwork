"""Compare three DREDge motion estimates side-by-side:
   (A) MP + DREDge        — DREDge fit to monopolar (MP) localizations
   (B) CNN ep0 + DREDge   — DREDge fit to pretrain CNN localizations
   (C) CNN ep20 + DREDge  — DREDge fit to CNN-after-20-epoch localizations

Three rows, all on shared y-anchor + t-anchor:
   Row 1: disp(t, y) heatmaps for A, B, C (shared color scale)
   Row 2: pairwise difference heatmaps |A-B|, |A-C|, |B-C|
   Row 3: per-y-anchor time traces overlaid (A blue, B orange, C green)
          one panel per y-anchor (9 anchors → 9 small subplots)
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


def load_motion(p: Path) -> dict:
    d = dict(np.load(p))
    return {
        "disp": d["disp"].astype(np.float64),
        "t_anchors": d["t_anchors"].astype(np.float64),
        "y_anchors": d["y_anchors"].astype(np.float64),
    }


def styled_axes(ax, xlabel: str = "", ylabel: str = "", title: str = "", fs: int = 8):
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE)
    ax.tick_params(colors=TEXT, labelsize=fs - 1)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT, fontsize=fs)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT, fontsize=fs)
    if title:
        ax.set_title(title, color=TEXT, fontsize=fs + 1)
    ax.grid(alpha=0.12, linewidth=0.4)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mp",     default="results/mp_dredge_all_spikes/motion.npz",     type=Path)
    p.add_argument("--cnn0",   default="results/dredge_cnn_all/motion.npz",            type=Path)
    p.add_argument("--cnn20",  default="results/dredge2_cnn_all_ep20/motion.npz",      type=Path)
    p.add_argument("--out",    default="figures/dredge_motion_comparison.png",         type=Path)
    args = p.parse_args()

    A = load_motion(args.mp);     A_lbl = "MP + DREDge"
    B = load_motion(args.cnn0);   B_lbl = "CNN ep0 + DREDge"
    C = load_motion(args.cnn20);  C_lbl = "CNN ep20 + DREDge"

    # Sanity: all three share t_anchors and y_anchors
    assert np.allclose(A["t_anchors"], B["t_anchors"]) and np.allclose(A["t_anchors"], C["t_anchors"])
    assert np.allclose(A["y_anchors"], B["y_anchors"]) and np.allclose(A["y_anchors"], C["y_anchors"])
    t_anc, y_anc = A["t_anchors"], A["y_anchors"]
    T, K = A["disp"].shape

    # Shared color scale for the three heatmaps: ±max abs displacement across all three
    vmax = max(np.abs(d["disp"]).max() for d in (A, B, C))
    vmin = -vmax

    # Difference scale: half the vmax (differences are smaller)
    dmax = max(np.abs(A["disp"] - B["disp"]).max(),
               np.abs(A["disp"] - C["disp"]).max(),
               np.abs(B["disp"] - C["disp"]).max())
    print(f"vmax = {vmax:.1f}μm, dmax = {dmax:.1f}μm")

    # Stats
    print("\n--- per-pair stats ---")
    for (name, X, Y) in [(f"{A_lbl} vs {B_lbl}", A, B),
                          (f"{A_lbl} vs {C_lbl}", A, C),
                          (f"{B_lbl} vs {C_lbl}", B, C)]:
        d = X["disp"] - Y["disp"]
        rms = float(np.sqrt(np.mean(d * d)))
        r = float(np.corrcoef(X["disp"].ravel(), Y["disp"].ravel())[0, 1])
        print(f"  {name:<40s}  RMS Δ = {rms:5.2f}μm,  Pearson ρ = {r:.4f}")

    # ---- Build the big figure: 3 rows × 3 cols (top + middle), then 9 trace panels ----
    fig = plt.figure(figsize=(20, 18), facecolor="black", constrained_layout=False)
    # Custom GridSpec
    gs = fig.add_gridspec(
        nrows=3, ncols=3,
        height_ratios=[1.2, 1.2, 2.2],
        hspace=0.30, wspace=0.18,
        left=0.05, right=0.97, top=0.95, bottom=0.05,
    )

    # Row 1: heatmaps of disp for A, B, C
    extent = [float(t_anc[0]), float(t_anc[-1]), float(y_anc[0]), float(y_anc[-1])]
    for col, (mot, lbl) in enumerate([(A, A_lbl), (B, B_lbl), (C, C_lbl)]):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(
            mot["disp"].T,                    # (K, T) for y-axis = depth, x-axis = time
            origin="lower", aspect="auto",
            extent=extent, vmin=vmin, vmax=vmax,
            cmap="RdBu_r", interpolation="nearest",
        )
        styled_axes(ax, "time (s)", "y / depth (μm)", title=f"{lbl}   ·   Δy(t, y)")
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label("Δy (μm)", color=TEXT, fontsize=8)
        cb.ax.tick_params(colors=TEXT, labelsize=7)
        cb.outline.set_edgecolor(SPINE)

    # Row 2: difference heatmaps
    for col, (X, Y, lbl) in enumerate([
        (A, B, f"{A_lbl} − {B_lbl}"),
        (A, C, f"{A_lbl} − {C_lbl}"),
        (B, C, f"{B_lbl} − {C_lbl}"),
    ]):
        ax = fig.add_subplot(gs[1, col])
        d = X["disp"] - Y["disp"]
        im = ax.imshow(
            d.T, origin="lower", aspect="auto",
            extent=extent, vmin=-dmax, vmax=dmax,
            cmap="PuOr_r", interpolation="nearest",
        )
        styled_axes(ax, "time (s)", "y / depth (μm)",
                    title=f"Δ-of-Δy:  {lbl}   (RMS={float(np.sqrt(np.mean(d*d))):.2f}μm)")
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label("Δ disp (μm)", color=TEXT, fontsize=8)
        cb.ax.tick_params(colors=TEXT, labelsize=7)
        cb.outline.set_edgecolor(SPINE)

    # Row 3: time traces per y-anchor (3×3 grid of subplots inside this row)
    # We have 9 y-anchors; use a 3×3 sub-grid
    sub_gs = gs[2, :].subgridspec(3, 3, hspace=0.30, wspace=0.18)
    for k in range(K):
        r, c = k // 3, k % 3
        ax = fig.add_subplot(sub_gs[r, c])
        ax.plot(t_anc, A["disp"][:, k], color="#3b8bff", lw=0.9, alpha=0.85, label=A_lbl)
        ax.plot(t_anc, B["disp"][:, k], color="#ff8c3a", lw=0.9, alpha=0.85, label=B_lbl)
        ax.plot(t_anc, C["disp"][:, k], color="#5fd57b", lw=0.9, alpha=0.85, label=C_lbl)
        ax.axhline(0, color=SPINE, lw=0.4, ls="--", alpha=0.4)
        styled_axes(ax, "time (s)" if r == 2 else "", "Δy (μm)" if c == 0 else "",
                    title=f"y-anchor = {y_anc[k]:.0f}μm")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(-vmax, vmax)
        if r == 0 and c == 0:
            ax.legend(loc="upper right", facecolor="black", edgecolor=SPINE,
                       labelcolor=TEXT, fontsize=7, framealpha=0.7)

    fig.suptitle("DREDge motion estimates: MP vs CNN-ep0 vs CNN-ep20   (dataset1_p1, "
                  f"all {2475738:,} spikes, soft σ=4μm grid, SI-canonical preset)",
                  color=TEXT, fontsize=13, y=0.985)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {args.out}  ({args.out.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
