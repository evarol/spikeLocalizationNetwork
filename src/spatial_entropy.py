"""Bookmarked visualization: spatial entropy time series.

For each 1-second time bin, compute the Shannon entropy of the bin's normalized
2D X-Y localization histogram, and plot it as a function of time.

Matches Eq. 10 in the paper:
    p_b(v) = H_b(v) / Σ_v H_b(v)
    H_b    = - Σ_v p_b(v) log(p_b(v) + ε)

Mirrors the line plots in Fig. 2D.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class SpatialEntropyStyle:
    color: str = "#1f1f1f"
    linewidth: float = 0.7
    figsize: tuple[float, float] = (14.0, 3.0)
    dpi: int = 200
    xlabel: str = "time (s)"
    ylabel: str = "spatial entropy H (nats)"


def per_bin_entropy(H: np.ndarray) -> np.ndarray:
    """Shannon entropy of each row of H, where rows are histograms.

    Returns NaN for rows with zero total mass.
    """
    totals = H.sum(axis=1, keepdims=True)
    nonempty = totals[:, 0] > 0
    p = np.zeros_like(H, dtype=np.float64)
    p[nonempty] = H[nonempty] / totals[nonempty]
    entropy = -np.sum(p * np.log(p + EPS), axis=1)
    entropy[~nonempty] = np.nan
    return entropy.astype(np.float32)


def plot_entropy_trace(
    ax: plt.Axes,
    entropy: np.ndarray,
    bin_s: float = 1.0,
    *,
    style: SpatialEntropyStyle = SpatialEntropyStyle(),
    title: str | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    t = np.arange(len(entropy), dtype=np.float64) * bin_s + 0.5 * bin_s
    ax.plot(t, entropy, color=style.color, linewidth=style.linewidth, rasterized=True)
    ax.set_xlabel(style.xlabel)
    ax.set_ylabel(style.ylabel)
    ax.set_xlim(0.0, float(t.max() + 0.5 * bin_s))
    if ylim is not None:
        ax.set_ylim(*ylim)
    if title is not None:
        ax.set_title(title, fontsize=9)


def save_single_panel(
    out_path,
    entropy: np.ndarray,
    bin_s: float,
    *,
    title: str,
    style: SpatialEntropyStyle = SpatialEntropyStyle(),
    ylim: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=style.figsize, constrained_layout=True)
    plot_entropy_trace(ax, entropy, bin_s, style=style, title=title, ylim=ylim)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=style.dpi, bbox_inches="tight")
    plt.close(fig)
