"""Bookmarked visualization: depth raster.

A spike scatter on (x=time, y=depth) where the color is log10(α) on a
white→black scale. Mirrors the rows of Fig. 2A in the paper.

Defaults here define the canonical look; tweak via the CLI in
make_depth_rasters.py or by passing kwargs to ``plot_depth_raster``.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class DepthRasterStyle:
    cmap: str = "binary"          # white→black
    s: float = 0.5                 # scatter marker size
    alpha: float = 0.5             # alpha blending; dense bands darken
    figsize: tuple[float, float] = (14.0, 4.0)
    dpi: int = 200
    cbar_label: str = "log₁₀ α"
    xlabel: str = "time (s)"
    ylabel: str = "depth y (μm)"


def percentile_clip(log_alpha: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> tuple[float, float]:
    return float(np.percentile(log_alpha, lo)), float(np.percentile(log_alpha, hi))


def plot_depth_raster(
    ax: plt.Axes,
    t_s: np.ndarray,
    y_um: np.ndarray,
    log_alpha: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    style: DepthRasterStyle = DepthRasterStyle(),
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> plt.cm.ScalarMappable:
    """Draw one depth-raster panel on ``ax``. Returns the mappable (for colorbar)."""
    order = np.argsort(log_alpha)
    sc = ax.scatter(
        t_s[order], y_um[order],
        c=log_alpha[order], cmap=style.cmap,
        vmin=vmin, vmax=vmax,
        s=style.s, alpha=style.alpha,
        linewidths=0, rasterized=True,
    )
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(style.xlabel)
    ax.set_ylabel(style.ylabel)
    if title is not None:
        ax.set_title(title, fontsize=9)
    return sc


def save_single_panel(
    out_path,
    t_s: np.ndarray,
    y_um: np.ndarray,
    log_alpha: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    title: str,
    style: DepthRasterStyle = DepthRasterStyle(),
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=style.figsize, constrained_layout=True)
    sc = plot_depth_raster(
        ax, t_s, y_um, log_alpha,
        vmin=vmin, vmax=vmax, style=style,
        title=title, xlim=xlim, ylim=ylim,
    )
    cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label(style.cbar_label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=style.dpi, bbox_inches="tight")
    plt.close(fig)
