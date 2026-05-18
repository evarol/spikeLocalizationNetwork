"""Bookmarked visualization: X-Y pairwise correlation matrix.

For each 1-second time bin, build a 2D histogram of spike (x, y) localizations.
The visualization is the T×T matrix of Pearson correlation coefficients between
each pair of bin histograms (flattened to vectors).

This is *Pearson*, not cross-correlation: no spatial shift is searched for,
each bin is compared to every other bin at zero lag.

Mirrors the panels in Fig. 2C of the paper, but without windowing — every
pair (i, j) is computed, producing the full T×T matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class XYCorrStyle:
    cmap: str = "Reds"
    figsize: tuple[float, float] = (6.0, 5.5)
    dpi: int = 200
    cbar_label: str = "Pearson ρ"
    xlabel: str = "time bin j (s)"
    ylabel: str = "time bin i (s)"


@dataclass(frozen=True)
class XYHistGrid:
    x_lo: float = -80.0
    x_hi: float = 120.0
    x_bin_um: float = 8.0
    y_lo: float = -40.0
    y_hi: float = 3860.0
    y_bin_um: float = 8.0

    @property
    def x_edges(self) -> np.ndarray:
        return np.arange(self.x_lo, self.x_hi + self.x_bin_um, self.x_bin_um)

    @property
    def y_edges(self) -> np.ndarray:
        return np.arange(self.y_lo, self.y_hi + self.y_bin_um, self.y_bin_um)

    @property
    def x_centers(self) -> np.ndarray:
        n = int(round((self.x_hi - self.x_lo) / self.x_bin_um)) + 1
        return self.x_lo + np.arange(n) * self.x_bin_um

    @property
    def y_centers(self) -> np.ndarray:
        n = int(round((self.y_hi - self.y_lo) / self.y_bin_um)) + 1
        return self.y_lo + np.arange(n) * self.y_bin_um


def build_xy_histograms(
    t_bin: np.ndarray,      # int per spike, in [0, T)
    x_um: np.ndarray,
    y_um: np.ndarray,
    T: int,
    grid: XYHistGrid,
    dtype=np.float32,
    soft_sigma_um: float = 0.0,
) -> np.ndarray:
    """Return H of shape (T, D).

    If soft_sigma_um == 0, builds a hard 2-D histogram (one bin per spike) on
    voxel edges from grid.x_edges, grid.y_edges. D = nx_edge_bins * ny_edge_bins.

    If soft_sigma_um > 0, builds a separable-Gaussian soft histogram on voxel
    CENTERS (x_centers, y_centers). Mass per spike is split across centers
    according to a 2-D isotropic Gaussian of bandwidth σ. D = nxc * nyc.
    """
    if soft_sigma_um <= 0.0:
        return _hard_xy_hist(t_bin, x_um, y_um, T, grid, dtype)
    return _soft_xy_hist(t_bin, x_um, y_um, T, grid, soft_sigma_um, dtype)


def _hard_xy_hist(t_bin, x_um, y_um, T, grid, dtype):
    xe, ye = grid.x_edges, grid.y_edges
    nx, ny = len(xe) - 1, len(ye) - 1
    D = nx * ny
    in_x = (x_um >= xe[0]) & (x_um < xe[-1])
    in_y = (y_um >= ye[0]) & (y_um < ye[-1])
    keep = in_x & in_y
    if not keep.all():
        t_bin = t_bin[keep]; x_um = x_um[keep]; y_um = y_um[keep]
    xi = np.searchsorted(xe, x_um, side="right") - 1
    yi = np.searchsorted(ye, y_um, side="right") - 1
    flat = xi * ny + yi
    idx = t_bin.astype(np.int64) * D + flat.astype(np.int64)
    counts = np.bincount(idx, minlength=T * D)
    return counts.reshape(T, D).astype(dtype)


def _soft_xy_hist(t_bin, x_um, y_um, T, grid, sigma, dtype):
    """Separable-Gaussian scatter onto voxel centers, accumulated per frame.

    Memory-bounded: builds H one frame at a time and only keeps (T, D) at the end.
    """
    xc = grid.x_centers.astype(np.float32)
    yc = grid.y_centers.astype(np.float32)
    nxc, nyc = len(xc), len(yc)
    D = nxc * nyc

    half_w_x = int(np.ceil(3.0 * sigma / grid.x_bin_um))
    half_w_y = int(np.ceil(3.0 * sigma / grid.y_bin_um))
    inv2 = 1.0 / (2.0 * sigma * sigma)

    # Keep only spikes within 3σ of grid bounds.
    in_x = (x_um >= xc[0] - 3 * sigma) & (x_um <= xc[-1] + 3 * sigma)
    in_y = (y_um >= yc[0] - 3 * sigma) & (y_um <= yc[-1] + 3 * sigma)
    keep = in_x & in_y
    t_bin = t_bin[keep].astype(np.int64)
    x_um = x_um[keep].astype(np.float32)
    y_um = y_um[keep].astype(np.float32)

    # Sort by frame so we can slice contiguously.
    order = np.argsort(t_bin, kind="stable")
    t_bin = t_bin[order]; x_um = x_um[order]; y_um = y_um[order]

    counts = np.bincount(t_bin, minlength=T)
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)

    xi0 = np.clip(np.round((x_um - xc[0]) / grid.x_bin_um).astype(np.int64), 0, nxc - 1)
    yi0 = np.clip(np.round((y_um - yc[0]) / grid.y_bin_um).astype(np.int64), 0, nyc - 1)

    x_offsets = np.arange(-half_w_x, half_w_x + 1, dtype=np.int64)
    y_offsets = np.arange(-half_w_y, half_w_y + 1, dtype=np.int64)

    H = np.zeros((T, D), dtype=dtype)
    H_view = H.reshape(T, nxc, nyc)

    for t in range(T):
        a, b = offsets[t], offsets[t + 1]
        if b == a:
            continue
        xi0_t = xi0[a:b]; yi0_t = yi0[a:b]
        xs = x_um[a:b]; ys = y_um[a:b]
        frame = np.zeros((nxc, nyc), dtype=np.float64)
        for dx in x_offsets:
            xi = xi0_t + dx
            valid_x = (xi >= 0) & (xi < nxc)
            if not valid_x.any():
                continue
            xi_c = np.clip(xi, 0, nxc - 1)
            cx = xc[xi_c]
            wx = np.exp(-((cx - xs) ** 2) * inv2)
            wx[~valid_x] = 0.0
            for dy in y_offsets:
                yi = yi0_t + dy
                valid = valid_x & (yi >= 0) & (yi < nyc)
                if not valid.any():
                    continue
                yi_c = np.clip(yi, 0, nyc - 1)
                cy = yc[yi_c]
                wy = np.exp(-((cy - ys) ** 2) * inv2)
                w = wx * wy
                w[~valid] = 0.0
                flat = xi_c * nyc + yi_c                      # clipped → in-bounds
                np.add.at(frame.reshape(-1), flat, w)
        H_view[t] = frame
    return H


def pairwise_pearson(H: np.ndarray) -> np.ndarray:
    """T×T Pearson correlation between rows of H. Empty rows → 0 off-diagonal."""
    Hc = H - H.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(Hc, axis=1, keepdims=True)
    safe = norms > 0
    Hn = np.where(safe, Hc / np.where(safe, norms, 1.0), 0.0)
    C = Hn @ Hn.T
    # Force diagonal exactly 1 where row was non-empty (otherwise 0).
    diag = np.where(safe[:, 0], 1.0, 0.0)
    np.fill_diagonal(C, diag)
    return C.astype(np.float32)


def plot_corr_matrix(
    ax: plt.Axes,
    C: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float = 1.0,
    style: XYCorrStyle = XYCorrStyle(),
    title: str | None = None,
) -> plt.cm.ScalarMappable:
    if vmin is None:
        # robust lower bound: drop top 0.1% of off-diagonal noise
        off = C[~np.eye(C.shape[0], dtype=bool)]
        vmin = float(np.percentile(off, 1))
    im = ax.imshow(
        C, origin="lower", aspect="equal",
        cmap=style.cmap, vmin=vmin, vmax=vmax,
        interpolation="nearest", rasterized=True,
    )
    ax.set_xlabel(style.xlabel)
    ax.set_ylabel(style.ylabel)
    if title is not None:
        ax.set_title(title, fontsize=9)
    return im


def save_single_panel(
    out_path,
    C: np.ndarray,
    *,
    title: str,
    style: XYCorrStyle = XYCorrStyle(),
    vmin: float | None = None,
    vmax: float = 1.0,
) -> None:
    fig, ax = plt.subplots(figsize=style.figsize, constrained_layout=True)
    im = plot_corr_matrix(ax, C, vmin=vmin, vmax=vmax, style=style, title=title)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(style.cbar_label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=style.dpi, bbox_inches="tight")
    plt.close(fig)
