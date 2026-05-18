"""
Differentiable soft spatial histograms over a regular voxel grid.

Each spike s contributes a separable Gaussian kernel K_sigma(v - p_s) to the
histogram of its time bin t(s). The kernel is truncated at 3 sigma and
deposited via index_add_, which keeps the operation differentiable in p_s and
parallelizable on GPU.

Public:
    SoftHistogram2D(x_edges, y_edges, sigma_xy, trunc=3.0)
        forward(points: (N, 2), bin_ids: (N,)) -> (T, Gx, Gy)
    SoftHistogram3D(x_edges, y_edges, z_edges, sigma_xyz, trunc=3.0)
        forward(points: (N, 3), bin_ids: (N,)) -> (T, Gx, Gy, Gz)

Edges are 1D tensors of voxel CENTERS (not bin edges). Anything outside
[centers[0]-pitch/2, centers[-1]+pitch/2] is silently dropped from the
histogram (no out-of-grid spillover).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def _gaussian_weights_1d(coords: torch.Tensor, centers: torch.Tensor,
                         sigma: float, trunc_radius: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each input coord, return (idx, w) where idx[s, k] is the voxel index of
    the k-th tap on the truncated kernel of point s and w[s, k] is the
    unnormalized Gaussian weight.

    coords:  (N,)   float
    centers: (G,)   float, must be uniformly spaced
    sigma:   float
    trunc_radius: int, number of voxels on each side -> kernel size = 2*trunc_radius + 1.

    Returns:
        idx: (N, K) long  (out-of-grid indices clamped to 0; mask at 0 weight via valid mask)
        w:   (N, K) float (zeroed where idx is out of grid)
    """
    pitch = float(centers[1] - centers[0])
    # closest voxel index to each coord (rounded to int, can be negative or > G-1)
    rel = (coords - centers[0]) / pitch
    base = torch.round(rel).to(torch.long)            # (N,)
    G = centers.numel()
    K = 2 * trunc_radius + 1
    offsets = torch.arange(-trunc_radius, trunc_radius + 1,
                           device=coords.device, dtype=torch.long)  # (K,)
    idx = base[:, None] + offsets[None, :]            # (N, K)
    valid = (idx >= 0) & (idx < G)                    # (N, K)
    # Distance from coord to voxel center, in micrometers
    voxel_pos = centers[idx.clamp(0, G - 1)]          # (N, K)
    d = coords[:, None] - voxel_pos                   # (N, K)
    w = torch.exp(-0.5 * (d / sigma) ** 2)            # (N, K)
    w = w * valid.to(w.dtype)
    idx = idx.clamp(0, G - 1)                         # safe-index for index_add later
    return idx, w


class SoftHistogram2D(nn.Module):
    """
    Differentiable 2D soft histogram on a uniform grid.
    """

    def __init__(self, x_centers: torch.Tensor, y_centers: torch.Tensor,
                 sigma_x: float, sigma_y: float, trunc: float = 3.0):
        super().__init__()
        self.register_buffer("x_centers", x_centers.float(), persistent=False)
        self.register_buffer("y_centers", y_centers.float(), persistent=False)
        self.sigma_x = float(sigma_x)
        self.sigma_y = float(sigma_y)
        # truncation in voxels (round up)
        px = float(x_centers[1] - x_centers[0])
        py = float(y_centers[1] - y_centers[0])
        self.trunc_x = int(max(1, round(trunc * self.sigma_x / px)))
        self.trunc_y = int(max(1, round(trunc * self.sigma_y / py)))

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (self.x_centers.numel(), self.y_centers.numel())

    def forward(self, points: torch.Tensor, bin_ids: torch.Tensor,
                n_bins: int) -> torch.Tensor:
        """
        points: (N, 2) — (x, y)
        bin_ids: (N,) long — which time bin each point belongs to (0..n_bins-1)
        n_bins: total time bins in this batch
        returns: (n_bins, Gx, Gy) — soft histograms (unnormalized; can be negative? no, weights are >=0)
        """
        Gx = self.x_centers.numel()
        Gy = self.y_centers.numel()
        Kx = 2 * self.trunc_x + 1
        Ky = 2 * self.trunc_y + 1

        idx_x, wx = _gaussian_weights_1d(points[:, 0], self.x_centers, self.sigma_x, self.trunc_x)
        idx_y, wy = _gaussian_weights_1d(points[:, 1], self.y_centers, self.sigma_y, self.trunc_y)
        # Outer product over x and y taps → (N, Kx, Ky) weights, (N, Kx, Ky) flat indices
        w = wx[:, :, None] * wy[:, None, :]                      # (N, Kx, Ky)
        flat_idx = idx_x[:, :, None] * Gy + idx_y[:, None, :]    # (N, Kx, Ky), in [0, Gx*Gy)
        # Per-bin scatter add: combine bin_ids into the linear index
        full_idx = bin_ids[:, None, None] * (Gx * Gy) + flat_idx  # (N, Kx, Ky)
        H = torch.zeros(n_bins * Gx * Gy, dtype=points.dtype, device=points.device)
        H = H.index_add(0, full_idx.reshape(-1), w.reshape(-1))
        return H.reshape(n_bins, Gx, Gy)


class SoftHistogram3D(nn.Module):
    """
    Differentiable 3D soft histogram (diagnostic only by default — heavier than 2D).
    """

    def __init__(self, x_centers: torch.Tensor, y_centers: torch.Tensor, z_centers: torch.Tensor,
                 sigma_x: float, sigma_y: float, sigma_z: float, trunc: float = 3.0):
        super().__init__()
        self.register_buffer("x_centers", x_centers.float(), persistent=False)
        self.register_buffer("y_centers", y_centers.float(), persistent=False)
        self.register_buffer("z_centers", z_centers.float(), persistent=False)
        self.sigma_x = float(sigma_x)
        self.sigma_y = float(sigma_y)
        self.sigma_z = float(sigma_z)
        px = float(x_centers[1] - x_centers[0])
        py = float(y_centers[1] - y_centers[0])
        pz = float(z_centers[1] - z_centers[0])
        self.trunc_x = int(max(1, round(trunc * self.sigma_x / px)))
        self.trunc_y = int(max(1, round(trunc * self.sigma_y / py)))
        self.trunc_z = int(max(1, round(trunc * self.sigma_z / pz)))

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        return (self.x_centers.numel(), self.y_centers.numel(), self.z_centers.numel())

    def forward(self, points: torch.Tensor, bin_ids: torch.Tensor,
                n_bins: int) -> torch.Tensor:
        Gx = self.x_centers.numel()
        Gy = self.y_centers.numel()
        Gz = self.z_centers.numel()
        idx_x, wx = _gaussian_weights_1d(points[:, 0], self.x_centers, self.sigma_x, self.trunc_x)
        idx_y, wy = _gaussian_weights_1d(points[:, 1], self.y_centers, self.sigma_y, self.trunc_y)
        idx_z, wz = _gaussian_weights_1d(points[:, 2], self.z_centers, self.sigma_z, self.trunc_z)
        # (N, Kx, Ky, Kz)
        w = wx[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :]
        flat_idx = (idx_x[:, :, None, None] * (Gy * Gz)
                    + idx_y[:, None, :, None] * Gz
                    + idx_z[:, None, None, :])
        full_idx = bin_ids[:, None, None, None] * (Gx * Gy * Gz) + flat_idx
        H = torch.zeros(n_bins * Gx * Gy * Gz, dtype=points.dtype, device=points.device)
        H = H.index_add(0, full_idx.reshape(-1), w.reshape(-1))
        return H.reshape(n_bins, Gx, Gy, Gz)


def make_centers(low: float, high: float, pitch: float) -> torch.Tensor:
    """Voxel centers from [low, high] with given pitch. Includes both ends approximately."""
    n = int(round((high - low) / pitch)) + 1
    return torch.linspace(low, low + (n - 1) * pitch, n)


def default_2d_grid() -> tuple[torch.Tensor, torch.Tensor]:
    """Default x-y grid for the SLN-DREDGE pipeline (per plan)."""
    return make_centers(-40.0, 80.0, 4.0), make_centers(0.0, 3840.0, 4.0)


def default_3d_grid() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Default x-y-z grid for diagnostics."""
    return (make_centers(-40.0, 80.0, 4.0),
            make_centers(0.0, 3840.0, 4.0),
            make_centers(-60.0, 60.0, 8.0))


# ---- self-test ---------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    xc, yc = default_2d_grid()
    sh = SoftHistogram2D(xc, yc, sigma_x=4.0, sigma_y=4.0)
    N = 5000
    pts = torch.stack([torch.empty(N).uniform_(-30, 70),
                       torch.empty(N).uniform_(0, 3840)], dim=1)
    bins = torch.randint(0, 3, (N,))
    pts.requires_grad_(True)
    H = sh(pts, bins, n_bins=3)
    print("H shape:", H.shape, "sum (should ≈ N for normalized kernel):", H.sum().item(),
          "vs N =", N)  # sum < N because kernel is unnormalized but fine for relative loss
    H.sum().backward()
    print("grad ok:", pts.grad.abs().max().item() > 0)
