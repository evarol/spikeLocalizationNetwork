"""
Differentiable histogram-based losses for SLN ↔ DREDGE iterative training.

    pairwise_ncc(H, valid_mask, window=30)
        Normalized cross-correlation averaged over time-bin pairs (i, j) with
        |i-j| <= window and both bins valid. Range [-1, 1]; higher = better
        spatial registration.

    spatial_entropy(H, eps=1e-8)
        Per-bin Shannon entropy of the normalized histogram. Range [0, log(G)]
        where G is the voxel count. Higher = more spread.

    tether_loss(pred, baseline)
        Mean squared deviation from the pretrained baseline localization.

These functions assume H has shape (T, ...) where the trailing dims are spatial.
"""

from __future__ import annotations

import torch


def pairwise_ncc(H: torch.Tensor, valid: torch.Tensor, window: int) -> torch.Tensor:
    """
    H: (T, *spatial) — soft histograms per time bin (need not be normalized).
    valid: (T,) bool — True for bins with enough spikes to score.
    window: max |i - j| to include.

    Returns a scalar = mean rho_{ij} over valid pairs (i != j, |i-j|<=window).
    NCC is centered + L2 normalized:
        rho_{ij} = <H_i - mean_i, H_j - mean_j> / (||H_i - mean_i|| ||H_j - mean_j||)
    """
    T = H.shape[0]
    Hf = H.reshape(T, -1)                    # (T, V)
    means = Hf.mean(dim=1, keepdim=True)     # (T, 1)
    centered = Hf - means                    # (T, V)
    norms = centered.norm(dim=1).clamp_min(1e-12)  # (T,)
    normed = centered / norms[:, None]       # (T, V)

    # Build pair indices. For T~1957 and W=30 this is ~60K pairs.
    device = H.device
    idx_i = []
    idx_j = []
    for offset in range(1, min(window + 1, T)):  # clamp window to T to avoid empty ranges
        if T - offset > 0:  # only add if valid range
            i = torch.arange(0, T - offset, device=device)
            j = i + offset
            idx_i.append(i)
            idx_j.append(j)
    if not idx_i:
        return torch.zeros((), device=device, dtype=H.dtype)
    idx_i = torch.cat(idx_i)
    idx_j = torch.cat(idx_j)
    pair_valid = valid[idx_i] & valid[idx_j]
    if not pair_valid.any():
        return torch.zeros((), device=device, dtype=H.dtype)
    idx_i = idx_i[pair_valid]
    idx_j = idx_j[pair_valid]

    rho = (normed[idx_i] * normed[idx_j]).sum(dim=1)   # (P,)
    return rho.mean()


def spatial_entropy(H: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    H: (T, *spatial) — soft histograms (need not be normalized).
    Returns: (T,) Shannon entropy in nats.
    """
    T = H.shape[0]
    Hf = H.reshape(T, -1)
    s = Hf.sum(dim=1, keepdim=True).clamp_min(eps)
    p = Hf / s
    ent = -(p * (p + eps).log()).sum(dim=1)
    return ent


def tether_loss(pred: torch.Tensor, baseline: torch.Tensor,
                axes_weight: torch.Tensor | None = None) -> torch.Tensor:
    """
    pred, baseline: (N, 3) or (N, 2). MSE between them, optionally re-weighted per axis.
    """
    diff = pred - baseline
    if axes_weight is not None:
        diff = diff * axes_weight.to(diff.dtype).to(diff.device)
    return (diff ** 2).mean()


# ---- self-test ---------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    T, V = 50, 31 * 961
    H = torch.rand(T, 31, 961, requires_grad=True)
    valid = torch.ones(T, dtype=torch.bool)
    rho = pairwise_ncc(H, valid, window=10)
    ent = spatial_entropy(H).mean()
    print(f"rho={rho.item():.4f}  entropy_mean={ent.item():.4f}")
    (rho + ent).backward()
    print("grad ok:", H.grad.abs().max().item() > 0)
