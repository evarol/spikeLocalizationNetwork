"""
Thin wrapper around spikeinterface's DREDGE motion estimation, plus utilities to
apply the resulting motion field to a per-spike global localization.

DREDGE estimates a (T_motion, n_spatial_windows) motion array Δy(t, y_anchor)
that the user can interpolate to (t(s), y(s)) per spike. We additionally
estimate a separate scalar Δx(t) per bin via 1D rigid registration of the
x-marginal histograms (DREDGE itself only does y).

Public:
    build_peaks(spike_times, spike_channels, spike_amplitudes) -> structured array
    build_locs(GL) -> structured array (peak_locations dtype)
    run_dredge(recording, peaks, locs, *, fs, rigid=False, win_step_um=400, win_scale_um=450,
               estimate_x_drift=True, x_bin_um=4) -> {Δy: Motion, Δx_per_bin: ndarray, t_bin: ndarray, ...}
    apply_motion_torch(GL, t_idx, motion_dict) -> GL_corrected
    apply_motion_np(GL, t_idx, motion_dict) -> GL_corrected
"""

from __future__ import annotations

import numpy as np


PEAK_DTYPE = np.dtype([
    ("sample_index", "int64"),
    ("channel_index", "int64"),
    ("amplitude", "float64"),
    ("segment_index", "int64"),
])
LOC_DTYPE = np.dtype([("x", "float64"), ("y", "float64"), ("z", "float64")])


def build_peaks(spike_times: np.ndarray, spike_channels: np.ndarray,
                spike_amplitudes: np.ndarray) -> np.ndarray:
    n = len(spike_times)
    peaks = np.empty(n, dtype=PEAK_DTYPE)
    peaks["sample_index"] = spike_times.astype(np.int64, copy=False)
    peaks["channel_index"] = spike_channels.astype(np.int64, copy=False)
    peaks["amplitude"] = spike_amplitudes.astype(np.float64, copy=False)
    peaks["segment_index"] = 0
    return peaks


def build_locs(GL: np.ndarray) -> np.ndarray:
    """GL: (N, 2) or (N, 3). Missing z is set to 0."""
    n = GL.shape[0]
    locs = np.empty(n, dtype=LOC_DTYPE)
    locs["x"] = GL[:, 0].astype(np.float64, copy=False)
    locs["y"] = GL[:, 1].astype(np.float64, copy=False)
    if GL.shape[1] >= 3:
        locs["z"] = GL[:, 2].astype(np.float64, copy=False)
    else:
        locs["z"] = 0.0
    return locs


def run_dredge(recording, peaks: np.ndarray, locs: np.ndarray, *,
               rigid: bool = False, win_step_um: float = 400.0, win_scale_um: float = 450.0,
               bin_s: float = 1.0, estimate_x_drift: bool = True,
               x_bin_um: float = 4.0, x_range: tuple[float, float] = (-40.0, 80.0),
               x_max_shift_um: float = 30.0,
               mincorr: float | None = None,
               verbose: bool = False) -> dict:
    """
    Returns dict with:
        motion: spikeinterface Motion object (y direction)
        Δy_func(t_s, y_s): callable for vectorized motion query [µm]
        Δx_per_bin: (T_bins,) ndarray (zeros if estimate_x_drift=False)
        t_bin_centers: (T_bins,) ndarray seconds
    """
    from spikeinterface.sortingcomponents.motion.motion_estimation import estimate_motion

    extra_kwargs = {}
    if mincorr is not None:
        extra_kwargs["mincorr"] = float(mincorr)

    motion = estimate_motion(
        recording,
        peaks=peaks,
        peak_locations=locs,
        direction="y",
        rigid=rigid,
        win_step_um=win_step_um,
        win_scale_um=win_scale_um,
        method="dredge_ap",
        bin_s=bin_s,
        progress_bar=verbose,
        verbose=verbose,
        **extra_kwargs,
    )

    # ---- x-drift via 1D cross-correlation of per-bin x-marginal histograms ----
    n_bins = int(np.ceil(recording.get_total_duration() / bin_s))
    t_bin_centers = (np.arange(n_bins) + 0.5) * bin_s
    Δx_per_bin = np.zeros(n_bins, dtype=np.float64)
    if estimate_x_drift:
        fs = float(recording.get_sampling_frequency())
        t_s = peaks["sample_index"] / fs
        bin_id = np.clip((t_s / bin_s).astype(np.int64), 0, n_bins - 1)
        x_edges = np.arange(x_range[0], x_range[1] + x_bin_um, x_bin_um)
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        # Build per-bin histogram of x-locations
        H = np.zeros((n_bins, x_centers.size), dtype=np.float32)
        for b in range(n_bins):
            mask = bin_id == b
            if not mask.any():
                continue
            H[b], _ = np.histogram(locs["x"][mask], bins=x_edges)
        ref = H.sum(axis=0)
        ref_n = (ref - ref.mean()) / (ref.std() + 1e-12)
        n_x = x_centers.size
        # Find shift in voxels that maximizes correlation; clamp to ±max_shift
        max_shift_vox = int(round(x_max_shift_um / x_bin_um))
        for b in range(n_bins):
            if H[b].sum() < 50:
                continue
            cur_n = (H[b] - H[b].mean()) / (H[b].std() + 1e-12)
            best, best_s = -np.inf, 0
            for s in range(-max_shift_vox, max_shift_vox + 1):
                # roll cur by -s, compute corr with ref
                lo = max(0, s)
                hi = min(n_x, n_x + s)
                a = ref_n[lo:hi]
                b_arr = cur_n[lo - s:hi - s]
                if a.size < 4:
                    continue
                c = float(np.dot(a, b_arr) / a.size)
                if c > best:
                    best, best_s = c, s
            Δx_per_bin[b] = best_s * x_bin_um

    # ---- Build a vectorized Δy(t, y) lookup ----
    # spikeinterface.Motion exposes .displacement (n_t, n_y) and .temporal_bins, .spatial_bins per segment.
    # Note: spatial_bins_um is shared across all segments; temporal_bins_s and displacement are per-segment.
    seg = 0
    y_anchors = motion.spatial_bins_um                       # (n_y_anchor,) — shared across segments
    t_anchors = motion.temporal_bins_s[seg]                  # (n_t,) — per segment
    disp = motion.displacement[seg]                          # (n_t, n_y_anchor) — per segment

    def Δy_func(t_s: np.ndarray, y_s: np.ndarray) -> np.ndarray:
        """Bilinear interpolation of disp at (t_s, y_s)."""
        t_s = np.asarray(t_s, dtype=np.float64)
        y_s = np.asarray(y_s, dtype=np.float64)
        # Indices via searchsorted, clamped
        ti = np.clip(np.searchsorted(t_anchors, t_s) - 1, 0, len(t_anchors) - 2)
        yi = np.clip(np.searchsorted(y_anchors, y_s) - 1, 0, len(y_anchors) - 2)
        ta0 = t_anchors[ti]
        ta1 = t_anchors[ti + 1]
        ya0 = y_anchors[yi]
        ya1 = y_anchors[yi + 1]
        wt = np.clip((t_s - ta0) / np.maximum(ta1 - ta0, 1e-9), 0, 1)
        wy = np.clip((y_s - ya0) / np.maximum(ya1 - ya0, 1e-9), 0, 1)
        d00 = disp[ti, yi]
        d10 = disp[ti + 1, yi]
        d01 = disp[ti, yi + 1]
        d11 = disp[ti + 1, yi + 1]
        return ((1 - wt) * (1 - wy) * d00 + wt * (1 - wy) * d10
                + (1 - wt) * wy * d01 + wt * wy * d11)

    return {
        "motion": motion,
        "Δy_func": Δy_func,
        "Δx_per_bin": Δx_per_bin,
        "t_bin_centers": t_bin_centers,
        "y_anchors": np.asarray(y_anchors),
        "t_anchors": np.asarray(t_anchors),
        "disp": np.asarray(disp),
    }


def apply_motion_np(GL: np.ndarray, t_s: np.ndarray, t_idx: np.ndarray,
                    motion: dict) -> np.ndarray:
    """
    GL: (N, 2 or 3); t_s: (N,) seconds; t_idx: (N,) bin index used for Δx.
    Returns drift-corrected GL (subtracts motion).
    """
    GL = GL.astype(np.float64, copy=True)
    Δy = motion["Δy_func"](t_s, GL[:, 1])
    GL[:, 1] -= Δy
    GL[:, 0] -= motion["Δx_per_bin"][t_idx]
    return GL


def apply_motion_torch(GL: 'torch.Tensor', t_s: np.ndarray, t_idx: np.ndarray,
                       motion: dict, device=None) -> 'torch.Tensor':
    """
    Differentiable in GL but NOT in motion (motion is treated as a constant
    offset). We pre-compute Δy and Δx as torch tensors and subtract.
    """
    import torch
    Δy = motion["Δy_func"](t_s, GL[:, 1].detach().cpu().numpy())
    Δx = motion["Δx_per_bin"][t_idx]
    Δy_t = torch.as_tensor(Δy, dtype=GL.dtype, device=GL.device if device is None else device)
    Δx_t = torch.as_tensor(Δx, dtype=GL.dtype, device=GL.device if device is None else device)
    out = GL.clone()
    out[:, 0] = GL[:, 0] - Δx_t
    out[:, 1] = GL[:, 1] - Δy_t
    return out


# ---- self-test (requires data on disk; skipped if missing) -----------------------
if __name__ == "__main__":
    import os
    if os.path.exists("data/dataset1") and os.path.exists("results/spike_times.npy"):
        from window_video_common import load_preprocessed_recording
        rec = load_preprocessed_recording("data/dataset1")
        st = np.load("results/spike_times.npy")
        sc = np.load("results/spike_channels.npy")
        sa = np.load("results/spike_amplitudes.npy")
        # Use a 60-s slice for self-test
        fs = float(rec.get_sampling_frequency())
        mask = (st < int(60 * fs)) & (st >= 0)
        peaks = build_peaks(st[mask], sc[mask], sa[mask])
        # Quick monopolar-from-cache substitute: use channel locations as locs
        ch = np.load("results/channel_locations.npy")
        locs = np.empty(mask.sum(), dtype=LOC_DTYPE)
        locs["x"] = ch[sc[mask], 0]
        locs["y"] = ch[sc[mask], 1]
        locs["z"] = 0.0
        sub = rec.frame_slice(0, int(60 * fs))
        out = run_dredge(sub, peaks, locs, rigid=True, verbose=False)
        print("OK; Δy disp shape:", out["disp"].shape, "Δx range:",
              out["Δx_per_bin"].min(), out["Δx_per_bin"].max())
    else:
        print("Skipped self-test: data/results not present.")
