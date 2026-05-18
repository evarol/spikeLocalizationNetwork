"""
Shared helpers for relative-XYZ waveform localization experiments.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch import nn

from window_video_common import load_localization_results

RESULTS_DIR = "results"
TPCA_LOC_DIR = os.path.join(RESULTS_DIR, "tpca_monopolar_full")
DATASET_DIR = os.path.join(RESULTS_DIR, "relative_xyz_dataset_500k")
MODEL_DIR = os.path.join(RESULTS_DIR, "relative_xyz_model")

RNG_SEED = 42
N_NEIGHBORS = 10
MS_BEFORE = 1.0
MS_AFTER = 2.0
TRAIN_FRAMES = (0, 1370)
VAL_FRAMES = (1370, 1664)
TEST_FRAMES = (1664, 1958)
TRAIN_SAMPLES = 350_000
VAL_SAMPLES = 75_000
TEST_SAMPLES = 75_000


def load_tpca_targets(results_dir: str = RESULTS_DIR, loc_dir: str = TPCA_LOC_DIR):
    return load_localization_results(results_dir=results_dir, loc_dir=loc_dir)


def sample_window(fs: float, ms_before: float = MS_BEFORE, ms_after: float = MS_AFTER) -> np.ndarray:
    n_before = int(round(ms_before / 1000.0 * fs))
    n_after = int(round(ms_after / 1000.0 * fs))
    return np.arange(-n_before, n_after, dtype=np.int64)


def split_frame_ranges() -> Dict[str, Tuple[int, int]]:
    return {
        "train": TRAIN_FRAMES,
        "val": VAL_FRAMES,
        "test": TEST_FRAMES,
    }


def split_sizes() -> Dict[str, int]:
    return {
        "train": TRAIN_SAMPLES,
        "val": VAL_SAMPLES,
        "test": TEST_SAMPLES,
    }


def build_neighborhood_lookup(channel_locations: np.ndarray, n_neighbors: int = N_NEIGHBORS):
    n_channels = channel_locations.shape[0]
    channel_lookup = np.zeros((n_channels, n_neighbors), dtype=np.int32)
    anchor_lookup = np.zeros((n_channels, 3), dtype=np.float32)

    for peak_channel in range(n_channels):
        dists = np.linalg.norm(channel_locations - channel_locations[peak_channel], axis=1)
        nearest = np.argsort(dists)[:n_neighbors]
        locs = channel_locations[nearest]
        order = np.lexsort((locs[:, 0], locs[:, 1]))
        ordered = nearest[order]
        channel_lookup[peak_channel] = ordered.astype(np.int32)
        anchor_lookup[peak_channel, :2] = channel_locations[ordered].mean(axis=0).astype(np.float32)
        anchor_lookup[peak_channel, 2] = 0.0

    return channel_lookup, anchor_lookup


def valid_spike_mask(spike_times: np.ndarray, total_samples: int, sample_offsets: np.ndarray) -> np.ndarray:
    start = spike_times.astype(np.int64) + int(sample_offsets[0])
    end = spike_times.astype(np.int64) + int(sample_offsets[-1])
    return (start >= 0) & (end < total_samples)


def sample_spike_indices(
    frame_ids: np.ndarray,
    valid_mask: np.ndarray,
    seed: int = RNG_SEED,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    sampled = {}
    for split_name, (frame_start, frame_end) in split_frame_ranges().items():
        candidates = np.flatnonzero(
            valid_mask & (frame_ids >= frame_start) & (frame_ids < frame_end)
        )
        n_needed = split_sizes()[split_name]
        if candidates.size < n_needed:
            raise RuntimeError(
                f"Not enough valid spikes for {split_name}: need {n_needed}, have {candidates.size}"
            )
        chosen = rng.choice(candidates, size=n_needed, replace=False)
        chosen.sort()
        sampled[split_name] = chosen.astype(np.int64)
    return sampled


def extract_waveforms_by_peak_channel(
    traces: np.ndarray,
    trace_start: int,
    spike_times: np.ndarray,
    peak_channels: np.ndarray,
    channel_lookup: np.ndarray,
    sample_offsets: np.ndarray,
) -> np.ndarray:
    spike_times = spike_times.astype(np.int64, copy=False)
    peak_channels = peak_channels.astype(np.int32, copy=False)
    n_spikes = spike_times.size
    t_size = sample_offsets.size
    n_neighbors = channel_lookup.shape[1]
    waveforms = np.zeros((n_spikes, n_neighbors, t_size), dtype=np.float32)
    trace_len = traces.shape[0]

    rel_times = spike_times - int(trace_start)
    for peak_channel in np.unique(peak_channels):
        peak_channel = int(peak_channel)
        group_idx = np.flatnonzero(peak_channels == peak_channel)
        channel_ids = channel_lookup[peak_channel]
        offsets = rel_times[group_idx, None] + sample_offsets[None, :]
        valid = (offsets[:, 0] >= 0) & (offsets[:, -1] < trace_len)

        if np.any(valid):
            valid_idx = group_idx[valid]
            valid_offsets = offsets[valid]
            wf = traces[valid_offsets[:, :, None], channel_ids[None, None, :]]
            waveforms[valid_idx] = np.transpose(wf, (0, 2, 1)).astype(np.float32, copy=False)

        if not np.all(valid):
            invalid_idx = group_idx[~valid]
            invalid_offsets = offsets[~valid]
            for out_pos, row_offsets in zip(invalid_idx, invalid_offsets):
                snippet = np.zeros((t_size, n_neighbors), dtype=np.float32)
                valid_samples = (row_offsets >= 0) & (row_offsets < trace_len)
                if np.any(valid_samples):
                    snippet[valid_samples] = traces[row_offsets[valid_samples]][:, channel_ids]
                waveforms[out_pos] = snippet.T

    return waveforms


def rmse_per_axis(true_xyz: np.ndarray, pred_xyz: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((pred_xyz - true_xyz) ** 2, axis=0))


def rmse_3d(true_xyz: np.ndarray, pred_xyz: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((pred_xyz - true_xyz) ** 2, axis=1))))


def metric_dict(true_xyz: np.ndarray, pred_xyz: np.ndarray) -> Dict[str, float]:
    axis_rmse = rmse_per_axis(true_xyz, pred_xyz)
    return {
        "rmse_x": float(axis_rmse[0]),
        "rmse_y": float(axis_rmse[1]),
        "rmse_z": float(axis_rmse[2]),
        "rmse_xyz": rmse_3d(true_xyz, pred_xyz),
    }


def baseline_from_anchor(anchors: np.ndarray, true_xyz: np.ndarray) -> Dict[str, float]:
    return metric_dict(true_xyz, anchors)


def save_json(path: str, payload: Dict):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


class RelativeXYZNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(N_NEIGHBORS, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, N_NEIGHBORS, sample_window(30_000.0).size, dtype=torch.float32)
            n_flat = int(np.prod(self.features(dummy).shape[1:]))

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3),
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(waveforms))


def set_torch_determinism(seed: int = RNG_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_label_map() -> Dict[int, str]:
    return {
        0: "train",
        1: "val",
        2: "test",
    }


def split_name_to_id() -> Dict[str, int]:
    return {
        "train": 0,
        "val": 1,
        "test": 2,
    }


def chunk_bounds(total: int, chunk_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, total, chunk_size):
        yield start, min(total, start + chunk_size)
