"""
SLN ↔ DREDGE iterative self-registration trainer.

Outer loop alternates:
  1. DREDGE step  — estimate motion T(t,y) from current SLN-predicted GL.
  2. SLN step     — train RelativeXYZNet to maximize pairwise NCC of drift-corrected
                    soft histograms, regularized by per-bin spatial entropy and
                    tethered to the pretrained baseline.

Acceptance criteria (tracked but not gated mid-run):
  - mean pairwise NCC gain on val bins
  - mean spatial entropy maintenance/gain on val bins

Usage:
  python train_sln_dredge_iterative.py \
    --outer-iters 8 --sln-epochs 5 --batch-bins 32 \
    --lambda-corr 1.0 --lambda-ent 0.1 --lambda-teth 0.01 \
    --out-dir results/sln_dredge_iter
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from typing import Dict

import numpy as np
import torch
from torch import nn

from relative_xyz_common import (
    DATASET_DIR,
    MODEL_DIR,
    RelativeXYZNet,
    set_torch_determinism,
)
from soft_histogram import SoftHistogram2D, default_2d_grid
from histogram_losses import pairwise_ncc, spatial_entropy, tether_loss
from dredge_wrapper import build_peaks, build_locs, run_dredge


# ----------------------------------------------------------------------------
# Data loading


def load_dataset(dataset_dir: str):
    waveforms = np.load(os.path.join(dataset_dir, "waveforms.npy"), mmap_mode="r")
    anchors = np.load(os.path.join(dataset_dir, "anchors.npy"))            # (N, 3)
    frame_ids = np.load(os.path.join(dataset_dir, "frame_ids.npy"))        # (N,) int32
    spike_indices = np.load(os.path.join(dataset_dir, "spike_indices.npy"))
    split = np.load(os.path.join(dataset_dir, "split.npy"))                # 0=train,1=val,2=test
    targets_global = np.load(os.path.join(dataset_dir, "targets_global.npy"))
    with open(os.path.join(dataset_dir, "manifest.json")) as f:
        manifest = json.load(f)
    return {
        "waveforms": waveforms,
        "anchors": anchors,
        "frame_ids": frame_ids.astype(np.int64),
        "spike_indices": spike_indices,
        "split": split,
        "targets_global": targets_global,
        "manifest": manifest,
    }


def build_frame_offsets(frame_ids: np.ndarray, n_frames: int) -> np.ndarray:
    counts = np.bincount(frame_ids, minlength=n_frames)
    offsets = np.concatenate([[0], counts.cumsum()]).astype(np.int64)
    return offsets


def sort_by_frame(ds: dict, n_frames: int):
    perm = np.argsort(ds["frame_ids"], kind="stable")
    # Fast path: if frame_ids are already sorted, leave waveforms as mmap to avoid
    # allocating 9 GB+ in RAM (which can swap-thrash on long runs). Mmap'd reads of
    # contiguous block_idx ranges are nearly as fast as in-memory slicing for our
    # access pattern, and the OS page cache manages memory pressure naturally.
    is_identity = np.array_equal(perm, np.arange(len(perm)))
    if not is_identity:
        ds["waveforms"] = np.asarray(ds["waveforms"])[perm]
        ds["anchors"] = ds["anchors"][perm]
        ds["frame_ids"] = ds["frame_ids"][perm]
        ds["spike_indices"] = ds["spike_indices"][perm]
        ds["split"] = ds["split"][perm]
        ds["targets_global"] = ds["targets_global"][perm]
    # If already sorted, the arrays above are correct as-is; only build offsets.
    ds["frame_offsets"] = build_frame_offsets(ds["frame_ids"], n_frames)
    return ds


# ----------------------------------------------------------------------------
# Model + inference


def load_pretrained_sln(checkpoint_path: str, device) -> tuple[RelativeXYZNet, float]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = RelativeXYZNet().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    waveform_scale = float(ckpt["waveform_scale"])
    return model, waveform_scale


@torch.no_grad()
def predict_all(model: RelativeXYZNet, waveforms: np.ndarray, anchors: np.ndarray,
                waveform_scale: float, device, batch_size: int = 4096) -> np.ndarray:
    """Predict global localization for every spike in the (sorted) dataset."""
    model.eval()
    out = np.zeros((waveforms.shape[0], 3), dtype=np.float32)
    for s in range(0, waveforms.shape[0], batch_size):
        e = min(s + batch_size, waveforms.shape[0])
        W = torch.from_numpy(waveforms[s:e].astype(np.float32) / waveform_scale).to(device)
        rel = model(W).cpu().numpy()
        out[s:e] = rel + anchors[s:e]
    return out


# ----------------------------------------------------------------------------
# DREDGE step


def dredge_step(GL_full: np.ndarray, spike_times: np.ndarray, spike_channels: np.ndarray,
                spike_amplitudes: np.ndarray, recording, *,
                rigid: bool, win_step_um: float, win_scale_um: float,
                bin_s: float, estimate_x_drift: bool) -> dict:
    peaks = build_peaks(spike_times, spike_channels, spike_amplitudes)
    locs = build_locs(GL_full)
    return run_dredge(
        recording, peaks, locs,
        rigid=rigid, win_step_um=win_step_um, win_scale_um=win_scale_um,
        bin_s=bin_s, estimate_x_drift=estimate_x_drift,
    )


def gl_to_corrected(GL: np.ndarray, frame_ids: np.ndarray, motion: dict, fs: float) -> np.ndarray:
    """Apply motion correction (numpy, no grad)."""
    t_s = frame_ids.astype(np.float64) + 0.5  # 1-second bins, time at bin center
    Δy = motion["Δy_func"](t_s, GL[:, 1].astype(np.float64))
    out = GL.astype(np.float64, copy=True)
    out[:, 1] -= Δy
    out[:, 0] -= motion["Δx_per_bin"][np.clip(frame_ids.astype(np.int64), 0,
                                               len(motion["Δx_per_bin"]) - 1)]
    return out.astype(np.float32)


# ----------------------------------------------------------------------------
# SLN step


class BinBlockSampler:
    """Yield contiguous bin-blocks. Each minibatch = one block of B bins."""

    def __init__(self, frame_offsets: np.ndarray, bin_range: tuple[int, int],
                 block_size: int, min_spikes_per_bin: int, seed: int = 0):
        self.frame_offsets = frame_offsets
        self.bin_lo, self.bin_hi = bin_range
        self.block_size = int(block_size)
        self.min_spikes = int(min_spikes_per_bin)
        self.rng = np.random.default_rng(seed)

    def n_starts(self) -> int:
        return max(0, (self.bin_hi - self.bin_lo) - self.block_size + 1)

    def epoch(self):
        n = self.n_starts()
        if n <= 0:
            return
        order = self.rng.permutation(n) + self.bin_lo
        for start in order:
            yield int(start)

    def gather(self, start: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (spike_idx, bin_id_local, valid_mask). spike_idx is a flat 1D array
        into the sorted dataset; bin_id_local is in [0, block_size)."""
        end = start + self.block_size
        offs = self.frame_offsets
        block_lo = int(offs[start])
        block_hi = int(offs[end])
        n_block = block_hi - block_lo
        if n_block == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64), np.zeros(self.block_size, bool)
        spike_idx = np.arange(block_lo, block_hi, dtype=np.int64)
        # bin id local: which slot within [start, end)
        cum = offs[start:end + 1] - block_lo
        bin_id_local = np.repeat(np.arange(self.block_size), np.diff(cum))
        counts_per_bin = np.diff(cum)
        valid = counts_per_bin >= self.min_spikes
        return spike_idx, bin_id_local.astype(np.int64), valid


def sln_loss(model, batch_W, batch_anchors, batch_baseline_GL, bin_id_local, n_bins_in_block,
             motion, t_s_block, t_idx_block,
             soft_hist: SoftHistogram2D,
             waveform_scale: float,
             lambda_corr: float, lambda_ent: float, lambda_teth: float,
             window: int, valid_mask: torch.Tensor) -> dict:
    rel = model(batch_W / waveform_scale)             # (M, 3)
    GL_xy = rel[:, :2] + batch_anchors[:, :2]         # (M, 2)
    # Apply motion (constant offsets, NOT differentiable through DREDGE)
    Δy = torch.as_tensor(motion["Δy_func"](t_s_block, GL_xy[:, 1].detach().cpu().numpy()),
                         dtype=GL_xy.dtype, device=GL_xy.device)
    Δx = torch.as_tensor(motion["Δx_per_bin"][t_idx_block],
                         dtype=GL_xy.dtype, device=GL_xy.device)
    GL_corr = torch.stack([GL_xy[:, 0] - Δx, GL_xy[:, 1] - Δy], dim=1)
    H = soft_hist(GL_corr, bin_id_local, n_bins_in_block)         # (T, Gx, Gy)
    rho = pairwise_ncc(H, valid_mask, window=window)
    ent = spatial_entropy(H)[valid_mask].mean() if valid_mask.any() else torch.tensor(0.0, device=H.device)
    teth = tether_loss(GL_xy, batch_baseline_GL[:, :2])
    L = -lambda_corr * rho - lambda_ent * ent + lambda_teth * teth
    return {"loss": L, "rho": rho.detach(), "ent": ent.detach(), "teth": teth.detach(),
            "n_valid_bins": int(valid_mask.sum().item())}


def evaluate(model, loader: BinBlockSampler, ds: dict, motion: dict,
             soft_hist: SoftHistogram2D, waveform_scale: float, fs: float,
             window: int, device,
             max_blocks: int = 50) -> dict:
    """Compute mean rho and mean entropy on val bins, no grad."""
    model.eval()
    rhos, ents, n_blocks = [], [], 0
    with torch.no_grad():
        for start in loader.epoch():
            spike_idx, bin_id_local, valid = loader.gather(start)
            if not valid.any():
                continue
            W = torch.from_numpy(ds["waveforms"][spike_idx].astype(np.float32) / waveform_scale).to(device)
            anchors = torch.from_numpy(ds["anchors"][spike_idx].astype(np.float32)).to(device)
            rel = model(W)
            GL_xy = rel[:, :2] + anchors[:, :2]
            t_s = (ds["frame_ids"][spike_idx].astype(np.float64) + 0.5)
            t_idx = ds["frame_ids"][spike_idx].astype(np.int64)
            Δy = torch.as_tensor(motion["Δy_func"](t_s, GL_xy[:, 1].detach().cpu().numpy()),
                                 dtype=GL_xy.dtype, device=device)
            Δx = torch.as_tensor(motion["Δx_per_bin"][t_idx],
                                 dtype=GL_xy.dtype, device=device)
            GL_corr = torch.stack([GL_xy[:, 0] - Δx, GL_xy[:, 1] - Δy], dim=1)
            bin_id_t = torch.from_numpy(bin_id_local).to(device)
            valid_t = torch.from_numpy(valid).to(device)
            H = soft_hist(GL_corr, bin_id_t, loader.block_size)
            rho = pairwise_ncc(H, valid_t, window=window)
            ent = spatial_entropy(H)[valid_t].mean()
            rhos.append(float(rho))
            ents.append(float(ent))
            n_blocks += 1
            if n_blocks >= max_blocks:
                break
    if not rhos:
        return {"rho_mean": float("nan"), "ent_mean": float("nan"), "n_blocks": 0}
    return {"rho_mean": float(np.mean(rhos)), "ent_mean": float(np.mean(ents)),
            "n_blocks": n_blocks}


# ----------------------------------------------------------------------------
# Main


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default=DATASET_DIR)
    p.add_argument("--base-model-dir", default=MODEL_DIR)
    p.add_argument("--out-dir", default="results/sln_dredge_iter")
    p.add_argument("--data-dir", default="data/dataset1")

    p.add_argument("--outer-iters", type=int, default=8)
    p.add_argument("--sln-epochs", type=int, default=5)
    p.add_argument("--batch-bins", type=int, default=32)
    p.add_argument("--window", type=int, default=30)

    p.add_argument("--lambda-corr", type=float, default=1.0)
    p.add_argument("--lambda-ent", type=float, default=0.1)
    p.add_argument("--lambda-teth", type=float, default=0.01)
    p.add_argument("--min-spikes-per-bin", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)

    p.add_argument("--dredge-rigid", action="store_true",
                   help="Force rigid (single y-shift per bin) instead of nonrigid")
    p.add_argument("--no-x-drift", action="store_true",
                   help="Disable separate x-drift estimation")
    p.add_argument("--win-step-um", type=float, default=400.0)
    p.add_argument("--win-scale-um", type=float, default=450.0)

    p.add_argument("--frame-range", default=None,
                   help="Subset bins for smoke test, e.g. 0:100")
    p.add_argument("--monopolar-z", default=None,
                   help="Path to per-spike monopolar z (N_full,). If provided, "
                        "every persisted GL array uses this z (gathered via "
                        "spike_indices) instead of the SLN's z prediction.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-eval-blocks", type=int, default=20)
    p.add_argument("--device", default=None)
    return p.parse_args()


def choose_device(arg=None):
    if arg is not None:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()
    set_torch_determinism(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = choose_device(args.device)
    print(f"Device: {device}")

    # ---- Load dataset
    print("Loading dataset...")
    ds = load_dataset(args.dataset_dir)
    fs = float(ds["manifest"]["fs"])
    n_frames_total = int(ds["frame_ids"].max()) + 1
    ds = sort_by_frame(ds, n_frames_total)
    print(f"  {ds['waveforms'].shape[0]} spikes, {n_frames_total} bins, fs={fs}")

    # Optional frame-range subset (used for smoke test)
    if args.frame_range:
        lo, hi = (int(x) for x in args.frame_range.split(":"))
        mask = (ds["frame_ids"] >= lo) & (ds["frame_ids"] < hi)
        for k in ("waveforms", "anchors", "frame_ids", "spike_indices",
                  "split", "targets_global"):
            ds[k] = np.asarray(ds[k])[mask]
        ds["frame_ids"] = ds["frame_ids"] - lo
        n_frames_total = hi - lo
        ds["frame_offsets"] = build_frame_offsets(ds["frame_ids"], n_frames_total)
        print(f"  subset to bins [{lo}, {hi}): {ds['waveforms'].shape[0]} spikes")

    # ---- Train/val/test bin splits
    split_ranges = ds["manifest"]["split_frame_ranges"]
    if args.frame_range:
        # smoke test — just use all bins as train, last 10% as val
        split_ranges = {
            "train": [0, int(0.9 * n_frames_total)],
            "val": [int(0.9 * n_frames_total), n_frames_total],
            "test": [int(0.9 * n_frames_total), n_frames_total],
        }
    else:
        # Convert dict-of-list to dict-of-tuple
        split_ranges = {k: tuple(v) for k, v in split_ranges.items()}
    print(f"  splits: {split_ranges}")

    # ---- Load original spike metadata for DREDGE
    print("Loading original spike metadata for DREDGE...")
    spike_times_full = np.load("results/spike_times.npy")
    spike_channels_full = np.load("results/spike_channels.npy")
    spike_amplitudes_full = np.load("results/spike_amplitudes.npy")
    # Map dataset rows back to original indices
    si = ds["spike_indices"]
    spike_times = spike_times_full[si]
    spike_channels = spike_channels_full[si]
    spike_amplitudes = spike_amplitudes_full[si]

    # ---- Load preprocessed recording for DREDGE
    print("Loading recording...")
    from window_video_common import load_preprocessed_recording
    recording = load_preprocessed_recording(args.data_dir)
    if args.frame_range:
        # only use the slice that matches
        lo_s, hi_s = lo, hi
        recording = recording.frame_slice(int(lo_s * fs), int(hi_s * fs))
        spike_times = spike_times - int(lo_s * fs)
        # mask to slice
        keep = (spike_times >= 0) & (spike_times < int((hi_s - lo_s) * fs))
        if not keep.all():
            for k in ("waveforms", "anchors", "frame_ids", "split", "targets_global"):
                ds[k] = ds[k][keep]
            spike_times = spike_times[keep]
            spike_channels = spike_channels[keep]
            spike_amplitudes = spike_amplitudes[keep]
            ds["frame_offsets"] = build_frame_offsets(ds["frame_ids"], n_frames_total)

    # ---- Load model
    print("Loading pretrained SLN...")
    model, waveform_scale = load_pretrained_sln(
        os.path.join(args.base_model_dir, "best_model.pt"), device)
    print(f"  waveform_scale={waveform_scale}")

    # ---- Optionally load monopolar z to override SLN's z output
    monopolar_z_subset = None
    if args.monopolar_z is not None:
        mp_z_full = np.load(args.monopolar_z).astype(np.float32)
        monopolar_z_subset = mp_z_full[ds["spike_indices"]]
        print(f"  loaded monopolar z from {args.monopolar_z}  "
              f"(N_full={len(mp_z_full):,}, N_subset={len(monopolar_z_subset):,})")

    # ---- Compute baseline GL
    print("Computing baseline GL...")
    baseline_GL = predict_all(model, ds["waveforms"], ds["anchors"], waveform_scale, device)
    if monopolar_z_subset is not None:
        baseline_GL[:, 2] = monopolar_z_subset
    np.save(os.path.join(args.out_dir, "baseline_GL.npy"), baseline_GL)
    print(f"  baseline GL median (x,y) = ({np.median(baseline_GL[:, 0]):.1f}, "
          f"{np.median(baseline_GL[:, 1]):.1f})")

    # ---- Histogram + samplers
    xc, yc = default_2d_grid()
    soft_hist = SoftHistogram2D(xc, yc, sigma_x=4.0, sigma_y=4.0).to(device)
    train_loader = BinBlockSampler(
        ds["frame_offsets"], split_ranges["train"], args.batch_bins,
        args.min_spikes_per_bin, seed=args.seed)
    val_loader = BinBlockSampler(
        ds["frame_offsets"], split_ranges["val"], args.batch_bins,
        args.min_spikes_per_bin, seed=args.seed + 1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    # ---- History
    history_path = os.path.join(args.out_dir, "history.csv")
    fields = ["outer_iter", "epoch", "phase", "rho_mean", "ent_mean",
              "teth_mean", "n_blocks", "elapsed_s",
              "Δy_p95_um", "Δx_p95_um"]
    with open(history_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    def log_row(row):
        with open(history_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)

    best = {"rho_val": -math.inf, "ent_val": 0.0, "outer_iter": -1}

    GL_current = baseline_GL.copy()

    for outer in range(1, args.outer_iters + 1):
        t0 = time.time()
        # ---------------- DREDGE step ---------------------------------------
        print(f"\n[outer {outer}] DREDGE step...")
        try:
            motion = dredge_step(
                GL_current, spike_times, spike_channels, spike_amplitudes,
                recording,
                rigid=args.dredge_rigid,
                win_step_um=args.win_step_um, win_scale_um=args.win_scale_um,
                bin_s=1.0,
                estimate_x_drift=not args.no_x_drift,
            )
        except Exception as e:
            print(f"DREDGE failed: {e}; continuing with zero motion")
            T_bins = n_frames_total
            motion = {
                "Δy_func": lambda t, y: np.zeros_like(np.asarray(y, dtype=np.float64)),
                "Δx_per_bin": np.zeros(T_bins),
                "t_bin_centers": np.arange(T_bins) + 0.5,
                "y_anchors": np.array([0.0, 3840.0]),
                "t_anchors": np.array([0.0, T_bins * 1.0]),
                "disp": np.zeros((T_bins, 1)),
            }
        Δy_p95 = float(np.percentile(np.abs(motion["disp"]), 95)) if motion["disp"].size else 0.0
        Δx_p95 = float(np.percentile(np.abs(motion["Δx_per_bin"]), 95))
        print(f"  Δy_p95={Δy_p95:.1f}µm  Δx_p95={Δx_p95:.1f}µm")

        # Pre-evaluate before any SLN updates
        e0 = evaluate(model, val_loader, ds, motion, soft_hist, waveform_scale, fs,
                      args.window, device, max_blocks=args.max_eval_blocks)
        print(f"  pre-train val rho={e0['rho_mean']:.4f}  ent={e0['ent_mean']:.4f}")
        log_row({"outer_iter": outer, "epoch": 0, "phase": "val_predredge_done",
                 "rho_mean": e0["rho_mean"], "ent_mean": e0["ent_mean"],
                 "teth_mean": float("nan"), "n_blocks": e0["n_blocks"],
                 "elapsed_s": time.time() - t0,
                 "Δy_p95_um": Δy_p95, "Δx_p95_um": Δx_p95})

        # ---------------- SLN step ------------------------------------------
        for ep in range(1, args.sln_epochs + 1):
            model.train()
            ep_rho, ep_ent, ep_teth, n = 0.0, 0.0, 0.0, 0
            for start in train_loader.epoch():
                spike_idx, bin_id_local, valid = train_loader.gather(start)
                if valid.sum() < 4:
                    continue
                W_np = ds["waveforms"][spike_idx].astype(np.float32)
                anchors_np = ds["anchors"][spike_idx].astype(np.float32)
                baseline_np = baseline_GL[spike_idx].astype(np.float32)
                t_s = ds["frame_ids"][spike_idx].astype(np.float64) + 0.5
                t_idx = ds["frame_ids"][spike_idx].astype(np.int64)

                W = torch.from_numpy(W_np).to(device)
                anchors = torch.from_numpy(anchors_np).to(device)
                baseline = torch.from_numpy(baseline_np).to(device)
                bin_id_t = torch.from_numpy(bin_id_local).to(device)
                valid_t = torch.from_numpy(valid).to(device)

                out = sln_loss(
                    model, W, anchors, baseline, bin_id_t, train_loader.block_size,
                    motion, t_s, t_idx, soft_hist, waveform_scale,
                    args.lambda_corr, args.lambda_ent, args.lambda_teth,
                    args.window, valid_t,
                )
                optimizer.zero_grad(set_to_none=True)
                out["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                ep_rho += float(out["rho"])
                ep_ent += float(out["ent"])
                ep_teth += float(out["teth"])
                n += 1
            ep_rho = ep_rho / max(1, n)
            ep_ent = ep_ent / max(1, n)
            ep_teth = ep_teth / max(1, n)

            ev = evaluate(model, val_loader, ds, motion, soft_hist, waveform_scale, fs,
                          args.window, device, max_blocks=args.max_eval_blocks)
            print(f"  [iter {outer} ep {ep}] train rho={ep_rho:.4f} ent={ep_ent:.4f} teth={ep_teth:.2f} | "
                  f"val rho={ev['rho_mean']:.4f} ent={ev['ent_mean']:.4f}")
            log_row({"outer_iter": outer, "epoch": ep, "phase": "train",
                     "rho_mean": ep_rho, "ent_mean": ep_ent,
                     "teth_mean": ep_teth, "n_blocks": n,
                     "elapsed_s": time.time() - t0,
                     "Δy_p95_um": Δy_p95, "Δx_p95_um": Δx_p95})
            log_row({"outer_iter": outer, "epoch": ep, "phase": "val",
                     "rho_mean": ev["rho_mean"], "ent_mean": ev["ent_mean"],
                     "teth_mean": float("nan"), "n_blocks": ev["n_blocks"],
                     "elapsed_s": time.time() - t0,
                     "Δy_p95_um": Δy_p95, "Δx_p95_um": Δx_p95})

            if ev["rho_mean"] > best["rho_val"]:
                best.update({"rho_val": ev["rho_mean"], "ent_val": ev["ent_mean"],
                             "outer_iter": outer, "epoch": ep})
                torch.save({"model_state_dict": model.state_dict(),
                            "outer_iter": outer, "epoch": ep,
                            "val_metrics": ev,
                            "waveform_scale": waveform_scale},
                           os.path.join(args.out_dir, "sln_best.pt"))

        # End of outer iter — refresh GL_current and save checkpoint
        print(f"[outer {outer}] refreshing GL_current...")
        GL_current = predict_all(model, ds["waveforms"], ds["anchors"], waveform_scale, device)
        if monopolar_z_subset is not None:
            GL_current[:, 2] = monopolar_z_subset
        np.save(os.path.join(args.out_dir, f"GL_iter{outer:02d}.npy"), GL_current)
        torch.save({"model_state_dict": model.state_dict(),
                    "outer_iter": outer, "waveform_scale": waveform_scale},
                   os.path.join(args.out_dir, f"sln_iter{outer:02d}.pt"))

        # Save the latest motion for this iter
        np.savez(
            os.path.join(args.out_dir, f"motion_iter{outer:02d}.npz"),
            disp=motion["disp"],
            t_anchors=motion["t_anchors"],
            y_anchors=motion["y_anchors"],
            Δx_per_bin=motion["Δx_per_bin"],
        )

    with open(os.path.join(args.out_dir, "best_metrics.json"), "w") as f:
        json.dump(best, f, indent=2)
    print(f"\nDone. Best at outer={best['outer_iter']} ep={best['epoch']} "
          f"rho_val={best['rho_val']:.4f} ent_val={best['ent_val']:.4f}")


if __name__ == "__main__":
    main()
