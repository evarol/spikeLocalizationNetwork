"""Train an SLN (CNN or Transformer) for N inner epochs given a pre-computed
DREDge motion field, with the motion held fixed (no further DREDge passes).

Loss: −λ_ρ · ρ̄ − λ_H · H̄ + λ_teth · L_teth on the canonical 4 μm σ=4 soft
histograms. Same as train_sln_dredge_iterative.py's Phase B but architecture-
agnostic and MPS-friendly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from relative_xyz_common import (
    RelativeXYZNet, set_torch_determinism,
)
from transformer_sln import TransformerSLN
from soft_histogram import SoftHistogram2D, default_2d_grid
from histogram_losses import pairwise_ncc, spatial_entropy, tether_loss

# Re-use the BinBlockSampler + dataset loading from iterative trainer (no
# DREDge there at import time, so the import is safe).
from train_sln_dredge_iterative import (
    BinBlockSampler, load_dataset, sort_by_frame, choose_device,
)

DATASET_DIR_DEFAULT = "results/relative_xyz_dataset_500k"


def load_sln(arch: str, ckpt_path: Path, device: torch.device):
    """Returns (model, scale, forward_fn) where forward_fn(W, anchors_xy) → (B, 3)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    scale = float(ckpt["waveform_scale"])
    if arch == "cnn":
        model = RelativeXYZNet()
        model.load_state_dict(state)
        model = model.to(device)
        def forward_fn(W, anchors_xy):
            return model(W / scale)
    elif arch == "transformer":
        cen_mean = tuple(ckpt["centroid_mean"]); cen_std = tuple(ckpt["centroid_std"])
        model = TransformerSLN(centroid_mean=cen_mean, centroid_std=cen_std)
        model.load_state_dict(state)
        model = model.to(device)
        def forward_fn(W, anchors_xy):
            return model(W / scale, anchors_xy)
    else:
        raise ValueError(f"unknown arch: {arch}")
    return model, scale, forward_fn


def load_motion(motion_path: Path) -> dict:
    """Load motion .npz and build a Δy_func + Δx_per_bin compatible with the loss."""
    npz = dict(np.load(motion_path))
    disp = np.asarray(npz["disp"], dtype=np.float64)
    t_anchors = np.asarray(npz["t_anchors"], dtype=np.float64)
    y_anchors = np.asarray(npz["y_anchors"], dtype=np.float64)
    T = disp.shape[0]
    n_y = len(y_anchors)

    def Δy_func(t_s, y_s):
        t_s = np.asarray(t_s, dtype=np.float64)
        y_s = np.asarray(y_s, dtype=np.float64)
        ti = np.clip(np.searchsorted(t_anchors, t_s) - 1, 0, len(t_anchors) - 2)
        if n_y == 1:
            return disp[ti, 0]
        yi = np.clip(np.searchsorted(y_anchors, y_s) - 1, 0, n_y - 2)
        ta0 = t_anchors[ti]; ta1 = t_anchors[ti + 1]
        ya0 = y_anchors[yi]; ya1 = y_anchors[yi + 1]
        wt = np.clip((t_s - ta0) / np.maximum(ta1 - ta0, 1e-9), 0, 1)
        wy = np.clip((y_s - ya0) / np.maximum(ya1 - ya0, 1e-9), 0, 1)
        d00 = disp[ti, yi]; d10 = disp[ti + 1, yi]
        d01 = disp[ti, yi + 1]; d11 = disp[ti + 1, yi + 1]
        return ((1 - wt) * (1 - wy) * d00 + wt * (1 - wy) * d10
                + (1 - wt) * wy * d01 + wt * wy * d11)

    Δx_per_bin = np.asarray(npz.get("Δx_per_bin",
                                   npz.get("dx_per_bin",
                                           np.zeros(T))), dtype=np.float64)
    if Δx_per_bin.shape[0] != T:
        Δx_per_bin = np.zeros(T, dtype=np.float64)

    return {
        "Δy_func": Δy_func,
        "Δx_per_bin": Δx_per_bin,
        "disp": disp,
        "t_anchors": t_anchors,
        "y_anchors": y_anchors,
    }


@torch.no_grad()
def predict_all(forward_fn, waveforms, anchors: np.ndarray,
                device: torch.device, batch: int = 2048) -> np.ndarray:
    """waveforms may be a numpy array/mmap OR a pre-loaded torch tensor on `device`."""
    n = waveforms.shape[0]
    out = np.zeros((n, 3), dtype=np.float32)
    is_tensor = isinstance(waveforms, torch.Tensor)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        if is_tensor:
            W = waveforms[s:e]
        else:
            W = torch.from_numpy(np.asarray(waveforms[s:e]).astype(np.float32)).to(device)
        C = torch.from_numpy(anchors[s:e, :2].astype(np.float32)).to(device)
        rel = forward_fn(W, C).cpu().numpy()
        out[s:e] = rel + anchors[s:e]
    return out


def fetch_W(ds: dict, spike_idx_np: np.ndarray, device: torch.device) -> torch.Tensor:
    """Fetch a batch of waveforms onto `device`. If ds['waveforms_t'] is set (MPS-
    resident pre-loaded tensor), use a single index_select to avoid mmap/disk I/O.
    Otherwise fall back to the numpy/mmap path."""
    wf_t = ds.get("waveforms_t")
    if wf_t is not None:
        idx_t = torch.from_numpy(spike_idx_np).to(device)
        return wf_t.index_select(0, idx_t)
    W_np = np.asarray(ds["waveforms"][spike_idx_np]).astype(np.float32)
    return torch.from_numpy(W_np).to(device)


def sln_loss(forward_fn, batch_W, batch_anchors, batch_baseline_GL,
             bin_id_local, n_bins_in_block,
             motion, t_s_block, t_idx_block,
             soft_hist: SoftHistogram2D,
             lambda_corr: float, lambda_ent: float, lambda_teth: float,
             window: int, valid_mask: torch.Tensor) -> dict:
    rel = forward_fn(batch_W, batch_anchors[:, :2])         # (M, 3)
    GL_xy = rel[:, :2] + batch_anchors[:, :2]               # (M, 2)
    Δy = torch.as_tensor(motion["Δy_func"](t_s_block, GL_xy[:, 1].detach().cpu().numpy()),
                         dtype=GL_xy.dtype, device=GL_xy.device)
    Δx = torch.as_tensor(motion["Δx_per_bin"][t_idx_block],
                         dtype=GL_xy.dtype, device=GL_xy.device)
    GL_corr = torch.stack([GL_xy[:, 0] - Δx, GL_xy[:, 1] - Δy], dim=1)
    H = soft_hist(GL_corr, bin_id_local, n_bins_in_block)
    rho = pairwise_ncc(H, valid_mask, window=window)
    ent = spatial_entropy(H)[valid_mask].mean() if valid_mask.any() else torch.tensor(0.0, device=H.device)
    teth = tether_loss(GL_xy, batch_baseline_GL[:, :2])
    L = -lambda_corr * rho - lambda_ent * ent + lambda_teth * teth
    return {"loss": L, "rho": rho.detach(), "ent": ent.detach(), "teth": teth.detach()}


def evaluate(forward_fn, loader: BinBlockSampler, ds: dict, motion: dict,
             soft_hist: SoftHistogram2D, window: int, device,
             max_blocks: int) -> dict:
    rhos, ents, n = [], [], 0
    with torch.no_grad():
        for start in loader.epoch():
            spike_idx, bin_id_local, valid = loader.gather(start)
            if not valid.any():
                continue
            W = fetch_W(ds, spike_idx, device)
            anchors = torch.from_numpy(ds["anchors"][spike_idx].astype(np.float32)).to(device)
            rel = forward_fn(W, anchors[:, :2])
            GL_xy = rel[:, :2] + anchors[:, :2]
            t_s = ds["frame_ids"][spike_idx].astype(np.float64) + 0.5
            t_idx = ds["frame_ids"][spike_idx].astype(np.int64)
            Δy = torch.as_tensor(motion["Δy_func"](t_s, GL_xy[:, 1].detach().cpu().numpy()),
                                 dtype=GL_xy.dtype, device=device)
            Δx = torch.as_tensor(motion["Δx_per_bin"][t_idx],
                                 dtype=GL_xy.dtype, device=device)
            GL_corr = torch.stack([GL_xy[:, 0] - Δx, GL_xy[:, 1] - Δy], dim=1)
            bin_id_t = torch.from_numpy(bin_id_local).to(device)
            valid_t = torch.from_numpy(valid).to(device)
            H = soft_hist(GL_corr, bin_id_t, loader.block_size)
            rhos.append(float(pairwise_ncc(H, valid_t, window=window)))
            ents.append(float(spatial_entropy(H)[valid_t].mean()))
            n += 1
            if n >= max_blocks:
                break
    return {
        "rho_mean": float(np.mean(rhos)) if rhos else float("nan"),
        "ent_mean": float(np.mean(ents)) if ents else float("nan"),
        "n_blocks": n,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["cnn", "transformer"], required=True)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--motion", required=True, type=Path)
    p.add_argument("--dataset_dir", default=DATASET_DIR_DEFAULT, type=Path)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--sln_epochs", type=int, default=20)
    p.add_argument("--batch_bins", type=int, default=32)
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--lambda_corr", type=float, default=1.0)
    p.add_argument("--lambda_ent", type=float, default=0.1)
    p.add_argument("--lambda_teth", type=float, default=0.01)
    p.add_argument("--min_spikes_per_bin", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=5.0,
                   help="Max grad L2 norm for clip_grad_norm_. Lower (e.g. 1.0) helps "
                        "stabilize fine-tuning resumes from converged ckpts.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_eval_blocks", type=int, default=20)
    p.add_argument("--device", default=None)
    p.add_argument("--mps_preload", action="store_true", default=None,
                   help="Pre-load full waveforms tensor to MPS to avoid mmap thrash. "
                        "Default: enabled when device=mps, disabled otherwise.")
    p.add_argument("--no_mps_preload", action="store_false", dest="mps_preload",
                   help="Disable MPS pre-load (use mmap path even on MPS).")
    p.add_argument("--mps_empty_cache_every", type=int, default=50,
                   help="On MPS, call torch.mps.empty_cache() every N training iters "
                        "to mitigate allocator fragmentation. 0 = disable.")
    p.add_argument("--save_every_epoch", action="store_true", default=False,
                   help="Save per-epoch snapshots (sln_ep01.pt, sln_ep02.pt, ...) in "
                        "addition to sln_best.pt.")
    p.add_argument("--iter_log_every", type=int, default=0,
                   help="If >0, print per-iter throughput every N iters (for live "
                        "monitoring of long runs).")
    return p.parse_args()


def main():
    args = parse_args()
    set_torch_determinism(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "config.json", "w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v
                   for k, v in vars(args).items()}, f, indent=2)

    device = choose_device(args.device)
    print(f"device = {device}")

    print(f"Loading dataset: {args.dataset_dir}")
    ds = load_dataset(str(args.dataset_dir))
    n_frames_total = int(ds["frame_ids"].max()) + 1
    ds = sort_by_frame(ds, n_frames_total)
    print(f"  {ds['waveforms'].shape[0]:,} spikes, {n_frames_total} bins")

    # MPS pre-load: keep the entire waveforms tensor on GPU so per-iter fetch is a
    # single index_select instead of mmap + CPU→MPS transfer (which thrashes the
    # OS page cache on long runs with multi-GB datasets).
    mps_preload = args.mps_preload
    if mps_preload is None:
        mps_preload = (device.type == "mps")
    if mps_preload:
        t0 = time.time()
        wf_np = np.asarray(ds["waveforms"])
        ds["waveforms_t"] = torch.from_numpy(wf_np.astype(np.float32, copy=False)).to(device)
        if device.type == "mps":
            torch.mps.synchronize()
        nbytes = ds["waveforms_t"].element_size() * ds["waveforms_t"].nelement()
        print(f"  pre-loaded {nbytes/1e9:.2f} GB waveforms to {device} in {time.time()-t0:.1f}s")
        # We can drop the host-side reference now (the tensor copy is on device)
        # but keep `ds['waveforms']` for shape introspection.

    # Split ranges (frame-based, same as iterative trainer)
    sr = ds["manifest"]["split_frame_ranges"]
    split_ranges = {k: (int(v[0]), int(v[1]) + 1) for k, v in sr.items()}

    print(f"Loading {args.arch} ckpt: {args.ckpt}")
    model, scale, forward_fn = load_sln(args.arch, args.ckpt, device)
    print(f"  waveform_scale = {scale:.4f}, params = {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    print(f"Loading motion: {args.motion}")
    motion = load_motion(args.motion)
    print(f"  disp shape = {motion['disp'].shape}, "
          f"Δy_p95 = {np.percentile(np.abs(motion['disp']), 95):.1f}μm")

    print("Computing baseline GL...")
    wf_for_pred = ds.get("waveforms_t", ds["waveforms"])
    baseline_GL = predict_all(forward_fn, wf_for_pred, ds["anchors"], device)
    np.save(args.out_dir / "baseline_GL.npy", baseline_GL)
    print(f"  baseline GL median (x, y) = ({np.median(baseline_GL[:, 0]):.1f}, "
          f"{np.median(baseline_GL[:, 1]):.1f})")

    xc, yc = default_2d_grid()
    soft_hist = SoftHistogram2D(xc, yc, sigma_x=4.0, sigma_y=4.0).to(device)
    train_loader = BinBlockSampler(ds["frame_offsets"], split_ranges["train"],
                                    args.batch_bins, args.min_spikes_per_bin,
                                    seed=args.seed)
    val_loader = BinBlockSampler(ds["frame_offsets"], split_ranges["val"],
                                  args.batch_bins, args.min_spikes_per_bin,
                                  seed=args.seed + 1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)

    history_path = args.out_dir / "history.csv"
    fields = ["epoch", "phase", "rho_mean", "ent_mean", "teth_mean", "n_blocks", "elapsed_s"]
    with open(history_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    def log_row(row):
        with open(history_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)

    t_start = time.time()
    e0 = evaluate(forward_fn, val_loader, ds, motion, soft_hist,
                  args.window, device, max_blocks=args.max_eval_blocks)
    print(f"Pre-train val rho={e0['rho_mean']:.4f}  ent={e0['ent_mean']:.4f}")
    log_row({"epoch": 0, "phase": "val_pretrain",
             "rho_mean": e0["rho_mean"], "ent_mean": e0["ent_mean"],
             "teth_mean": float("nan"), "n_blocks": e0["n_blocks"],
             "elapsed_s": time.time() - t_start})

    best = {"rho_val": -math.inf, "epoch": -1}

    for ep in range(1, args.sln_epochs + 1):
        model.train()
        ep_rho, ep_ent, ep_teth, n = 0.0, 0.0, 0.0, 0
        iter_t0 = time.time()
        for start in train_loader.epoch():
            spike_idx, bin_id_local, valid = train_loader.gather(start)
            if valid.sum() < 4:
                continue
            anchors_np = ds["anchors"][spike_idx].astype(np.float32)
            baseline_np = baseline_GL[spike_idx].astype(np.float32)
            t_s = ds["frame_ids"][spike_idx].astype(np.float64) + 0.5
            t_idx = ds["frame_ids"][spike_idx].astype(np.int64)

            W = fetch_W(ds, spike_idx, device)
            anchors = torch.from_numpy(anchors_np).to(device)
            baseline = torch.from_numpy(baseline_np).to(device)
            bin_id_t = torch.from_numpy(bin_id_local).to(device)
            valid_t = torch.from_numpy(valid).to(device)

            out = sln_loss(forward_fn, W, anchors, baseline,
                           bin_id_t, train_loader.block_size,
                           motion, t_s, t_idx, soft_hist,
                           args.lambda_corr, args.lambda_ent, args.lambda_teth,
                           args.window, valid_t)
            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            ep_rho += float(out["rho"]); ep_ent += float(out["ent"]); ep_teth += float(out["teth"])
            n += 1

            # Periodic MPS allocator reset — mitigates the ~30× steady-state
            # slowdown we measured on TR-all (cf. POSTDREDGE_COMPARISON.md).
            if (device.type == "mps" and args.mps_empty_cache_every > 0
                    and n % args.mps_empty_cache_every == 0):
                torch.mps.empty_cache()

            if args.iter_log_every > 0 and n % args.iter_log_every == 0:
                dt = time.time() - iter_t0
                print(f"    [ep {ep:02d} iter {n:4d}]  "
                      f"last {args.iter_log_every} iters: {dt:.1f}s "
                      f"({dt/args.iter_log_every*1000:.0f} ms/iter), "
                      f"rho={float(out['rho']):.4f}", flush=True)
                iter_t0 = time.time()

        ep_rho /= max(1, n); ep_ent /= max(1, n); ep_teth /= max(1, n)
        ev = evaluate(forward_fn, val_loader, ds, motion, soft_hist,
                      args.window, device, max_blocks=args.max_eval_blocks)
        elapsed = time.time() - t_start
        print(f"[ep {ep:02d}]  train rho={ep_rho:.4f}  ent={ep_ent:.4f}  teth={ep_teth:.2f}  | "
              f"val rho={ev['rho_mean']:.4f}  ent={ev['ent_mean']:.4f}   "
              f"({elapsed:.0f}s elapsed)")
        log_row({"epoch": ep, "phase": "train",
                 "rho_mean": ep_rho, "ent_mean": ep_ent,
                 "teth_mean": ep_teth, "n_blocks": n, "elapsed_s": elapsed})
        log_row({"epoch": ep, "phase": "val",
                 "rho_mean": ev["rho_mean"], "ent_mean": ev["ent_mean"],
                 "teth_mean": float("nan"), "n_blocks": ev["n_blocks"],
                 "elapsed_s": elapsed})

        ckpt_save = {"model_state_dict": model.state_dict(),
                     "epoch": ep, "val_rho": ev["rho_mean"],
                     "waveform_scale": scale}
        if args.arch == "transformer":
            ckpt_save["centroid_mean"] = model.centroid_mean.detach().cpu().numpy().tolist()
            ckpt_save["centroid_std"]  = model.centroid_std.detach().cpu().numpy().tolist()
        if ev["rho_mean"] > best["rho_val"]:
            best.update({"rho_val": ev["rho_mean"], "epoch": ep})
            torch.save(ckpt_save, args.out_dir / "sln_best.pt")
        if args.save_every_epoch:
            torch.save(ckpt_save, args.out_dir / f"sln_ep{ep:02d}.pt")

    print(f"\nDone. Best val rho = {best['rho_val']:.4f} at epoch {best['epoch']}.")
    with open(args.out_dir / "best_metrics.json", "w") as f:
        json.dump(best, f, indent=2)


if __name__ == "__main__":
    main()
