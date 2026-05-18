"""Round-2 DREDge on the CNN-SLN all-spike ep20 model's pre-motion predictions.

Pipeline:
  MP → pretrain CNN → DREDge1 → 20 epochs of SLN training with DREDge1 frozen
       → ep20 CNN  → DREDge2 (this script) → apply motion2 → viz

Saves:
  results/dredge2_cnn_all_ep20/
    motion.npz          DREDge2 motion field (disp, t_anchors, y_anchors, Δx_per_bin)
    GL_pre_dredge.npy   CNN ep20 raw predictions (anchor + rel, no motion)
    GL_post_dredge.npy  GL_pre - motion2 applied
    x.npy y.npy z.npy   1-D arrays for the method-agnostic viz pipeline
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from dredge_wrapper import build_peaks, build_locs, run_dredge


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gl_pre", default="results/postdredge_cnn_all_ep20_all_spikes/GL_pre_dredge.npy",
                   type=Path, help="(N, 3) CNN ep20 raw predictions (anchor + rel, no motion)")
    p.add_argument("--data_dir", default="data/dataset1", type=Path)
    p.add_argument("--channels", default="results/spike_channels.npy", type=Path)
    p.add_argument("--amplitudes", default="results/spike_amplitudes.npy", type=Path)
    p.add_argument("--spike_times", default="results/spike_times.npy", type=Path)
    p.add_argument("--out_dir", default="results/dredge2_cnn_all_ep20", type=Path)
    # SI-canonical DREDge config (matches DREDge1 and POSTDREDGE_COMPARISON.md)
    p.add_argument("--rigid", action="store_true", default=False)
    p.add_argument("--win_step_um", type=float, default=400.0)
    p.add_argument("--win_scale_um", type=float, default=400.0)
    p.add_argument("--no_x_drift", action="store_true", default=False)
    p.add_argument("--bin_s", type=float, default=1.0)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    GL_pre = np.load(args.gl_pre).astype(np.float32)
    spike_times = np.load(args.spike_times)
    spike_channels = np.load(args.channels)
    spike_amplitudes = np.load(args.amplitudes)
    assert len(GL_pre) == len(spike_times) == len(spike_channels) == len(spike_amplitudes), \
        "length mismatch"
    print(f"N = {len(GL_pre):,} spikes")
    print(f"  GL_pre y∈[{GL_pre[:,1].min():.1f}, {GL_pre[:,1].max():.1f}]μm")

    print("Loading recording (needed for DREDge: fs + total_duration)...")
    from window_video_common import load_preprocessed_recording
    recording = load_preprocessed_recording(str(args.data_dir))
    print(f"  fs = {recording.get_sampling_frequency()} Hz, "
          f"duration = {recording.get_total_duration():.1f} s")

    print(f"\nRunning DREDge2 (rigid={args.rigid}, win_step={args.win_step_um}, "
          f"win_scale={args.win_scale_um}, x_drift={'on' if not args.no_x_drift else 'off'})...")
    peaks = build_peaks(spike_times, spike_channels, spike_amplitudes)
    locs = build_locs(GL_pre)
    motion = run_dredge(
        recording, peaks, locs,
        rigid=args.rigid, win_step_um=args.win_step_um, win_scale_um=args.win_scale_um,
        bin_s=args.bin_s, estimate_x_drift=not args.no_x_drift,
        verbose=True,
    )
    Δy_p95 = float(np.percentile(np.abs(motion["disp"]), 95))
    Δx_p95 = float(np.percentile(np.abs(motion["Δx_per_bin"]), 95))
    print(f"  Δy_p95={Δy_p95:.2f}μm  Δx_p95={Δx_p95:.2f}μm")
    print(f"  disp shape={motion['disp'].shape}, y_anchors={len(motion['y_anchors'])}")

    # Save motion as the canonical .npz format
    np.savez(args.out_dir / "motion.npz",
             disp=motion["disp"], t_anchors=motion["t_anchors"],
             y_anchors=motion["y_anchors"], **({"Δx_per_bin": motion["Δx_per_bin"]} if not args.no_x_drift else {}))
    print(f"  saved {args.out_dir / 'motion.npz'}")

    # Apply motion to GL_pre → GL_post_dredge2
    fs = float(recording.get_sampling_frequency())
    t_s = spike_times.astype(np.float64) / fs
    t_idx = np.clip(t_s.astype(np.int64), 0, motion["disp"].shape[0] - 1)
    GL_post = GL_pre.astype(np.float64, copy=True)
    Δy = motion["Δy_func"](t_s, GL_post[:, 1])
    GL_post[:, 1] -= Δy
    GL_post[:, 0] -= motion["Δx_per_bin"][t_idx]
    print(f"  applied motion: mean |Δy|={np.mean(np.abs(Δy)):.2f}μm, "
          f"max |Δy|={np.max(np.abs(Δy)):.1f}μm")

    np.save(args.out_dir / "GL_pre_dredge.npy", GL_pre.astype(np.float32))
    np.save(args.out_dir / "GL_post_dredge.npy", GL_post.astype(np.float32))
    np.save(args.out_dir / "x.npy", GL_post[:, 0].astype(np.float32))
    np.save(args.out_dir / "y.npy", GL_post[:, 1].astype(np.float32))
    np.save(args.out_dir / "z.npy", GL_post[:, 2].astype(np.float32))
    print(f"  wrote GL_pre_dredge, GL_post_dredge, x/y/z to {args.out_dir}")
    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
