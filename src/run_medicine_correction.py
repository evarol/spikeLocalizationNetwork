"""Apply MEDICINE (medicine-neuro) drift correction to a set of localizations.

MEDICINE is a drop-in alternative to DREDge: it estimates depth-dependent
(non-rigid) motion from (peak_time, peak_depth, peak_amplitude) and we correct
the depth axis (y) with it, leaving x, z unchanged — same convention as our
DREDge pipeline.

Input: an (N,3) uncorrected localization GL_pre (x, y, z) + spike times + amps.
Output (out_dir):
    motion.npy, time_bins.npy, depth_bins.npy   (MEDICINE native)
    GL_pre_dredge.npy   (copy of the uncorrected input, for viz)
    GL_post_dredge.npy, x.npy, y.npy, z.npy      (y drift-corrected)
    + MEDICINE's own diagnostic figures (raster before/after, motion).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gl_pre", required=True, type=Path, help="(N,3) uncorrected localization")
    p.add_argument("--spike_times", required=True, type=Path)
    p.add_argument("--amplitudes", required=True, type=Path)
    p.add_argument("--fs", required=True, type=Path)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--num_depth_bins", type=int, default=2)
    p.add_argument("--training_steps", type=int, default=10000)
    args = p.parse_args()

    from medicine import run_medicine
    from medicine.plotting import _correct_motion_on_peaks

    GL = np.load(args.gl_pre).astype(np.float64)
    st = np.load(args.spike_times)
    amp = np.load(args.amplitudes).astype(np.float64)
    fs = float(np.load(args.fs).reshape(-1)[0])
    peak_times = st.astype(np.float64) / fs
    peak_depths = GL[:, 1].copy()                       # y = depth axis
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"MEDICINE: N={len(GL):,}  y∈[{peak_depths.min():.0f},{peak_depths.max():.0f}]µm  "
          f"T={peak_times.max():.0f}s  depth_bins={args.num_depth_bins}  steps={args.training_steps}")

    trainer, time_bins, depth_bins, motion = run_medicine(
        peak_times=peak_times, peak_depths=peak_depths, peak_amplitudes=np.abs(amp),
        output_dir=args.out_dir, num_depth_bins=args.num_depth_bins,
        training_steps=args.training_steps, plot_figures=True,
    )
    motion = np.asarray(motion); time_bins = np.asarray(time_bins); depth_bins = np.asarray(depth_bins)

    y_corr = _correct_motion_on_peaks(peak_times, peak_depths, motion, time_bins, depth_bins)
    GL_post = GL.copy(); GL_post[:, 1] = y_corr

    np.save(args.out_dir / "GL_pre_dredge.npy", GL.astype(np.float32))
    np.save(args.out_dir / "GL_post_dredge.npy", GL_post.astype(np.float32))
    np.save(args.out_dir / "x.npy", GL_post[:, 0].astype(np.float32))
    np.save(args.out_dir / "y.npy", GL_post[:, 1].astype(np.float32))
    np.save(args.out_dir / "z.npy", GL_post[:, 2].astype(np.float32))

    dy = peak_depths - y_corr
    print(f"  motion shape={motion.shape}  |Δy| p50={np.percentile(np.abs(dy),50):.2f} "
          f"p95={np.percentile(np.abs(dy),95):.2f} max={np.abs(dy).max():.2f}µm")
    print(f"  wrote {args.out_dir}/")


if __name__ == "__main__":
    main()
