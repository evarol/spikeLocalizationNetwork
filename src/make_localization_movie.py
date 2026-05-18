"""Generate a localization-movie (method-agnostic).

Pass per-method paths to drive the movie:
  --x_path / --y_path / --z_path  : per-spike localizations for the method
  --motion_path                    : DREDge motion .npz (or --zero_motion for raw)
  --corr_matrix_path               : T×T Pearson matrix (from make_xy_pairwise_correlation)
  --entropy_path                   : T-length entropy trace (from make_spatial_entropy)
  --label / --color                : display name / accent color

Bottom timeseries panels are precomputed:
  motion       — from --motion_path  (or zeros if --zero_motion)
  mean ρ       — row-mean of --corr_matrix_path (off-diagonal)
  entropy      — --entropy_path
"""

from __future__ import annotations

import argparse
import os
import subprocess
from multiprocessing import Pool, cpu_count
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from localization_movie import (
    LOG_VMAX, LOG_VMIN,
    MovieFrameData, prepare_frame_index, render_frame,
)


_WORKER_DATA: MovieFrameData | None = None


def _init_worker(data: MovieFrameData):
    global _WORKER_DATA
    _WORKER_DATA = data


def _worker_render(args):
    frame_idx, frames_dir = args
    fig = render_frame(frame_idx, _WORKER_DATA)
    fig.savefig(os.path.join(frames_dir, f"frame_{frame_idx:05d}.png"),
                dpi=100, facecolor="black")
    plt.close(fig)


def load_motion_trace(motion_path: Path | None, T: int) -> np.ndarray:
    """Load a rigid Δy(t) trace from a DREDge motion .npz, or zeros if None."""
    if motion_path is None:
        return np.zeros(T, dtype=np.float32)
    npz = dict(np.load(motion_path))
    disp = npz["disp"]                     # (T, n_anchors)
    if disp.shape[0] != T:
        raise ValueError(f"motion T={disp.shape[0]} vs expected {T}")
    if disp.shape[1] == 1:
        return disp[:, 0].astype(np.float32)
    return disp.mean(axis=1).astype(np.float32)


def load_movie_data(args) -> MovieFrameData:
    spike_times = np.load(args.t_path)
    alpha = np.load(args.alpha_path).astype(np.float32)
    x = np.load(args.x_path).astype(np.float32)
    y = np.load(args.y_path).astype(np.float32)
    z = np.load(args.z_path).astype(np.float32)
    GL = np.stack([x, y, z], axis=1)
    assert len(GL) == len(spike_times) == len(alpha), "length mismatch"

    ch_locs = np.load(args.ch_path).astype(np.float32)
    ch_xyz = np.zeros((len(ch_locs), 3), dtype=np.float32)
    ch_xyz[:, 0] = ch_locs[:, 0]
    ch_xyz[:, 1] = ch_locs[:, 1]

    fs = float(np.load(args.fs_path)[0])
    t_s = spike_times.astype(np.float64) / fs

    # Determine T from the precomputed correlation matrix.
    C = np.load(args.corr_matrix_path)
    T = C.shape[0]
    assert C.shape == (T, T), f"correlation matrix must be square: {C.shape}"
    mean_corr = (C.sum(axis=1) - np.diag(C)) / (T - 1)

    entropy = np.load(args.entropy_path)
    assert len(entropy) == T, f"entropy length {len(entropy)} ≠ T={T}"

    motion_path = None if args.zero_motion else args.motion_path
    motion_trace = load_motion_trace(motion_path, T)

    frame_ids = np.clip(t_s.astype(np.int32), 0, T - 1)
    log_alpha = np.log10(np.clip(alpha, 1.0, None))

    motion_label = (
        "Δy applied (μm) — zero (uncorrected)" if args.zero_motion
        else "Δy applied (μm)"
    )

    data = MovieFrameData(
        frame_ids=frame_ids,
        GL=GL,
        log_alpha=log_alpha,
        ch_xyz=ch_xyz,
        motion_trace=motion_trace,
        mean_corr_trace=mean_corr.astype(np.float32),
        entropy_trace=entropy.astype(np.float32),
        method_label=args.label,
        method_color=args.color,
        motion_label=motion_label,
        corr_label="mean Pearson ρ",
        entropy_label="entropy (nats)",
    )
    return prepare_frame_index(data)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--t_path", default="results/spike_times.npy", type=Path)
    p.add_argument("--x_path", default="results/tpca_monopolar_full/spike_locs_x.npy", type=Path)
    p.add_argument("--y_path", default="results/tpca_monopolar_full/spike_locs_y.npy", type=Path)
    p.add_argument("--z_path", default="results/tpca_monopolar_full/spike_locs_z.npy", type=Path)
    p.add_argument("--alpha_path", default="results/spike_locs_alpha.npy", type=Path)
    p.add_argument("--fs_path", default="results/fs.npy", type=Path)
    p.add_argument("--ch_path", default="results/channel_locations.npy", type=Path)
    p.add_argument("--motion_path", default="results/sln_dredge_iter_full/motion_iter01.npz", type=Path,
                   help="DREDge motion .npz. Use --zero_motion to ignore.")
    p.add_argument("--zero_motion", action="store_true",
                   help="Override motion to zeros (use for uncorrected methods like raw monopolar).")
    p.add_argument("--corr_matrix_path", default="figures/xy_pairwise_corr.npy", type=Path)
    p.add_argument("--entropy_path", default="figures/spatial_entropy.npy", type=Path)
    p.add_argument("--label", default="Raw Monopolar", help="method display label")
    p.add_argument("--color", default="#ff8a55", help="method accent color")
    p.add_argument("--frames_dir", default="figures/localization_movie_frames", type=Path)
    p.add_argument("--out", default="figures/localization_movie.mp4", type=Path)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--workers", type=int, default=min(8, cpu_count()))
    p.add_argument("--n_frames", type=int, default=0, help="0 = all frames")
    p.add_argument("--draft", action="store_true", help="render only frame 900 as preview")
    args = p.parse_args()

    print(f"loading data…  label={args.label!r}")
    data = load_movie_data(args)
    T = len(data.motion_trace)
    print(f"  T={T} frames, N={len(data.GL):,} spikes, ch={len(data.ch_xyz)}")
    print(f"  motion Δy ∈ [{data.motion_trace.min():.2f}, {data.motion_trace.max():.2f}] μm")
    print(f"  mean ρ ∈ [{data.mean_corr_trace.min():.3f}, {data.mean_corr_trace.max():.3f}]")
    print(f"  entropy H ∈ [{np.nanmin(data.entropy_trace):.3f}, {np.nanmax(data.entropy_trace):.3f}] nats")

    args.frames_dir.mkdir(parents=True, exist_ok=True)

    if args.draft:
        fig = render_frame(900, data)
        out = args.frames_dir / "draft_frame_00900.png"
        fig.savefig(out, dpi=100, facecolor="black")
        plt.close(fig)
        print(f"draft frame → {out}")
        return

    n_render = T if args.n_frames <= 0 else min(args.n_frames, T)
    jobs = [(i, str(args.frames_dir)) for i in range(n_render)]

    print(f"rendering {n_render} frames with {args.workers} workers…")
    with Pool(args.workers, initializer=_init_worker, initargs=(data,)) as pool:
        for _ in tqdm(pool.imap_unordered(_worker_render, jobs, chunksize=4),
                      total=n_render):
            pass

    print("ffmpeg → mp4…")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-framerate", str(args.fps),
        "-i", str(args.frames_dir / "frame_%05d.png"),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:color=black",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "20", "-preset", "fast",
        str(args.out),
    ]
    subprocess.run(cmd, check=True)
    print(f"wrote {args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
