"""
Shared utilities for the fixed-window spike localization video design.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "data/dataset1"
RESULTS_DIR = "results"
N_GROUPS = 10
GROUP_SIZE = 10
MS_BEFORE = 10.0
MS_AFTER = 10.0
X_RANGE = (-25, 73)
Y_RANGE = (0, 3840)
RNG_SEED = 42
SPIKE_COLORS = plt.cm.tab10(np.linspace(0, 1, N_GROUPS))
LOG_VMIN = np.log10(900)
LOG_VMAX = np.log10(12000)
RECENTER_SEARCH_MS = 0.75
ZSCORE_CHUNK_S = 20.0


def compute_frame_offsets(frame_ids):
    counts = np.bincount(frame_ids, minlength=int(frame_ids.max()) + 1)
    return np.concatenate([[0], counts.cumsum()]).astype(np.int64)


def nearest_channels(ch_locs, center_channel, n=GROUP_SIZE):
    dists = np.linalg.norm(ch_locs - ch_locs[center_channel], axis=1)
    return np.argsort(dists)[:n].astype(np.int32)


def load_localization_results(results_dir=RESULTS_DIR, loc_dir=None):
    """
    Load spike lists from ``results_dir`` (times, channels, fs, geometry).

    If ``loc_dir`` is set, load spike_locs_x/y/(z)/alpha from that directory instead
    (e.g. rolling tPCA monopolar outputs).
    """
    fs = float(np.load(os.path.join(results_dir, "fs.npy"))[0])
    spike_times = np.load(os.path.join(results_dir, "spike_times.npy"))
    spike_channels = np.load(os.path.join(results_dir, "spike_channels.npy")).astype(np.int32)
    loc_root = loc_dir if loc_dir is not None else results_dir
    locs_x = np.load(os.path.join(loc_root, "spike_locs_x.npy"))
    locs_y = np.load(os.path.join(loc_root, "spike_locs_y.npy"))
    alpha_path = os.path.join(loc_root, "spike_locs_alpha.npy")
    if os.path.exists(alpha_path):
        alpha = np.load(alpha_path)
    else:
        alpha = np.abs(np.load(os.path.join(results_dir, "spike_amplitudes.npy")))
    ch_locs = np.load(os.path.join(results_dir, "channel_locations.npy"))

    frame_ids = (spike_times / fs).astype(np.int32)
    frame_offsets = compute_frame_offsets(frame_ids)
    out = {
        "fs": fs,
        "spike_times": spike_times,
        "spike_channels": spike_channels,
        "locs_x": locs_x,
        "locs_y": locs_y,
        "alpha": alpha,
        "ch_locs": ch_locs,
        "frame_ids": frame_ids,
        "frame_offsets": frame_offsets,
    }
    z_path = os.path.join(loc_root, "spike_locs_z.npy")
    if os.path.exists(z_path):
        out["locs_z"] = np.load(z_path)
    return out


def select_fixed_groups(spike_channels, ch_locs, n_groups=N_GROUPS, group_size=GROUP_SIZE):
    channel_counts = np.bincount(spike_channels, minlength=ch_locs.shape[0])
    active_channels = np.flatnonzero(channel_counts > 0)
    if len(active_channels) == 0:
        raise RuntimeError("No active channels found in spike results")

    active_y = ch_locs[active_channels, 1]
    depth_edges = np.linspace(active_y.min(), active_y.max() + 1e-6, n_groups + 1)

    groups = []
    centers = []
    used_channels = set()

    def try_add(center_channel):
        subset = nearest_channels(ch_locs, int(center_channel), group_size)
        subset_set = set(int(ch) for ch in subset)
        if used_channels & subset_set:
            return False
        groups.append(subset)
        centers.append(int(center_channel))
        used_channels.update(subset_set)
        return True

    for i in range(n_groups):
        in_bin = active_channels[
            (ch_locs[active_channels, 1] >= depth_edges[i]) &
            (ch_locs[active_channels, 1] < depth_edges[i + 1])
        ]
        ranked = in_bin[np.argsort(channel_counts[in_bin])[::-1]]
        for center_channel in ranked:
            if try_add(center_channel):
                break

    if len(groups) < n_groups:
        ranked_all = active_channels[np.argsort(channel_counts[active_channels])[::-1]]
        for center_channel in ranked_all:
            if len(groups) >= n_groups:
                break
            try_add(center_channel)

    if len(groups) < n_groups:
        raise RuntimeError(f"Could only build {len(groups)} non-overlapping channel groups")

    groups = np.stack(groups[:n_groups]).astype(np.int32)
    centers = np.array(centers[:n_groups], dtype=np.int32)

    order = np.argsort(ch_locs[centers, 1])
    groups = groups[order]
    centers = centers[order]
    return {
        "group_channels": groups,
        "group_centers": centers,
        "eligible_channels": np.unique(groups.ravel()).astype(np.int32),
        "channel_counts": channel_counts,
    }


def sample_spikes_per_frame(frame_offsets, spike_channels, eligible_channels,
                            locs_x=None, locs_y=None, seed=RNG_SEED):
    eligible_mask = np.isin(spike_channels, eligible_channels)
    n_frames = len(frame_offsets) - 1
    sampled = np.full(n_frames, -1, dtype=np.int64)

    for frame_idx in range(n_frames):
        start = int(frame_offsets[frame_idx])
        end = int(frame_offsets[frame_idx + 1])
        candidates = np.flatnonzero(eligible_mask[start:end]) + start
        if len(candidates) == 0:
            continue

        if locs_x is not None and locs_y is not None:
            visible = candidates[
                (locs_x[candidates] >= X_RANGE[0]) & (locs_x[candidates] <= X_RANGE[1]) &
                (locs_y[candidates] >= Y_RANGE[0]) & (locs_y[candidates] <= Y_RANGE[1])
            ]
            if len(visible) > 0:
                candidates = visible

        rng = np.random.default_rng(seed + frame_idx)
        sampled[frame_idx] = int(candidates[rng.integers(len(candidates))])

    return sampled


def load_preprocessed_recording(data_dir=DATA_DIR):
    import warnings
    warnings.filterwarnings("ignore")
    import spikeinterface.extractors as se
    import spikeinterface.preprocessing as sp

    rec = se.read_spikeglx(data_dir, stream_name="imec0.ap", load_sync_channel=False)
    rec = sp.bandpass_filter(rec, freq_min=300, freq_max=6000)
    rec = sp.common_reference(rec, reference="global", operator="median")
    return rec


def compute_channel_zscore_stats(recording, channel_ids, chunk_duration_s=ZSCORE_CHUNK_S):
    channel_ids = np.asarray(channel_ids, dtype=np.int32)
    recording_channel_ids = np.asarray(recording.get_channel_ids())
    selected_channel_ids = recording_channel_ids[channel_ids].tolist()
    fs = float(recording.get_sampling_frequency())
    chunk_size = max(1, int(round(chunk_duration_s * fs)))
    n_samples = int(recording.get_num_samples())

    count = 0
    sum_x = np.zeros(channel_ids.size, dtype=np.float64)
    sum_x2 = np.zeros(channel_ids.size, dtype=np.float64)

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        traces = recording.get_traces(
            start_frame=start,
            end_frame=end,
            channel_ids=selected_channel_ids,
        ).astype(np.float64, copy=False)
        if traces.size == 0:
            continue

        chunk_count = traces.shape[0]
        sum_x += traces.sum(axis=0)
        sum_x2 += np.square(traces, dtype=np.float64).sum(axis=0)
        count += chunk_count

    if count < 2:
        raise RuntimeError("Need at least two samples to compute z-score statistics")

    mean = sum_x / count
    var = (sum_x2 - (sum_x * sum_x) / count) / (count - 1)
    var = np.clip(var, 1e-12, None)
    std = np.sqrt(var)
    std = np.clip(std, 1e-6, None)
    return mean.astype(np.float32), std.astype(np.float32)


def load_frame_traces(recording, frame_idx, fs, ms_before=MS_BEFORE, ms_after=MS_AFTER,
                      extra_pad_ms=0.0):
    pad = int(round(max(ms_before, ms_after) / 1000 * fs))
    extra_pad = int(round(extra_pad_ms / 1000 * fs))
    frame_start = int(frame_idx * fs)
    frame_end = min(int((frame_idx + 1) * fs), recording.get_num_samples())
    trace_start = max(0, frame_start - pad - extra_pad)
    trace_end = min(recording.get_num_samples(), frame_end + pad + extra_pad)
    traces = recording.get_traces(start_frame=trace_start, end_frame=trace_end).astype(np.float32)
    return traces, trace_start


def recenter_spike_time_to_max_abs(traces, trace_start, spike_time, channel_ids, fs,
                                   search_ms=RECENTER_SEARCH_MS,
                                   zscore_mean=None, zscore_std=None):
    search_radius = max(1, int(round(search_ms / 1000 * fs)))
    rel = int(spike_time - trace_start)
    start = max(0, rel - search_radius)
    end = min(traces.shape[0], rel + search_radius + 1)
    if end - start <= 0:
        raise RuntimeError("Recentering search window fell outside available traces")

    window = traces[start:end, channel_ids].astype(np.float32, copy=False)
    if zscore_mean is not None and zscore_std is not None:
        window = (window - zscore_mean[None, :]) / zscore_std[None, :]

    sample_offset = int(np.abs(window).reshape(window.shape[0], -1).max(axis=1).argmax())
    return int(trace_start + start + sample_offset)


def extract_waveforms_for_spike(traces, trace_start, spike_time, group_channels, fs,
                                ms_before=MS_BEFORE, ms_after=MS_AFTER,
                                zscore_mean=None, zscore_std=None):
    n_before = int(round(ms_before / 1000 * fs))
    n_after = int(round(ms_after / 1000 * fs))
    t_size = n_before + n_after

    rel = int(spike_time - trace_start)
    s = rel - n_before
    e = rel + n_after
    snippet = traces[s:e, :]
    if snippet.shape[0] != t_size:
        raise RuntimeError("Waveform extraction window fell outside available traces")

    flat_channels = group_channels.ravel()
    channel_snippet = snippet[:, flat_channels]
    if zscore_mean is not None and zscore_std is not None:
        channel_snippet = (channel_snippet - zscore_mean[None, :]) / zscore_std[None, :]

    waveforms = channel_snippet.T.reshape(
        group_channels.shape[0], group_channels.shape[1], t_size
    )
    return waveforms.astype(np.float32)


def compute_highlight_window(waveforms):
    return int(np.abs(waveforms).reshape(waveforms.shape[0], -1).max(axis=1).argmax())


def make_figure():
    fig = plt.figure(figsize=(18, 26), facecolor="black")

    panel_h = 0.115
    panel_w = panel_h * 26.0 / 18.0
    x_left = 0.05
    x_mid = (1.0 - panel_w) / 2.0
    x_right = 1.0 - 0.05 - panel_w
    y_top = 0.85
    y_mid_up = 0.605
    y_mid_low = 0.335
    y_bottom = 0.06

    top_axes = [fig.add_axes([x, y_top, panel_w, panel_h]) for x in [x_left, x_mid, x_right]]
    left_axes = [fig.add_axes([x_left, y_mid_up, panel_w, panel_h]),
                 fig.add_axes([x_left, y_mid_low, panel_w, panel_h])]
    right_axes = [fig.add_axes([x_right, y_mid_up, panel_w, panel_h]),
                  fig.add_axes([x_right, y_mid_low, panel_w, panel_h])]
    bottom_axes = [fig.add_axes([x, y_bottom, panel_w, panel_h]) for x in [x_left, x_mid, x_right]]
    ax_panels = top_axes + left_axes + right_axes + bottom_axes

    scatter_w = 0.095
    scatter_h = 0.3946153846
    scatter_x = 0.4525
    scatter_y = 0.3026923077
    ax_scatter = fig.add_axes([scatter_x, scatter_y, scatter_w, scatter_h])

    for ax in ax_panels + [ax_scatter]:
        ax.set_facecolor("#0d0d0d")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    return fig, ax_scatter, ax_panels


def draw_window_panel(ax, waveform, channel_ids, ch_locs, is_highlight, color, label,
                      global_amp, panel_xlim, panel_ylim,
                      ms_before=MS_BEFORE, ms_after=MS_AFTER):
    border_color = color if is_highlight else mcolors.to_rgba(color, alpha=0.45)
    title_color = color if is_highlight else mcolors.to_rgba(color, alpha=0.75)
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(1.5 if is_highlight else 1.0)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    pos = ch_locs[channel_ids].astype(np.float32)
    pos_local = pos - pos.mean(axis=0, keepdims=True)
    t = np.linspace(-ms_before, ms_after, waveform.shape[1])
    max_amp = max(global_amp, 1.0)

    x_half = 7.0
    x_scale = x_half / max(ms_before, ms_after)
    amp_half = 26.0
    strongest = int(np.abs(waveform).reshape(len(channel_ids), -1).max(axis=1).argmax())

    for i, (cx, cy) in enumerate(pos_local):
        if is_highlight:
            trace_color = "#f5f5f5" if i == strongest else color
            trace_alpha = 0.95 if i == strongest else 0.72
            trace_lw = 1.15 if i == strongest else 0.85
            dot_color = "#ffffff" if i == strongest else color
            dot_size = 20 if i == strongest else 10
        else:
            trace_color = "#b0b0b0"
            trace_alpha = 0.18
            trace_lw = 0.5
            dot_color = "#777777"
            dot_size = 8

        x_trace = cx + t * x_scale
        y_trace = cy + (waveform[i] / max_amp) * amp_half
        ax.plot(x_trace, y_trace, color=trace_color, lw=trace_lw,
                alpha=trace_alpha, rasterized=True)
        ax.scatter([cx], [cy], s=dot_size, color=dot_color, zorder=4)
        ax.plot([cx, cx], [cy - amp_half, cy + amp_half],
                color="#ff6666", lw=0.2, alpha=0.22)

    ax.set_xlim(*panel_xlim)
    ax.set_ylim(*panel_ylim)
    ax.set_title(
        f"#{label}  ch {int(channel_ids[strongest])}",
        color=title_color, fontsize=6.5, pad=1
    )


def render_frame(frame_idx, data):
    fig, ax_scatter, ax_panels = make_figure()

    locs_x = data["locs_x"]
    locs_y = data["locs_y"]
    alpha = data["alpha"]
    ch_locs = data["ch_locs"]
    frame_offsets = data["frame_offsets"]
    group_channels = data["group_channels"]
    waveforms = data["waveforms"][frame_idx].astype(np.float32)
    sampled_spike_idx = int(data["sampled_spike_idx"][frame_idx])
    highlight_window = int(data["highlight_window"][frame_idx])
    ms_before = float(data["ms_before"])
    ms_after = float(data["ms_after"])

    norm = mcolors.Normalize(vmin=LOG_VMIN, vmax=LOG_VMAX, clip=True)
    cmap = matplotlib.colormaps["inferno"]

    start = int(frame_offsets[frame_idx])
    end = int(frame_offsets[frame_idx + 1])
    bx = locs_x[start:end]
    by = locs_y[start:end]
    ba = alpha[start:end]
    visible = (bx >= X_RANGE[0]) & (bx <= X_RANGE[1]) & (by >= Y_RANGE[0]) & (by <= Y_RANGE[1])
    ax_scatter.scatter(
        bx[visible], by[visible],
        c=cmap(norm(np.log10(np.clip(ba[visible], 900, None)))),
        s=6, alpha=0.46, linewidths=0, rasterized=True, zorder=1
    )

    ax_scatter.scatter(
        ch_locs[:, 0], ch_locs[:, 1],
        s=4, c="white", alpha=0.9, linewidths=0, marker="s", zorder=2
    )
    for group_idx in range(N_GROUPS):
        ax_scatter.scatter(
            ch_locs[group_channels[group_idx], 0],
            ch_locs[group_channels[group_idx], 1],
            s=6, c=[SPIKE_COLORS[group_idx]], alpha=0.95, linewidths=0, marker="s", zorder=3
        )

    if sampled_spike_idx >= 0:
        highlight_color = SPIKE_COLORS[highlight_window] if highlight_window >= 0 else "#ffffff"
        ax_scatter.scatter(
            [locs_x[sampled_spike_idx]], [locs_y[sampled_spike_idx]],
            s=150, facecolors=highlight_color, edgecolors="white",
            linewidths=1.2, zorder=6
        )

    ax_scatter.set_xlim(*X_RANGE)
    ax_scatter.set_ylim(*Y_RANGE)
    ax_scatter.set_xlabel("x (µm)", fontsize=8, color="white")
    ax_scatter.set_ylabel("depth (µm)", fontsize=8, color="white")
    ax_scatter.set_title(f"t = {frame_idx}-{frame_idx + 1} s", fontsize=9, color="white")
    ax_scatter.tick_params(colors="white", labelsize=7)
    for spine in ax_scatter.spines.values():
        spine.set_edgecolor("#555555")

    local_positions = []
    for group_idx in range(N_GROUPS):
        pos = ch_locs[group_channels[group_idx]].astype(np.float32)
        local_positions.append(pos - pos.mean(axis=0, keepdims=True))
    local_positions = np.stack(local_positions)

    x_half = 7.0
    amp_half = 26.0
    global_amp = float(np.percentile(np.abs(waveforms), 99))
    if global_amp < 1e-6:
        global_amp = 1.0

    panel_xlim = (
        float(local_positions[:, :, 0].min() - x_half - 4),
        float(local_positions[:, :, 0].max() + x_half + 4),
    )
    panel_ylim = (
        float(local_positions[:, :, 1].min() - amp_half - 10),
        float(local_positions[:, :, 1].max() + amp_half + 10),
    )

    for group_idx in range(N_GROUPS):
        draw_window_panel(
            ax_panels[group_idx],
            waveforms[group_idx],
            group_channels[group_idx],
            ch_locs,
            is_highlight=(group_idx == highlight_window),
            color=SPIKE_COLORS[group_idx],
            label=group_idx + 1,
            global_amp=global_amp,
            panel_xlim=panel_xlim,
            panel_ylim=panel_ylim,
            ms_before=ms_before,
            ms_after=ms_after,
        )

    return fig
