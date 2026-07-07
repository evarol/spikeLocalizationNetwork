"""Depth-vs-time raster (amplitude-weighted 2-D histogram) for all drift-correction
methods, as a 2×3 grid per dataset: rows=localizer (MP,SLN), cols=raw/DREDge/MEDICINE.
A good correction straightens the wavy activity bands."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

MD = Path("figures/ncc_matrices")     # reuse cached ρ̄
BG, TX, GR = "#0c0e13", "#e8ecf4", "#2a3040"
COLS = [("raw", "raw (no correction)"), ("dredge", "+ DREDge"), ("medicine", "+ MEDICINE")]
ROWS = [("mp", "MP"), ("sln", "SLN")]

# per dataset: times, fs, amplitudes, y-range, time-range, per-method y source
DS = {
  "np2": dict(
      title="NP2 (2.48M spikes, real drift)",
      times="results/spike_times.npy", fs="results/fs.npy", amps="results/spike_amplitudes.npy",
      ylim=(-40, 3900), ybin=4.0, tbin=1.0,
      src={"raw_mp": ("gl", "results/mp_dredge_all_spikes/GL_pre_dredge.npy"),
           "mp_dredge": ("file", "results/mp_dredge_all_spikes"),
           "mp_medicine": ("file", "results/np2_mp_medicine"),
           "raw_sln": ("gl", "results/cnn3d_outer_final_all_spikes/GL_pre_dredge.npy"),
           "sln_dredge": ("file", "results/cnn3d_outer_final_all_spikes"),
           "sln_medicine": ("file", "results/np2_sln_medicine")}),
  "npu": dict(
      title="NP-Ultra (16k spikes, ~0 drift)",
      times="results/np_ultra/raw/spike_times.npy", fs="results/np_ultra/raw/fs.npy",
      amps="results/np_ultra/raw/spike_amplitudes.npy",
      ylim=(-50, 300), ybin=3.0, tbin=1.0,
      src={"raw_mp": ("gl", "results/np_ultra/mp_dredge/GL_pre_dredge.npy"),
           "mp_dredge": ("file", "results/np_ultra/mp_dredge"),
           "mp_medicine": ("file", "results/np_ultra/mp_medicine"),
           "raw_sln": ("gl", "results/np_ultra/sln_final/GL_pre_dredge.npy"),
           "sln_dredge": ("file", "results/np_ultra/sln_final"),
           "sln_medicine": ("file", "results/np_ultra/sln_medicine")}),
}


def load_y(spec):
    return np.load(spec[1])[:, 1] if spec[0] == "gl" else np.load(Path(spec[1]) / "y.npy")


def rho_of(ds, row, col):
    key = f"{ds}_raw_{row}" if col == "raw" else f"{ds}_{row}_{col}"
    p = MD / f"{key}.json"
    return json.loads(p.read_text()).get("rho") if p.exists() else None


for ds, cfg in DS.items():
    st = np.load(cfg["times"]); fs = float(np.load(cfg["fs"]).reshape(-1)[0])
    t_s = st / fs; amp = np.abs(np.load(cfg["amps"]).astype(np.float64))
    tb = np.arange(0, t_s.max() + cfg["tbin"], cfg["tbin"])
    yb = np.arange(cfg["ylim"][0], cfg["ylim"][1] + cfg["ybin"], cfg["ybin"])
    # build all 6 histograms, then a shared vmax for fair comparison
    Hs = {}
    for row, _ in ROWS:
        for col, _ in COLS:
            key = f"raw_{row}" if col == "raw" else f"{row}_{col}"
            y = load_y(cfg["src"][key])
            H, _, _ = np.histogram2d(t_s, y, bins=[tb, yb], weights=amp)
            Hs[(row, col)] = H.T                       # depth on rows
    allmax = max(h.max() for h in Hs.values())
    vmin, vmax = allmax * 3e-3, allmax

    fig, ax = plt.subplots(2, 3, figsize=(16, 8.2), facecolor=BG)
    extent = [tb[0], tb[-1], yb[0], yb[-1]]
    for r, (rk, rlab) in enumerate(ROWS):
        for c, (ck, clab) in enumerate(COLS):
            a = ax[r, c]; H = np.clip(Hs[(rk, ck)], vmin, None)
            im = a.imshow(H, origin="lower", aspect="auto", extent=extent,
                          cmap="inferno", norm=LogNorm(vmin=vmin, vmax=vmax))
            rho = rho_of(ds, rk, ck)
            a.set_title(f"{rlab} {clab}" + (f"   ρ̄={rho:.3f}" if rho else ""), color=TX, fontsize=10)
            a.set_xlabel("time (s)", color=TX, fontsize=8); a.set_ylabel("depth y (µm)", color=TX, fontsize=8)
            a.tick_params(colors=TX, labelsize=7)
            for s in a.spines.values(): s.set_color(GR)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02); cb.set_label("Σ|amplitude| (log)", color=TX); cb.ax.tick_params(colors=TX)
    fig.suptitle(f"Depth-vs-time raster (amplitude-weighted) · {cfg['title']}", color=TX, fontsize=14, y=0.98)
    out = MD.parent / f"depth_raster_grid_{ds}.png"
    fig.savefig(out, dpi=120, facecolor=BG, bbox_inches="tight"); print("wrote", out)
