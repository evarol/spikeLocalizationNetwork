"""Render the T×T pairwise-NCC matrices as a 2×3 grid per dataset:
rows = localizer (MP, SLN), cols = drift correction (raw, +DREDge, +MEDICINE)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

MD = Path("figures/ncc_matrices")
BG, TX, GR = "#0c0e13", "#e8ecf4", "#2a3040"
COLS = [("raw", "raw (no correction)"), ("dredge", "+ DREDge"), ("medicine", "+ MEDICINE")]
ROWS = [("mp", "MP localizer"), ("sln", "SLN localizer")]
DATASETS = {"np2": "NP2 (2.48M spikes, real drift)", "npu": "NP-Ultra (16k, ~0 drift)"}


def key_for(ds, row, col):
    return f"{ds}_raw_{row}" if col == "raw" else f"{ds}_{row}_{col}"


for ds, dstitle in DATASETS.items():
    fig, ax = plt.subplots(2, 3, figsize=(15, 9.6), facecolor=BG)
    for r, (rk, rlabel) in enumerate(ROWS):
        for c, (ck, clabel) in enumerate(COLS):
            a = ax[r, c]; key = key_for(ds, rk, ck)
            f = MD / f"{key}.npy"
            if not f.exists():
                a.text(0.5, 0.5, f"missing\n{key}", ha="center", va="center", color="#f55", transform=a.transAxes)
                a.set_facecolor("#15181f"); continue
            C = np.load(f)
            rho = json.loads((MD / f"{key}.json").read_text()).get("rho")
            im = a.imshow(C, cmap="magma", vmin=0, vmax=1, origin="lower", aspect="auto")
            a.set_title(f"{rlabel} {clabel}    ρ̄={rho:.3f}", color=TX, fontsize=10)
            a.set_xlabel("time bin (s)", color=TX, fontsize=8); a.set_ylabel("time bin (s)", color=TX, fontsize=8)
            a.tick_params(colors=TX, labelsize=7)
            for s in a.spines.values(): s.set_color(GR)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02); cb.set_label("pairwise NCC", color=TX); cb.ax.tick_params(colors=TX)
    fig.suptitle(f"Pairwise NCC matrix over motion-corrected 1-s bins · {dstitle}", color=TX, fontsize=14, y=0.97)
    out = MD.parent / f"ncc_matrix_grid_{ds}.png"
    fig.savefig(out, dpi=120, facecolor=BG, bbox_inches="tight")
    print("wrote", out)
