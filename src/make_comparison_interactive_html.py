"""Comparison interactive visualizer: toggle between drift-correction methods
(raw / DREDge / MEDICINE, for MP and SLN localizers) and watch the aggregate
localization cloud update. One self-contained HTML per dataset.

Panels (colored by log10 alpha, Inferno): x-y (depth), x-z, z-y — same
convention as the static aggregate projections. Each method shows its pairwise
NCC rho-bar in the header.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

NCC = Path("figures/ncc_matrices")


def load_xyz(spec):
    """spec = ('gl', path) → GL_pre_dredge[:, :3]; ('dir', path) → x/y/z.npy."""
    if spec[0] == "gl":
        g = np.load(spec[1]); return g[:, 0], g[:, 1], g[:, 2]
    d = Path(spec[1]); return (np.load(d / "x.npy"), np.load(d / "y.npy"), np.load(d / "z.npy"))


def rho(key):
    p = NCC / f"{key}.json"
    return json.loads(p.read_text())["rho"] if p.exists() else None


DATASETS = {
    "np2": dict(
        title="NP2 (2.48M spikes · real drift)",
        alpha="results/spike_locs_alpha.npy", n=35000, seed=0,
        methods=[
            ("MP · raw",       "np2_raw_mp",     ("gl", "results/mp_dredge_all_spikes/GL_pre_dredge.npy")),
            ("MP + DREDge",    "np2_mp_dredge",  ("dir", "results/mp_dredge_all_spikes")),
            ("MP + MEDICINE",  "np2_mp_medicine",("dir", "results/np2_mp_medicine")),
            ("SLN · raw",      "np2_raw_sln",    ("gl", "results/cnn3d_outer_final_all_spikes/GL_pre_dredge.npy")),
            ("SLN + DREDge",   "np2_sln_dredge", ("dir", "results/cnn3d_outer_final_all_spikes")),
            ("SLN + MEDICINE", "np2_sln_medicine",("dir", "results/np2_sln_medicine")),
        ],
        out="figures/comparison_interactive_np2.html"),
    "npu": dict(
        title="NP-Ultra (16k spikes · ~0 drift)",
        alpha="results/np_ultra/raw/spike_locs_alpha.npy", n=16000, seed=0,
        methods=[
            ("MP · raw",       "npu_raw_mp",     ("gl", "results/np_ultra/mp_dredge/GL_pre_dredge.npy")),
            ("MP + DREDge",    "npu_mp_dredge",  ("dir", "results/np_ultra/mp_dredge")),
            ("MP + MEDICINE",  "npu_mp_medicine",("dir", "results/np_ultra/mp_medicine")),
            ("SLN · raw",      "npu_raw_sln",    ("gl", "results/np_ultra/sln_final/GL_pre_dredge.npy")),
            ("SLN + DREDge",   "npu_sln_dredge", ("dir", "results/np_ultra/sln_final")),
            ("SLN + MEDICINE", "npu_sln_medicine",("dir", "results/np_ultra/sln_medicine")),
        ],
        out="figures/comparison_interactive_npu.html"),
}


def r1(a): return [round(float(v), 1) for v in a]


def build(ds):
    cfg = DATASETS[ds]
    la_all = np.log10(np.clip(np.load(cfg["alpha"]).astype(np.float32), 1, None))
    N = len(la_all)
    rng = np.random.default_rng(cfg["seed"])
    idx = np.sort(rng.permutation(N)[:min(cfg["n"], N)])
    la = la_all[idx]
    avmin, avmax = float(np.percentile(la, 2)), float(np.percentile(la, 98))

    methods, allx, ally, allz = [], [], [], []
    for label, key, spec in cfg["methods"]:
        x, y, z = load_xyz(spec)
        xi, yi, zi = x[idx], y[idx], z[idx]
        allx.append(xi); ally.append(yi); allz.append(zi)
        methods.append({"label": label, "rho": rho(key),
                        "x": r1(xi), "y": r1(yi), "z": r1(zi)})
    # shared robust ranges across all methods
    def rng2(arrs, lo=1, hi=99, pad=.05):
        v = np.concatenate(arrs); a, b = np.percentile(v, lo), np.percentile(v, hi); m = (b - a) * pad
        return [round(float(a - m), 1), round(float(b + m), 1)]
    ranges = {"x": rng2(allx), "y": rng2(ally), "z": rng2(allz)}

    payload = json.dumps({"methods": methods, "la": r1(la),
                          "ranges": ranges, "avmin": round(avmin, 2), "avmax": round(avmax, 2),
                          "title": cfg["title"]}, separators=(",", ":"))
    html = TEMPLATE.replace("/*__DATA__*/", payload).replace("__TITLE__", cfg["title"])
    Path(cfg["out"]).write_text(html)
    print(f"wrote {cfg['out']}  ({Path(cfg['out']).stat().st_size/1e6:.1f} MB)  {len(idx):,} spikes × {len(methods)} methods")


TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Drift-method comparison — __TITLE__</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
 body{background:#0c0e13;color:#e8ecf4;font-family:-apple-system,Helvetica,Arial,sans-serif;margin:0;padding:12px}
 h2{font-size:16px;margin:2px 0 8px}
 #bar{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px}
 .mbtn{background:#1a1e28;color:#cfd6e4;border:1px solid #2a3040;border-radius:6px;padding:6px 11px;cursor:pointer;font-size:12px}
 .mbtn:hover{background:#232838}
 .mbtn.on{background:#2f6df6;color:#fff;border-color:#2f6df6}
 #panels{display:flex;gap:8px}
 .panel{flex:1;height:520px;background:#15181f;border-radius:6px}
 #info{color:#9fb0c8;font-size:12px;margin:4px 0}
 .hint{color:#6b7688;font-size:11px}
</style></head><body>
<h2>Drift-correction comparison · __TITLE__</h2>
<div id="bar"></div>
<div id="info"></div>
<div id="panels"><div id="p_xy" class="panel"></div><div id="p_xz" class="panel"></div><div id="p_zy" class="panel"></div></div>
<div class="hint">Click a method to swap the localization cloud. Panels: x-y (depth), x-z, z-y · colored by log₁₀α (Inferno). ρ̄ = mean pairwise NCC of drift-corrected 1-s histograms (higher = more temporally consistent).</div>
<script>
const D = /*__DATA__*/;
const R = D.ranges, AV = [D.avmin, D.avmax];
let cur = 1;   // default to first corrected method (MP+DREDge)
function panelData(m, xk, yk){
  return [{x:m[xk], y:m[yk], mode:"markers", type:"scattergl",
    marker:{size:2.2, color:D.la, colorscale:"Inferno", cmin:AV[0], cmax:AV[1], opacity:0.55}}];
}
function lay(xr, yr, xt, yt){
  return {margin:{l:44,r:6,t:26,b:36}, paper_bgcolor:"#15181f", plot_bgcolor:"#15181f",
    font:{color:"#9fb0c8", size:10}, xaxis:{title:xt, range:xr, gridcolor:"#232838", zeroline:false},
    yaxis:{title:yt, range:yr, gridcolor:"#232838", zeroline:false}, showlegend:false};
}
function draw(i){
  cur = i; const m = D.methods[i];
  Plotly.react("p_xy", panelData(m,"y","x"), lay(R.y, R.x, "y / depth (µm)", "x lateral (µm)"), {displayModeBar:false});
  Plotly.react("p_xz", panelData(m,"z","x"), lay(R.z, R.x, "z (µm)", "x lateral (µm)"), {displayModeBar:false});
  Plotly.react("p_zy", panelData(m,"y","z"), lay(R.y, R.z, "y / depth (µm)", "z (µm)"), {displayModeBar:false});
  document.querySelectorAll(".mbtn").forEach((b,k)=>b.classList.toggle("on", k===i));
  document.getElementById("info").textContent =
    `${m.label}  ·  pairwise-NCC ρ̄ = ${m.rho!=null?m.rho.toFixed(3):"n/a"}`;
}
const bar = document.getElementById("bar");
D.methods.forEach((m,i)=>{
  const b=document.createElement("button"); b.className="mbtn";
  b.textContent = m.label + (m.rho!=null? `  (ρ̄ ${m.rho.toFixed(2)})` : "");
  b.onclick=()=>draw(i); bar.appendChild(b);
});
draw(cur);
</script></body></html>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["np2", "npu", "both"], default="both")
    args = p.parse_args()
    for ds in (["np2", "npu"] if args.dataset == "both" else [args.dataset]):
        build(ds)


if __name__ == "__main__":
    main()
