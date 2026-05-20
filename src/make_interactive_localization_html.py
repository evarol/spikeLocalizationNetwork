"""interactive-visualizer  —  self-contained interactive HTML localization explorer.

A bookmarked visualization in the repertoire (method-agnostic, like depth-raster /
aggregate-projections / mp-comparison-scatter). Output convention:
    figures/interactive_localization_<method>.html

Layout:
  - Top-left: 6 pairwise scatter panels of the aggregate LOCAL coords
    (l_x, l_y, l_z) + α, colored by log10 α (Inferno, LOG_VMIN/VMAX = 2.95/3.88).
  - Top-right: side panels (waveforms + 3-D) for the currently-selected spike.
  - Bottom: 3 GLOBAL aggregate panels (x-y, x-z, z-y) in the exact
    aggregate_projections L-layout, with white channel-square markers.
  A dense background sample shows the full distribution; N_int randomly chosen
  spikes are silently "interactive" (rendered identically). Clicking ANY of the
  9 panels snaps to the nearest interactive spike (pixel-accurate) and renders:
    - waveform panel: 10-channel raw (blue) + tPCA-denoised (red) at probe locations
    - 3-D panel: local frame (l_x, l_y, l_z), channels at z=0, anchor, SLN ★.

Works for any method via --gl_pre / --global_dir / --label. If the model has a
4th output (CNN4D), the α dimension is the predicted log10 α; otherwise it falls
back to MP monopolar log10 α (--alpha_mp) so 2D/3D-output models also work.

Data (positions + the interactive spikes' waveforms + channel geometry) is
embedded as JSON. Plotly is loaded from CDN (swap to a local copy for offline).

Examples:
  # CNN4D outer-loop final (predicted α)
  python3 make_interactive_localization_html.py \\
    --gl_pre results/cnn4d_outer_final_all_spikes/GL_pre_dredge.npy \\
    --global_dir results/cnn4d_outer_final_all_spikes \\
    --label "CNN4D outer-loop final" \\
    --out figures/interactive_localization_cnn4d_outer_final.html

  # 2D-loss CNN all-spike ep20 (MP α fallback)
  python3 make_interactive_localization_html.py \\
    --gl_pre results/postdredge_cnn_all_ep20_all_spikes/GL_pre_dredge.npy \\
    --global_dir results/postdredge_cnn_all_ep20_all_spikes \\
    --label "CNN all-spike 2D-loss ep20" \\
    --out figures/interactive_localization_cnn_all_2d_ep20.html
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from relative_xyz_common import build_neighborhood_lookup


def fit_or_load_tpca_90(wf, n_components=8, n_fit_spikes=5000,
                        cache="results/tpca_90sample_model.pkl", seed=0):
    from sklearn.decomposition import PCA
    cache = Path(cache)
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(wf), n_fit_spikes, replace=False))
    traces = np.asarray(wf[idx]).astype(np.float32).reshape(-1, wf.shape[-1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed).fit(traces)
    with open(cache, "wb") as f:
        pickle.dump(pca, f)
    return pca


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gl_pre", default="results/cnn4d_outer_final_all_spikes/GL_pre_dredge.npy", type=Path,
                   help="(N,4) anchor + rel (no motion); rel = GL_pre - anchor4.")
    p.add_argument("--global_dir", default="results/cnn4d_outer_final_all_spikes", type=Path,
                   help="dir with x.npy/y.npy/z.npy = motion-corrected global localizations.")
    p.add_argument("--anchors", default="results/spike_anchors.npy", type=Path)
    p.add_argument("--channels", default="results/spike_channels.npy", type=Path)
    p.add_argument("--ch_locs", default="results/channel_locations.npy", type=Path)
    p.add_argument("--waveforms", default="results/all_spike_waveforms/waveforms_all.npy", type=Path)
    p.add_argument("--alpha_mp", default="results/spike_locs_alpha.npy", type=Path,
                   help="MP monopolar α — used as the color/4th dimension when the model "
                        "doesn't predict α (3-output ckpts, e.g. the 2D-loss CNN).")
    p.add_argument("--label", default="CNN4D outer-loop final",
                   help="title label for the visualizer (the method being shown).")
    p.add_argument("--svg", action="store_true",
                   help="Use SVG scatter (type='scatter') instead of WebGL (scattergl). "
                        "Slower for big backgrounds but avoids the browser WebGL-context "
                        "limit — useful for headless screenshots. Pair with a small --n_bg.")
    p.add_argument("--n_bg", type=int, default=60000, help="background scatter sample size")
    p.add_argument("--n_int", type=int, default=1000, help="number of interactive spikes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="figures/interactive_localization.html", type=Path)
    args = p.parse_args()

    GL = np.load(args.gl_pre).astype(np.float32)              # (N, 3) or (N, 4)
    anchors = np.load(args.anchors).astype(np.float32)        # (N, 3)
    spike_channels = np.load(args.channels).astype(np.int64)
    ch_locs = np.load(args.ch_locs).astype(np.float32)
    N = len(GL)

    # local coords: l_x, l_y = global - anchor; l_z anchor-relative (anchor=0).
    l_x = GL[:, 0] - anchors[:, 0]
    l_y = GL[:, 1] - anchors[:, 1]
    l_z = GL[:, 2]
    # α dimension: predicted log α if the model has a 4th output, else MP monopolar α.
    if GL.shape[1] >= 4:
        l_a = GL[:, 3]
        alpha_src = "predicted log₁₀α"
    else:
        l_a = np.log10(np.clip(np.load(args.alpha_mp).astype(np.float32), 1.0, None))
        alpha_src = "MP monopolar log₁₀α (model has no α head)"
    print(f"  α dimension: {alpha_src}")
    # global (motion-corrected) localizations for the aggregate x-y / z-y / x-z panels
    g_x = np.load(args.global_dir / "x.npy").astype(np.float32)
    g_y = np.load(args.global_dir / "y.npy").astype(np.float32)
    g_z = np.load(args.global_dir / "z.npy").astype(np.float32)
    assert len(g_x) == N, f"global x length {len(g_x)} != N={N}"
    print(f"N={N:,}  l_x∈[{l_x.min():.0f},{l_x.max():.0f}]  l_z∈[{l_z.min():.0f},{l_z.max():.0f}]  "
          f"log α∈[{l_a.min():.2f},{l_a.max():.2f}]  ·  global y∈[{g_y.min():.0f},{g_y.max():.0f}]")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)
    int_idx = np.sort(perm[:args.n_int])                          # interactive
    bg_pool = perm[args.n_int:]
    bg_idx = np.sort(bg_pool[:args.n_bg])                         # background (disjoint)

    def r2(a, nd=2):
        return [round(float(v), nd) for v in a]

    # Background payload (positions only): local + global coords
    bg = {
        "lx": r2(l_x[bg_idx]), "ly": r2(l_y[bg_idx]),
        "lz": r2(l_z[bg_idx]), "la": r2(l_a[bg_idx], 3),
        "gx": r2(g_x[bg_idx]), "gy": r2(g_y[bg_idx], 1), "gz": r2(g_z[bg_idx]),
    }

    # Interactive payload (positions + waveforms + channel geometry)
    wf = np.load(args.waveforms, mmap_mode="r")
    tpca = fit_or_load_tpca_90(wf)
    channel_lookup, anchor_lookup = build_neighborhood_lookup(ch_locs)

    W_int = np.asarray(wf[int_idx]).astype(np.float32)           # (n_int, 10, 90)
    print(f"  loaded {len(int_idx)} interactive waveforms; denoising…")
    spikes = []
    for s, gid in enumerate(int_idx):
        peak_ch = int(spike_channels[gid])
        ch_idx = channel_lookup[peak_ch]
        ch_xy = ch_locs[ch_idx]                                  # (10, 2)
        anchor = anchor_lookup[peak_ch]                          # (x, y, 0)
        Wr = W_int[s]                                            # (10, 90) raw
        Wd = tpca.inverse_transform(tpca.transform(Wr)).astype(np.float32)
        spikes.append({
            "id": int(gid), "peak_ch": peak_ch,
            "lx": round(float(l_x[gid]), 2), "ly": round(float(l_y[gid]), 2),
            "lz": round(float(l_z[gid]), 2), "la": round(float(l_a[gid]), 3),
            "ax": round(float(anchor[0]), 2), "ay": round(float(anchor[1]), 2),
            "cx": r2(ch_xy[:, 0], 1), "cy": r2(ch_xy[:, 1], 1),
            "wr": [r2(Wr[k], 2) for k in range(10)],
            "wd": [r2(Wd[k], 2) for k in range(10)],
        })

    interactive = {
        "lx": [sp["lx"] for sp in spikes], "ly": [sp["ly"] for sp in spikes],
        "lz": [sp["lz"] for sp in spikes], "la": [sp["la"] for sp in spikes],
        "gx": r2(g_x[int_idx]), "gy": r2(g_y[int_idx], 1), "gz": r2(g_z[int_idx]),
        "spikes": spikes,
    }

    # Channel markers for the global aggregate panels (probe geometry, z=0).
    ch = {"x": r2(ch_locs[:, 0], 1), "y": r2(ch_locs[:, 1], 1)}

    payload = json.dumps({"bg": bg, "int": interactive, "ch": ch}, separators=(",", ":"))
    print(f"  payload size: {len(payload)/1e6:.1f} MB")

    html = (HTML_TEMPLATE
            .replace("/*__DATA__*/", payload)
            .replace("__LABEL__", args.label)
            .replace("__ALPHASRC__", alpha_src)
            .replace("__NBG__", f"{args.n_bg//1000}k")
            .replace("__NINT__", str(args.n_int))
            .replace("__SCATTER_TYPE__", "scatter" if args.svg else "scattergl"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"wrote {args.out}  ({args.out.stat().st_size/1e6:.2f} MB)  "
          f"— open in a browser (needs internet for the Plotly CDN)")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Interactive localization — __LABEL__</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body { background:#0d0d0d; color:#eee; font-family:-apple-system,Helvetica,Arial,sans-serif; margin:0; padding:8px; }
  h2 { font-size:15px; font-weight:600; margin:4px 8px; }
  #wrap { display:flex; gap:8px; }
  #left { flex: 1.35; display:grid; grid-template-columns:repeat(3,1fr); grid-template-rows:repeat(2,1fr); gap:4px; }
  #right { flex:1.0; display:flex; flex-direction:column; gap:6px; }
  .scatter { width:100%; height:248px; }
  /* aggregate_projections L-layout: x-y wide top-left, x-z narrow top-right
     (shares lateral-x axis), z-y wide bottom-left (shares depth axis). */
  #globalblock { display:grid; grid-template-columns: 9fr 1.1fr;
                 grid-template-rows: 1fr 1.55fr; gap:4px; margin-top:6px; height:470px; }
  #g_xy { grid-column:1; grid-row:1; } #g_xz { grid-column:2; grid-row:1; }
  #g_zy { grid-column:1; grid-row:2; }
  .gscatter { width:100%; height:100%; }
  #wfpanel { width:100%; height:340px; }
  #panel3d { width:100%; height:330px; }
  #info { font-size:12px; padding:4px 8px; color:#ffd23f; min-height:18px; }
  .hint { font-size:11px; color:#888; margin:2px 8px 6px; }
  .sectlbl { font-size:12px; color:#9ad; margin:8px 8px 2px; font-weight:600; }
</style></head>
<body>
<h2>__LABEL__ — interactive localization · aggregate (l_x, l_y, l_z) + α · click any panel to inspect the nearest spike</h2>
<div class="hint">α dimension = __ALPHASRC__. Background = __NBG__-spike sample (looks like all spikes); __NINT__ spikes are silently interactive — a click snaps to the nearest one and renders its waveforms + 3-D localization on the right.</div>
<div id="wrap">
  <div id="left">
    <div id="s_xy" class="scatter"></div>
    <div id="s_xz" class="scatter"></div>
    <div id="s_xa" class="scatter"></div>
    <div id="s_yz" class="scatter"></div>
    <div id="s_ya" class="scatter"></div>
    <div id="s_za" class="scatter"></div>
  </div>
  <div id="right">
    <div id="info">click a panel to inspect a spike…</div>
    <div id="wfpanel"></div>
    <div id="panel3d"></div>
  </div>
</div>
<div class="sectlbl">Aggregate global localizations (motion-corrected x, y, z) — aggregate-projections layout · white ▪ = channels · click to inspect</div>
<div id="globalblock">
  <div id="g_xy" class="gscatter"></div>
  <div id="g_xz" class="gscatter"></div>
  <div id="g_zy" class="gscatter"></div>
</div>
<script>
const DATA = /*__DATA__*/;
const BG = DATA.bg, INT = DATA.int, CH = DATA.ch;
const ST = "__SCATTER_TYPE__";   // "scattergl" (WebGL) or "scatter" (SVG, for screenshots)
const LBL = {lx:"l_x (μm)", ly:"l_y (μm)", lz:"l_z (μm)", la:"log₁₀ α",
             gx:"x (μm)", gy:"y / depth (μm)", gz:"z (μm)"};
// 6 local pairwise panels + 3 global aggregate panels: [divId, xvar, yvar]
// Global panels follow aggregate_projections: x-y (depth horiz, x vert),
// x-z (z horiz, x vert), z-y (depth horiz, z vert).
const PANELS = [
  ["s_xy","lx","ly"], ["s_xz","lx","lz"], ["s_xa","lx","la"],
  ["s_yz","ly","lz"], ["s_ya","ly","la"], ["s_za","lz","la"],
  ["g_xy","gy","gx"], ["g_xz","gz","gx"], ["g_zy","gy","gz"],
];
const AVMIN=2.95, AVMAX=3.88;        // log₁₀α colormap range (matches static figs)
const DOTSZ=1.4;                      // smaller dots, same everywhere
const RANGES={gx:[-42,82], gy:[-10,3850], gz:[0,65]};   // fixed ranges for global panels
// channel positions per data-var: gx→lateral, gy→depth, gz→0
function chFor(v){ if(v==="gx") return CH.x; if(v==="gy") return CH.y;
                   if(v==="gz") return CH.x.map(_=>0); return null; }

function scatterTrace(xs, ys, cs, sz, op){
  return {x:xs, y:ys, mode:"markers", type:ST,
    marker:{size:sz, color:cs, colorscale:"Inferno", cmin:AVMIN, cmax:AVMAX,
            opacity:op, line:{width:0}}, hoverinfo:"skip", showlegend:false};
}
function chanTrace(xv, yv){   // white channel-square markers (probe geometry)
  return {x:chFor(xv), y:chFor(yv), mode:"markers", type:ST,
    marker:{size:3.2, color:"white", symbol:"square", opacity:0.32, line:{width:0}},
    hoverinfo:"skip", showlegend:false};
}
function selTrace(){   // selection highlight (last trace), starts empty
  return {x:[], y:[], mode:"markers", type:ST,
    marker:{size:13, color:"#00e5ff", symbol:"circle-open", line:{width:2.5, color:"#00e5ff"}},
    hoverinfo:"skip", showlegend:false};
}
function layout(xv, yv){
  const lay = {margin:{l:40,r:6,t:6,b:30}, paper_bgcolor:"#0d0d0d", plot_bgcolor:"#0d0d0d",
    font:{color:"#bbb", size:9},
    xaxis:{title:{text:LBL[xv],standoff:4}, gridcolor:"#333", zerolinecolor:"#555"},
    yaxis:{title:{text:LBL[yv],standoff:4}, gridcolor:"#333", zerolinecolor:"#555"}};
  if(RANGES[xv]){ lay.xaxis.range = RANGES[xv].slice(); lay.xaxis.autorange=false; }
  if(RANGES[yv]){ lay.yaxis.range = RANGES[yv].slice(); lay.yaxis.autorange=false; }
  return lay;
}
const PANELDIVS = [];
// Build panels: [channels (global only)] + background + interactive + selection.
PANELS.forEach(([id,xv,yv])=>{
  const isGlobal = id.startsWith("g_");
  const traces = [];
  if(isGlobal) traces.push(chanTrace(xv, yv));
  traces.push(scatterTrace(BG[xv], BG[yv], BG.la, DOTSZ, 0.5));
  traces.push(scatterTrace(INT[xv], INT[yv], INT.la, DOTSZ, 0.5));
  traces.push(selTrace());
  const selIdx = traces.length - 1;
  Plotly.newPlot(id, traces, layout(xv,yv), {displayModeBar:false, responsive:true});
  PANELDIVS.push({id, xv, yv, selIdx});
  // Robust click handling: scattergl's plotly_click is unreliable, so listen for
  // raw DOM clicks and find the nearest interactive spike in PIXEL space (so
  // "nearest" matches what looks nearest, even when axes have very different
  // ranges — e.g. the global depth axis spans 0–3840 while x spans ~120).
  const gd = document.getElementById(id);
  gd.addEventListener("click", (e)=>{
    const fl = gd._fullLayout; if(!fl || !fl.xaxis) return;
    const xa = fl.xaxis, ya = fl.yaxis;
    const rect = gd.getBoundingClientRect();
    const cpx = e.clientX - rect.left, cpy = e.clientY - rect.top;
    const xr = xa.range, yr = yr_(ya);
    const xlen = xa._length, ylen = ya._length, xoff = xa._offset, yoff = ya._offset;
    if(!xr || !xlen) return;
    let best=-1, bd=Infinity;
    for(let i=0;i<INT[xv].length;i++){
      // data → pixel within the div (x increases right, y increases up → invert)
      const sx = xoff + (INT[xv][i]-xr[0])/(xr[1]-xr[0]) * xlen;
      const sy = yoff + (1 - (INT[yv][i]-yr[0])/(yr[1]-yr[0])) * ylen;
      const dx=sx-cpx, dy=sy-cpy, d=dx*dx+dy*dy;
      if(d<bd){bd=d; best=i;}
    }
    if(best>=0) showSpike(best);
  }, true);
});
function yr_(ya){ return ya.range; }
function updateSelection(i){
  PANELDIVS.forEach(({id,xv,yv,selIdx})=>{
    Plotly.restyle(id, {x:[[INT[xv][i]]], y:[[INT[yv][i]]]}, [selIdx]);
  });
}

function ampScale(wr){
  let m=1e-6; for(const ch of wr) for(const v of ch) if(Math.abs(v)>m) m=Math.abs(v);
  return 17.0/m;   // biggest waveform spans ~34 μm
}
function showSpike(i){
  const sp = INT.spikes[i];
  updateSelection(i);
  document.getElementById("info").textContent =
    `spike #${sp.id} · peak ch ${sp.peak_ch} · anchor=(${sp.ax.toFixed(0)}, ${sp.ay.toFixed(0)})μm · `+
    `l_x=${sp.lx.toFixed(1)} l_y=${sp.ly.toFixed(1)} l_z=${sp.lz.toFixed(1)}μm · log₁₀α=${sp.la.toFixed(2)}`;
  // ---- waveform panel ----
  const nT=sp.wr[0].length, amp=ampScale(sp.wr), tw=14.0;
  const tnorm=[]; for(let j=0;j<nT;j++) tnorm.push((j/(nT-1)-0.5)*tw);
  const traces=[];
  for(let k=0;k<10;k++){
    const xs=tnorm.map(t=>sp.cx[k]+t);
    traces.push({x:xs, y:sp.wr[k].map(v=>sp.cy[k]+v*amp), mode:"lines",
      line:{color:"#7fbfff", width:1}, hoverinfo:"skip", showlegend:(k===0), name:"raw"});
    traces.push({x:xs, y:sp.wd[k].map(v=>sp.cy[k]+v*amp), mode:"lines",
      line:{color:"#ff4d4d", width:1.1}, hoverinfo:"skip", showlegend:(k===0), name:"tPCA denoised"});
  }
  // anchor + SLN (x,y)
  traces.push({x:[sp.ax], y:[sp.ay], mode:"markers",
    marker:{symbol:"circle-open", size:13, color:"#ffd23f", line:{width:2}},
    name:"anchor", hoverinfo:"skip"});
  traces.push({x:[sp.ax+sp.lx], y:[sp.ay+sp.ly], mode:"markers",
    marker:{symbol:"star", size:16, color:"#ff66dd", line:{color:"#fff",width:0.6}},
    name:"SLN (x,y)", hoverinfo:"skip"});
  Plotly.react("wfpanel", traces, {
    margin:{l:42,r:6,t:24,b:34}, paper_bgcolor:"#0d0d0d", plot_bgcolor:"#0d0d0d",
    font:{color:"#bbb", size:9}, title:{text:"waveforms @ probe (blue=raw, red=tPCA)", font:{size:11}},
    xaxis:{title:"x (μm)", range:[sp.ax-60, sp.ax+60], gridcolor:"#333"},
    yaxis:{title:"y / depth (μm)", range:[sp.ay-60, sp.ay+60], gridcolor:"#333", scaleanchor:"x", scaleratio:1},
    legend:{x:0.99,y:0.99,xanchor:"right",font:{size:8},bgcolor:"rgba(0,0,0,0.4)"}
  }, {displayModeBar:false});
  // ---- 3D panel ----
  const cxl=sp.cx.map((v,k)=>v-sp.ax), cyl=sp.cy.map((v,k)=>v-sp.ay), czl=sp.cx.map(_=>0);
  const t3=[
    {type:"scatter3d", x:cxl, y:cyl, z:czl, mode:"markers",
     marker:{size:3, color:"#888"}, name:"channels (z=0)", hoverinfo:"skip"},
    {type:"scatter3d", x:[0], y:[0], z:[0], mode:"markers",
     marker:{symbol:"circle-open", size:6, color:"#ffd23f", line:{width:2}}, name:"anchor", hoverinfo:"skip"},
    {type:"scatter3d", x:[sp.lx], y:[sp.ly], z:[sp.lz], mode:"markers",
     marker:{symbol:"diamond", size:7, color:[sp.la], colorscale:"Inferno", cmin:AVMIN, cmax:AVMAX,
             line:{color:"#fff",width:0.5}}, name:"SLN (l_x,l_y,l_z)", hoverinfo:"skip"},
    {type:"scatter3d", x:[sp.lx,sp.lx], y:[sp.ly,sp.ly], z:[0,sp.lz], mode:"lines",
     line:{color:"#ff66dd", width:3, dash:"dash"}, hoverinfo:"skip", showlegend:false},
  ];
  Plotly.react("panel3d", t3, {
    margin:{l:0,r:0,t:24,b:0}, paper_bgcolor:"#0d0d0d",
    font:{color:"#bbb", size:9}, title:{text:"3-D local frame · ◆ colored by log₁₀α", font:{size:11}},
    scene:{xaxis:{title:"l_x", color:"#bbb", gridcolor:"#333"},
           yaxis:{title:"l_y", color:"#bbb", gridcolor:"#333"},
           zaxis:{title:"l_z", color:"#bbb", gridcolor:"#333"},
           bgcolor:"#0d0d0d"},
    legend:{font:{size:8}, bgcolor:"rgba(0,0,0,0.4)"}
  }, {displayModeBar:false});
}
// auto-show one spike at load
showSpike(0);
</script>
</body></html>
"""


if __name__ == "__main__":
    main()
