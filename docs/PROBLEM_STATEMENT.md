# neuralSlam — Joint Spike Localization and Drift Correction

## Problem Statement

Neuropixels probes record extracellular voltage from 384 channels at 30 kHz. From this, spike-detection extracts millions of spiking events per session. Two critical preprocessing steps follow: (1) **spike localization** — estimating the 3D position (x, y, z) of each spike source relative to the probe, and (2) **drift correction** — removing the effect of slow mechanical motion of the probe relative to brain tissue.

The standard pipeline runs these sequentially: localize first (monopolar point-source model), then register (DREDge). This is suboptimal because the two problems confound each other — drift biases localization (spikes land on shifted channels), and localization errors corrupt drift estimates. Neither step gets feedback from the other.

**Our approach**: solve localization and drift estimation jointly via alternating optimization between a learned Spike Localization Network (SLN) and the DREDge drift estimator, connected by a self-supervised consistency objective that requires no ground-truth labels.

## Datasets

33 Neuropixels sessions across three cohorts:

| Dataset | Sessions | Probe | Region | Duration | Spikes |
|---------|----------|-------|--------|----------|--------|
| AL032 | 12 | NP 2.0, 4-shank | Visual cortex (V1) | 29–37 min | 2.8–2.9M |
| AL036 | 15 | NP 2.0, 4-shank | Visual cortex (V1) | 30–32 min | 2.7–2.8M |
| Steinmetz (6 recordings) | 6 | NP 1.0 + NP 2.0 | Mouse, multi-area | 32–45 min | 1.4–4.0M |

- Steinmetz recordings have mechanically imposed sawtooth drift profiles (ground-truth motion).
- AL032/AL036 are naturalistic chronic recordings with natural drift.
- All probes have 384 active channels.

## Preprocessing

1. **Spike detection**: via SpikeInterface. Produces N events per session, each with sample index, peak channel, monopolar amplitude, and waveform.
2. **Waveform extraction**: for each spike, extract the tPCA-denoised waveform on the 10 nearest channels (n_nbr=10), 90 temporal samples (1 ms before, 2 ms after peak). Shape: (10, 90).
3. **Monopolar localization**: analytical point-source model fits (x, y, z, alpha) per spike from denoised peak-to-peak amplitudes. These serve as pretraining targets and the baseline.
4. **Input normalization**: waveforms are divided by s0=100 and clipped to ±10. Channel centroids are z-scored per recording.

## Model — Spike Localization Network (SLN)

A **4-layer pre-LN Transformer encoder** (~590K params) that predicts a 3D offset from the channel centroid:

```
Input: waveform ∈ R^{B × 10 × 90}, centroid ∈ R^{B × 2}

Per-channel token embedding:
  - Linear(90 → 128) per channel → (B, 10, 128)     [11K params]
  - + learned row embedding(5 → 128)                  [640 params]
  - + learned column embedding(2 → 128)                [256 params]

Transformer encoder × 4:
  - 8-head self-attention, d_model=128                 [66K params each]
  - FFN: 128 → 256 → 128, GELU                        [66K params each]
  - Pre-LayerNorm, dropout 0.1

Mean pool over 10 channel tokens → (B, 128)

Centroid branch:
  - MLP: 2 → 64 → 64, GELU                            [4K params]

Concat token features ∥ centroid → (B, 192)
Head: 192 → 128 → 64 → 3                              [33K params]

Output: (Δx, Δy, Δz) residual from centroid anchor
```

The row/column positional embeddings encode the physical probe geometry (5 rows × 2 columns on NP 2.0 staggered lattice).

### Pretraining

Supervised regression against monopolar targets using Huber loss (SmoothL1, β=5). AdamW, lr=1e-3, weight decay=1e-4, mixed precision, 50 epochs, batch size 2048. Achieves 30–35 μm 3D RMSE. This checkpoint θ^(0) initializes the self-supervised phase.

## Optimization Algorithm

**Alternating block-coordinate descent** for K=8 outer iterations:

### Phase A — Drift step (θ frozen)
Run DREDge (via SpikeInterface) on the SLN's current localizations to get depth drift Δy(t) and lateral drift Δx(t). DREDge uses pairwise histogram cross-correlation — it is non-differentiable and treated as a black box.

### Phase B — SLN step (drift field T frozen)
For E=5 inner epochs, update θ via gradient descent on:

```
L(θ) = -λ_ρ · ρ̄(θ, T) - λ_H · H̄(θ, T) + λ_teth · L_teth(θ)
```

Where:
- **ρ̄ (pairwise NCC)**: Mean normalized cross-correlation between drift-corrected soft histograms within a temporal window W=30. Rewards temporal consistency. Primary objective.
- **H̄ (spatial entropy)**: Mean Shannon entropy of per-bin normalized histograms. Prevents representational collapse (all spikes collapsing to one point).
- **L_teth (tether)**: MSE between current and pretrained localizations. Prevents anatomically implausible solutions.

**Soft histograms**: Differentiable 2D histograms using separable Gaussian kernel scatter (σ=4 μm) on a voxel grid. This is what makes backprop through NCC possible.

### Hyperparameters
- λ_ρ=1.0, λ_H=0.1, λ_teth=0.01
- AdamW, lr=1e-4, weight decay=1e-5
- K=8 outer iterations, E=5 inner epochs
- Block size B=32 time bins, NCC window W=30
- Gradient clipping at 5.0
- ~2 hours per recording on a single GPU

## Baselines

All share the same detected spikes and channel geometry:

1. **MP+DREDge**: Monopolar localization → one-shot DREDge. The standard pipeline.
2. **CoM+DREDge**: Center-of-mass localization → DREDge.
3. **GridConv+DREDge**: Grid-convolution localization → DREDge.
4. **SLN+DREDge (ours)**: Iteratively trained SLN with DREDge.

For spike-sorting evaluation, also compare:
- **KS+KS**: Stock Kilosort4 (internal localization + drift correction)
- **mono+KS**: External monopolar localization + Kilosort drift correction
- **KS+SI-DREDge**: External DREDge drift + Kilosort sorting

## Evaluation Metrics

1. **Mean pairwise NCC (ρ̄)**: Construct per-second 2D localization histograms from drift-corrected positions, compute full T×T NCC matrix, report mean. Higher = more temporally consistent localization images.

2. **Mean spatial entropy (H̄)**: Mean per-bin Shannon entropy across all time bins. Guards against collapse — if a method inflates NCC by collapsing localizations, entropy drops.

3. **Spike-sorting quality** (via Kilosort4): Number of well-isolated ("good") units, good-to-total cluster fraction, total cluster count.

## Results Summary

- **NCC**: SLN+DREDge achieves higher pairwise NCC than MP+DREDge on the majority of the 33 sessions. Largest gains on Steinmetz imposed-motion benchmarks (where drift is largest). Consistent improvement on AL032/AL036 naturalistic recordings.
- **Entropy**: Points cluster near the diagonal (SLN vs MP), confirming no representational collapse.
- **Spike sorting**: SLN+DREDge yields highest mean good-unit count (~48 vs ~39 for stock Kilosort). Total cluster count stable (~1200–1300). Joint approach does not degrade sorting quality.
- **Qualitative**: SLN+DREDge resolves two units within a depth band that are merged under MP+DREDge (visible in zoomed rasters).

## Figures

All figures are in `paper/figures/`:

- **Fig1.png** — Method overview schematic. Panels: (A) raw spike raster, (B) waveform extraction, (C) SLN transformer architecture, (D) pairwise NCC matrix, (E) DREDge drift correction, (F) SLN vs monopolar localizations, (G) entropy trace, (H) per-axis scatter.
- **sln_drift_correction.pdf** — Qualitative drift correction on two imposed-motion recordings. Rows: raw monopolar, MP+DREDge, SLN+DREDge. Columns: (A) full-session rasters, (B) zoomed view, (C) NCC matrices, (D) entropy traces.
- **figure3.png** — Multi-dataset scatter plots. Top row: NCC (MP+DREDge x-axis vs SLN+DREDge y-axis) per dataset. Bottom row: entropy on same axes. Additional panels compare all four localizers on representative recordings.
- **fig4.pdf** — Spike-sorting quality bar plots. Three panels: good units, good/total fraction, total clusters. Four preprocessing configs compared (mean ± SEM, individual session dots).

## Key Dependencies

- SpikeInterface (spike detection, waveform extraction, DREDge integration)
- DREDge (drift estimation via pairwise histogram cross-correlation)
- Kilosort4 (downstream spike sorting evaluation)
- PyTorch (SLN training)

## Paper

The NeurIPS 2026 submission is in `paper/neurips.tex` with bibliography in `paper/references.bib`. Uses `paper/neurips_2026.sty` (custom style with numbered natbib citations). Compiles via `pdflatex → bibtex → pdflatex → pdflatex`.
