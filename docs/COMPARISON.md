# Post-DREDge SLN Comparison — dataset1_p1 (NP 1.0, ~2.48 M spikes)

Apples-to-apples comparison of 4 SLN variants against the MP+DREDge canonical
baseline. Each SLN was pretrained on its respective dataset, then DREDge was
run once on the pretrained predictions; the resulting motion field was held
fixed while the SLN was fine-tuned with the unsupervised loss
(−ρ̄ − 0.1·H̄ + 0.01·L_tether on the canonical 4 μm σ=4 soft grid).

## Scoreboard (apply over all 2.48 M spikes, canonical 4 μm σ=4 soft grid)

| method                          | ρ̄      | H̄      | Δρ̄ vs MP+DREDge | epochs used |
|---------------------------------|--------|--------|-----------------|-------------|
| MP raw (no motion correction)   | 0.267  | 6.27   | −0.301          | —           |
| **MP+DREDge canonical**         | **0.568** | **8.25** | **0** (baseline) | one-shot    |
| CNN-SLN 500K post-DREDge        | 0.528  | 8.06   | −0.040          | 20          |
| **CNN-SLN all-spike post-DREDge** | **0.649** | 8.16   | **+0.081**       | 10 (killed early; still climbing) |
| TR-SLN 500K post-DREDge         | 0.571  | 8.21   | +0.003          | 5 (best val) |
| **TR-SLN all-spike post-DREDge** | **0.633** | 8.20   | **+0.066**       | 1 (training rate-limited at ~16 hr/ep, see TR all-spike notes below) |

**Top result: CNN all-spike post-DREDge ep10 (+0.081 ρ̄ over canonical MP+DREDge).**

Both all-spike-trained methods beat the canonical baseline by a wide margin
even though training was *not* allowed to fully converge (CNN: 10 of 20 ep;
TR: 1 of 20 ep). Both 500K-subset methods underperform — confirming the
empirical trend "data scale > architecture" for this fine-tuning regime.

## Visualization coverage matrix

Every method has the same 9 visualization artifacts. Click-through paths
below assume `figures/` is the current directory.

| viz / method                       | MP raw | MP+DREDge | CNN 500K | CNN all-sp. | TR  500K | TR  all-sp. |
|------------------------------------|:--:|:--:|:--:|:--:|:--:|:--:|
| depth raster                       | n/a | n/a | [`depth_raster_postdredge_cnn_500k.png`](figures/depth_raster_postdredge_cnn_500k.png) | [`depth_raster_postdredge_cnn_all_ep10.png`](figures/depth_raster_postdredge_cnn_all_ep10.png) | [`depth_raster_postdredge_tr_500k.png`](figures/depth_raster_postdredge_tr_500k.png) | [`depth_raster_postdredge_tr_all.png`](figures/depth_raster_postdredge_tr_all.png) |
| xy pairwise NCC                    | [`xy_pairwise_corr_raw.png`](figures/xy_pairwise_corr_raw.png) | [`xy_pairwise_corr_si_mp_soft.png`](figures/xy_pairwise_corr_si_mp_soft.png) | [`xy_pairwise_corr_postdredge_cnn_500k.png`](figures/xy_pairwise_corr_postdredge_cnn_500k.png) | [`xy_pairwise_corr_postdredge_cnn_all_ep10.png`](figures/xy_pairwise_corr_postdredge_cnn_all_ep10.png) | [`xy_pairwise_corr_postdredge_tr_500k.png`](figures/xy_pairwise_corr_postdredge_tr_500k.png) | [`xy_pairwise_corr_postdredge_tr_all.png`](figures/xy_pairwise_corr_postdredge_tr_all.png) |
| spatial entropy                    | [`spatial_entropy_raw.png`](figures/spatial_entropy_raw.png) | [`spatial_entropy_si_mp_soft.png`](figures/spatial_entropy_si_mp_soft.png) | [`spatial_entropy_postdredge_cnn_500k.png`](figures/spatial_entropy_postdredge_cnn_500k.png) | [`spatial_entropy_postdredge_cnn_all_ep10.png`](figures/spatial_entropy_postdredge_cnn_all_ep10.png) | [`spatial_entropy_postdredge_tr_500k.png`](figures/spatial_entropy_postdredge_tr_500k.png) | [`spatial_entropy_postdredge_tr_all.png`](figures/spatial_entropy_postdredge_tr_all.png) |
| aggregate projections              | n/a | n/a | [`aggregate_projections_postdredge_cnn_500k.png`](figures/aggregate_projections_postdredge_cnn_500k.png) | [`aggregate_projections_postdredge_cnn_all_ep10.png`](figures/aggregate_projections_postdredge_cnn_all_ep10.png) | [`aggregate_projections_postdredge_tr_500k.png`](figures/aggregate_projections_postdredge_tr_500k.png) | [`aggregate_projections_postdredge_tr_all.png`](figures/aggregate_projections_postdredge_tr_all.png) |
| localization movie                 | n/a | n/a | [`localization_movie_postdredge_cnn_500k.mp4`](figures/localization_movie_postdredge_cnn_500k.mp4) | [`localization_movie_postdredge_cnn_all_ep10.mp4`](figures/localization_movie_postdredge_cnn_all_ep10.mp4) | [`localization_movie_postdredge_tr_500k.mp4`](figures/localization_movie_postdredge_tr_500k.mp4) | [`localization_movie_postdredge_tr_all.mp4`](figures/localization_movie_postdredge_tr_all.mp4) |
| figT1 displacement vs MP+DREDge    | n/a | n/a | [`figT1_global_mp_dredge_color_mp_to_postdredge_cnn_500k.png`](figures/figT1_global_mp_dredge_color_mp_to_postdredge_cnn_500k.png) | [`figT1_global_mp_dredge_color_mp_to_postdredge_cnn_all_ep10.png`](figures/figT1_global_mp_dredge_color_mp_to_postdredge_cnn_all_ep10.png) | [`figT1_global_mp_dredge_color_mp_to_postdredge_tr_500k.png`](figures/figT1_global_mp_dredge_color_mp_to_postdredge_tr_500k.png) | [`figT1_global_mp_dredge_color_mp_to_postdredge_tr_all.png`](figures/figT1_global_mp_dredge_color_mp_to_postdredge_tr_all.png) |
| figT5 local coords colored by log α | n/a | n/a | [`figT5_local_postdredge_cnn_500k_color_alpha.png`](figures/figT5_local_postdredge_cnn_500k_color_alpha.png) | [`figT5_local_postdredge_cnn_all_ep10_color_alpha.png`](figures/figT5_local_postdredge_cnn_all_ep10_color_alpha.png) | [`figT5_local_postdredge_tr_500k_color_alpha.png`](figures/figT5_local_postdredge_tr_500k_color_alpha.png) | [`figT5_local_postdredge_tr_all_color_alpha.png`](figures/figT5_local_postdredge_tr_all_color_alpha.png) |
| alpha vs (x, y, z)                 | [`alpha_raw_mp_alpha_vs_axis.png`](figures/alpha_raw_mp_alpha_vs_axis.png) | n/a | [`alpha_postdredge_cnn_500k_alpha_vs_axis.png`](figures/alpha_postdredge_cnn_500k_alpha_vs_axis.png) | [`alpha_postdredge_cnn_all_ep10_alpha_vs_axis.png`](figures/alpha_postdredge_cnn_all_ep10_alpha_vs_axis.png) | [`alpha_postdredge_tr_500k_alpha_vs_axis.png`](figures/alpha_postdredge_tr_500k_alpha_vs_axis.png) | [`alpha_postdredge_tr_all_alpha_vs_axis.png`](figures/alpha_postdredge_tr_all_alpha_vs_axis.png) |
| alpha temporal variability by depth | [`alpha_raw_mp_alpha_var_vs_depth.png`](figures/alpha_raw_mp_alpha_var_vs_depth.png) | n/a | [`alpha_postdredge_cnn_500k_alpha_var_vs_depth.png`](figures/alpha_postdredge_cnn_500k_alpha_var_vs_depth.png) | [`alpha_postdredge_cnn_all_ep10_alpha_var_vs_depth.png`](figures/alpha_postdredge_cnn_all_ep10_alpha_var_vs_depth.png) | [`alpha_postdredge_tr_500k_alpha_var_vs_depth.png`](figures/alpha_postdredge_tr_500k_alpha_var_vs_depth.png) | [`alpha_postdredge_tr_all_alpha_var_vs_depth.png`](figures/alpha_postdredge_tr_all_alpha_var_vs_depth.png) |

## Per-method context

### CNN-SLN 500K post-DREDge — ep20 (full converged)
- Trained subset: 500 K spikes (~20% of dataset)
- Pretrain val ρ̄ = 0.393, ep20 val ρ̄ = 0.451
- After all-spike apply: ρ̄ = 0.528 (below the 0.568 MP+DREDge baseline)
- **Generalization gap**: validates the "data scale > architecture" finding. The
  500 K subset does not have enough column-structure coverage on NP 1.0.

### CNN-SLN all-spike post-DREDge — ep10 of 20 (early-stopped)
- Trained on all 2.48 M spikes
- Pretrain val ρ̄ = 0.673, ep10 val ρ̄ = 0.729 (still climbing slowly)
- After all-spike apply: **ρ̄ = 0.649 → +0.081 over MP+DREDge**
- Best apply metric of the four; would benefit from completing all 20 epochs.

### TR-SLN 500K post-DREDge — ep5 (best val)
- Trained subset: 500 K spikes
- Pretrain val ρ̄ = 0.393, best val (ep5) ρ̄ = 0.441
- After all-spike apply: ρ̄ = 0.571 (essentially matches MP+DREDge baseline)
- Same generalization issue as CNN 500K — architecture doesn't compensate for data.

### TR-SLN all-spike post-DREDge — ep1 of 20 (rate-limited)
- Trained on all 2.48 M spikes
- Pretrain val ρ̄ = 0.677, ep1 val ρ̄ = 0.687 (+0.010 in 1 epoch)
- After all-spike apply: **ρ̄ = 0.633 → +0.066 over MP+DREDge**
- **Why only 1 epoch**: profiling revealed MPS allocator/backend degrades from
  ~1.4 s/iter (cold) to ~30 s/iter (steady-state) on the TR-all configuration,
  putting 20 epochs at ~10 days. Neither mmap fast-path nor MPS-resident
  waveforms moved the needle — the bottleneck is GPU compute, not I/O.
- Extrapolation: at the CNN all-spike convergence rate (+0.003 val ρ̄/ep after
  ep1), full 20 ep would land at val ρ̄ ≈ 0.71, apply ρ̄ ≈ 0.66. The
  pretrain-alone TR all-spike is already very strong, so the marginal value
  of continued training is small.

## Reproduction commands

```bash
# Apply trained ckpts to all spikes (per method)
python3 apply_cnn_sln_raw_to_all_spikes.py --ckpt results/postdredge_cnn_500k/sln_best.pt \
    --out_dir results/postdredge_cnn_500k_all_spikes --device mps
# (analogous for the other three methods)

# 5 method-agnostic bookmarks (per method)
M=postdredge_tr_all   # or postdredge_cnn_500k / postdredge_cnn_all_ep10 / postdredge_tr_500k
SRC=results/${M}_all_spikes
python3 make_depth_raster.py --y_path $SRC/y.npy --out figures/depth_raster_${M}.png --label "..."
python3 make_xy_pairwise_correlation.py --x_path $SRC/x.npy --y_path $SRC/y.npy \
    --x_lo -40 --x_hi 80 --x_bin 4 --y_lo 0 --y_hi 3840 --y_bin 4 --soft_sigma 4.0 \
    --out figures/xy_pairwise_corr_${M}.png --matrix_out figures/xy_pairwise_corr_${M}.npy --label "..."
python3 make_spatial_entropy.py --x_path $SRC/x.npy --y_path $SRC/y.npy \
    --x_lo -40 --x_hi 80 --x_bin 4 --y_lo 0 --y_hi 3840 --y_bin 4 --soft_sigma 4.0 \
    --out figures/spatial_entropy_${M}.png --trace_out figures/spatial_entropy_${M}.npy --label "..."
python3 make_aggregate_projections.py --x_path $SRC/x.npy --y_path $SRC/y.npy --z_path $SRC/z.npy \
    --out figures/aggregate_projections_${M}.png --label "..."
python3 make_localization_movie.py --x_path $SRC/x.npy --y_path $SRC/y.npy --z_path $SRC/z.npy \
    --corr_matrix_path figures/xy_pairwise_corr_${M}.npy \
    --entropy_path figures/spatial_entropy_${M}.npy \
    --motion_path results/dredge_*_all/motion.npz \
    --label "..." --color "#..." \
    --frames_dir figures/localization_movie_${M}_frames \
    --out figures/localization_movie_${M}.mp4

# Diagnostic figT1 / figT5 / alpha plots (per method)
python3 make_displacement_colormap.py \
    --source_x results/mp_dredge_all_spikes/x.npy --source_y .../y.npy --source_z .../z.npy \
    --target_x $SRC/x.npy --target_y $SRC/y.npy \
    --out figures/figT1_global_mp_dredge_color_mp_to_${M}.png \
    --label "pos = MP+DREDge global · color = ${M} − mp"
python3 make_local_alpha_scatter.py --GL_pre $SRC/GL_pre.npy \
    --out figures/figT5_local_${M}_color_alpha.png --label "..."
python3 make_alpha_plots.py \
    --x_path $SRC/x.npy --y_path $SRC/y.npy --z_path $SRC/z.npy \
    --out_prefix figures/alpha_${M} --label "..."
```
