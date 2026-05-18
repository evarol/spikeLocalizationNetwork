"""Transformer-SLN as described in the paper.

  per-channel token = Linear(90 → 128) applied to each of the 10 nearest-channel
                       waveforms, + learned row & column positional embeddings.
  4 × pre-LN encoder layers (8-head, d_model=128, FFN 256, GELU, dropout 0.1).
  Mean-pool over channel tokens.
  Centroid branch: Linear(2→64) → GELU → Linear(64→64).
  Head: Linear(192→128) → GELU → Linear(128→64) → GELU → Linear(64→3).

Output is the (Δx, Δy, Δz) residual from the channel anchor.

For NP 1.0 (which dataset1_p1 uses), each spike's 10 nearest channels form a
local 5-row × 2-col staggered neighborhood. Row/col indices are *positional
within the neighborhood*, not global probe columns — same convention as the
paper's NP 2.0 description, validated to apply to NP 1.0 too.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


N_NEIGHBORS = 10
N_TIME_SAMPLES = 90
N_ROWS = 5
N_COLS = 2
D_MODEL = 128
N_HEADS = 8
N_LAYERS = 4
D_FF = 256
DROPOUT = 0.1
CENTROID_HIDDEN = 64
HEAD_HIDDEN = 128
HEAD_MID = 64


class TransformerSLN(nn.Module):
    def __init__(self, centroid_mean: tuple[float, float] = (0.0, 0.0),
                 centroid_std: tuple[float, float] = (1.0, 1.0)):
        super().__init__()
        # Per-channel time→d_model projection (shared across the 10 channel tokens).
        self.token_proj = nn.Linear(N_TIME_SAMPLES, D_MODEL)

        self.row_emb = nn.Embedding(N_ROWS, D_MODEL)
        self.col_emb = nn.Embedding(N_COLS, D_MODEL)

        # Channel-index → (row, col) for the lex(y, x)-sorted 10 nearest channels.
        # Row = i // 2, col = i % 2. Fixed across all spikes (geometry-determined).
        row_idx = torch.tensor([i // N_COLS for i in range(N_NEIGHBORS)], dtype=torch.long)
        col_idx = torch.tensor([i %  N_COLS for i in range(N_NEIGHBORS)], dtype=torch.long)
        self.register_buffer("row_idx", row_idx)
        self.register_buffer("col_idx", col_idx)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=D_FF,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # pre-LN
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)

        self.centroid_mlp = nn.Sequential(
            nn.Linear(2, CENTROID_HIDDEN),
            nn.GELU(),
            nn.Linear(CENTROID_HIDDEN, CENTROID_HIDDEN),
        )

        self.head = nn.Sequential(
            nn.Linear(D_MODEL + CENTROID_HIDDEN, HEAD_HIDDEN),
            nn.GELU(),
            nn.Linear(HEAD_HIDDEN, HEAD_MID),
            nn.GELU(),
            nn.Linear(HEAD_MID, 3),
        )

        # Centroid normalization stats — baked into the module as buffers so
        # inference reuses the exact training-time normalization.
        self.register_buffer("centroid_mean", torch.tensor(centroid_mean, dtype=torch.float32))
        self.register_buffer("centroid_std",  torch.tensor(centroid_std,  dtype=torch.float32))

    def forward(self, waveforms: torch.Tensor, centroid: torch.Tensor) -> torch.Tensor:
        """waveforms: (B, 10, 90), centroid: (B, 2). Returns (B, 3)."""
        B = waveforms.shape[0]
        # Per-channel token embedding
        tokens = self.token_proj(waveforms)                      # (B, 10, 128)
        # Add positional embeddings; broadcast over batch via index buffers
        row_pe = self.row_emb(self.row_idx)                      # (10, 128)
        col_pe = self.col_emb(self.col_idx)                      # (10, 128)
        tokens = tokens + row_pe + col_pe

        x = self.encoder(tokens)                                  # (B, 10, 128)
        pooled = x.mean(dim=1)                                    # (B, 128)

        # Normalize centroid using baked-in stats
        cen_norm = (centroid - self.centroid_mean) / self.centroid_std
        cen_feat = self.centroid_mlp(cen_norm)                    # (B, 64)

        feat = torch.cat([pooled, cen_feat], dim=-1)              # (B, 192)
        return self.head(feat)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
