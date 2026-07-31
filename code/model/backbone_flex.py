"""
m/K-independent backbone: per-group embedding + Transformer.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEncoder(nn.Module):
    """Embed the per-sample global features (t/τ, δ, ε)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, hidden_dim), nn.ReLU())

    def forward(self, t_feat: torch.Tensor) -> torch.Tensor:  # (B, 3)
        return self.net(t_feat)                                # (B, D)


class Fusion(nn.Module):
    """Cat [market, pool, time] -> hidden_dim."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GroupEncoder(nn.Module):
    """Encode a single group's (state, valuation, transition) into D-dim."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (..., in_dim)
        return self.net(x)                                 # (..., D)


class FlexibleFairPivotBackbone(nn.Module):
    """
    m/K-independent backbone: per-group encoding + Transformer.

    Inputs:
        state_feat   (B, m, K)    within-group state proportions at round t
        time_feat    (B, 3)       (t/T, δ, ε)
        val_feat     (B, m, K)    valuations / each group's max valuation
        trans_feat   (B, m, 2*K²) transition matrices per group
        mask         (B, m)       True for valid groups, False for padding
        scale_feat   (B, m, 2)    (N_g/N_ref, max(v_g)/V_ref)

    Returns:
        h: (B, m, hidden_dim) shared per-group hidden state
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        m_max: int = 4,
        K_max: int = 4,
        trans_dim: int = None,       # 2 * K_max * K_max; auto-computed
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.m_max = m_max
        self.K_max = K_max
        self.trans_dim = trans_dim or 2 * K_max * K_max   # 2*4*4 = 32

        # Per-group input: relative state + relative valuation + transitions
        # plus two absolute scales (group size and realised max valuation).
        self.scale_dim = 2
        group_in = K_max + K_max + self.trans_dim + self.scale_dim  # 42
        self.group_encoder = GroupEncoder(group_in, hidden_dim)

        # Learnable positional embedding (m_max tokens)
        self.pos_embed = nn.Embedding(m_max, hidden_dim)

        # 2-layer TransformerEncoder over m groups
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True,  # (B, m, D)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Time encoder: global time context injected before inter-group attention
        self.time_encoder = TimeEncoder(hidden_dim)

        # Early fusion: group tokens + global time context -> hidden_dim
        self.fusion = Fusion(hidden_dim + hidden_dim, hidden_dim)

        # Final MLP (no LSTM — point-wise mapping)
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        state_feat: torch.Tensor,    # (B, m, actual_K) — auto-padded to K_max
        time_feat: torch.Tensor,     # (B, 3)
        val_feat: torch.Tensor,      # (B, m, actual_K)
        trans_feat: torch.Tensor,    # (B, m, 2*actual_K²)
        mask: torch.Tensor = None,   # (B, m)  True = valid
        scale_feat: torch.Tensor = None,  # (B, m, 2)
    ) -> torch.Tensor:
        """Return a ``(B, m, hidden_dim)`` hidden representation."""
        if state_feat.ndim != 3:
            raise ValueError(
                "state_feat must have shape (B, m, K); flatten (sample, time) "
                f"before calling the model, got {tuple(state_feat.shape)}"
            )
        B, actual_m, actual_K = state_feat.shape
        if time_feat.shape != (B, 3):
            raise ValueError(
                f"time_feat shape must be {(B, 3)}, got {tuple(time_feat.shape)}"
            )
        if scale_feat is None:
            raise ValueError(
                "scale_feat is required and must contain "
                "[group_size / N_ref, group_max_valuation / V_ref]"
            )
        expected_scale_shape = (B, actual_m, self.scale_dim)
        if tuple(scale_feat.shape) != expected_scale_shape:
            raise ValueError(
                f"scale_feat shape must be {expected_scale_shape}, got {tuple(scale_feat.shape)}"
            )

        # Auto-pad to K_max if actual_K < K_max.
        if actual_K < self.K_max:
            pad_K = self.K_max - actual_K
            state_feat = F.pad(state_feat, (0, pad_K))                # (B, m, K_max)
            val_feat = F.pad(val_feat, (0, pad_K))
            trans_pad = self.trans_dim - trans_feat.shape[-1]
            trans_feat = F.pad(trans_feat, (0, trans_pad))            # (B, m, trans_dim)

        group_in = torch.cat(
            [state_feat, val_feat, trans_feat, scale_feat], dim=-1
        )
        g_emb = self.group_encoder(group_in)                           # (B, m, D)

        positions = torch.arange(actual_m, device=g_emb.device)
        pos = self.pos_embed(positions)                                # (m, D)
        g_emb = g_emb + pos[None, :, :]                                # (B, m, D)

        t_emb = self.time_encoder(time_feat)                           # (B, D)
        t_emb = t_emb.unsqueeze(1).expand(-1, actual_m, -1)           # (B, m, D)
        fused = self.fusion(torch.cat([g_emb, t_emb], dim=-1))        # (B, m, D)

        # Time has already been folded into the batch dimension by the loader.
        # Chunk very large inference batches for CUDA backend compatibility.
        max_batch_rows = 8192
        if B <= max_batch_rows:
            g_trans = self.transformer(fused)                          # (B, m, D)
        else:
            chunks = [
                self.transformer(fused[start:start + max_batch_rows])
                for start in range(0, B, max_batch_rows)
            ]
            g_trans = torch.cat(chunks, dim=0)

        if mask is not None:
            if mask.ndim != 2:
                raise ValueError(f"mask must have shape (B, m), got {tuple(mask.shape)}")
            if mask.shape[-1] > actual_m:
                mask = mask[..., :actual_m]
            g_trans = g_trans * mask.unsqueeze(-1).float()

        return self.final_mlp(g_trans)                                # (B, m, D)


class FlexibleFairPivotNet(nn.Module):
    """
    φ-only network: backbone + single phi head.

    Learns only the group-selection distribution φ ∈ Δᵐ.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        m_max: int = 4,
        K_max: int = 4,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        nhead: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.m_max = m_max

        self.backbone = FlexibleFairPivotBackbone(
            hidden_dim=hidden_dim,
            m_max=m_max,
            K_max=K_max,
            num_layers=2,
            nhead=nhead,
            dropout=lora_dropout,
        )

        # Single head: D → 1 per-group → softmax across groups
        self.phi_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        state_feat: torch.Tensor,    # (B, m, K)
        time_feat: torch.Tensor,     # (B, 3)
        val_feat: torch.Tensor,      # (B, m, K)
        trans_feat: torch.Tensor,    # (B, m, 2*K²)
        mask: torch.Tensor = None,   # (B, m)
        scale_feat: torch.Tensor = None,  # (B, m, 2)
    ) -> dict:
        actual_m = state_feat.shape[1]

        h = self.backbone(
            state_feat, time_feat, val_feat, trans_feat, mask, scale_feat
        )  # (B, m, D)

        phi_logits = self.phi_head(h).squeeze(-1)      # (B, m)

        if mask is not None:
            if mask.shape[-1] > actual_m:
                mask = mask[..., :actual_m]
            phi_logits = phi_logits.masked_fill(~mask, -1e9)

        phi = F.softmax(phi_logits, dim=-1)            # softmax over m groups

        return {
            "phi": phi,       # (B, m) — positive and sums to one
            "h": h,           # (B, m, D), optional debugging representation
        }
