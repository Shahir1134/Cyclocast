"""
models/fusion_model.py
========================
Multimodal Fusion Network that combines satellite and atmospheric embeddings
into a shared representation for track, intensity, and confidence prediction.

Architecture — MVP (Concatenation + MLP)
-----------------------------------------

    Satellite branch:
        satellite_embedding (512) ──► Linear(512 → sat_proj_dim)  ──► GELU
        cyclone_probability  (1) ──┘  (concatenated before projection)

    Atmospheric branch (×4 horizons, shared-weight projection):
        atm_embed_6h  (128) ──► Linear(128 → atm_proj_dim) ──► GELU
        atm_embed_12h (128) ──► Linear(128 → atm_proj_dim) ──► GELU
        atm_embed_18h (128) ──► Linear(128 → atm_proj_dim) ──► GELU
        atm_embed_24h (128) ──► Linear(128 → atm_proj_dim) ──► GELU

    Optional:
        location (4)  ──► Linear(4 → loc_dim) ──► GELU

    Fusion:
        Concatenate all projections
        ──► MLP(shared_hidden_dim × 2 → shared_hidden_dim) ──► GELU ──► Dropout
        ──► Linear(shared_hidden_dim → shared_hidden_dim)
        ──► shared_representation   ← fed into all three heads

Extensibility
--------------
The FusionModel exposes `use_cross_attention` (default False). When True, the
simple concatenation is replaced by a multi-head cross-attention layer between
the satellite and atmospheric tokens. This can be activated without changing
the head architecture.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MultimodalFusionModel(nn.Module):
    """
    Multimodal fusion model (MVP: concatenation + shared MLP).

    Parameters
    ----------
    satellite_embed_dim : int
        Satellite embedding dimension. Default 512 (ResNet18 avgpool).
    atm_embed_dim : int
        Atmospheric embedding dimension per horizon. Default 128.
    sat_proj_dim : int
        Satellite projection dimension. Default 128.
    atm_proj_dim : int
        Atmospheric projection dimension per horizon. Default 128.
    shared_hidden_dim : int
        Shared MLP hidden dimension. Default 256.
    num_horizons : int
        Number of forecast horizons. Default 4.
    use_location_embedding : bool
        Whether to include (lat, lon) sinusoidal encoding. Default True.
    location_embed_dim : int
        Location projection dimension. Default 16.
    dropout : float
        Dropout probability in the shared MLP. Default 0.3.
    use_cross_attention : bool
        Future extension — enable cross-attention fusion. Default False.
    num_attention_heads : int
        Number of attention heads (used when use_cross_attention=True).
    """

    def __init__(
        self,
        satellite_embed_dim: int = 512,
        atm_embed_dim: int = 128,
        sat_proj_dim: int = 128,
        atm_proj_dim: int = 128,
        shared_hidden_dim: int = 256,
        num_horizons: int = 4,
        use_location_embedding: bool = True,
        location_embed_dim: int = 16,
        dropout: float = 0.3,
        use_cross_attention: bool = False,
        num_attention_heads: int = 4,
    ) -> None:
        super().__init__()

        self.num_horizons = num_horizons
        self.use_location_embedding = use_location_embedding
        self.use_cross_attention = use_cross_attention

        # ── Satellite branch ─────────────────────────────────────────────────
        # Input: [satellite_embedding (512) | cyclone_prob (1)] → sat_proj_dim
        sat_input_dim = satellite_embed_dim + 1  # +1 for cyclone_probability
        self.sat_proj = nn.Sequential(
            nn.Linear(sat_input_dim, sat_proj_dim),
            nn.GELU(),
        )

        # ── Atmospheric branch (shared across horizons) ──────────────────────
        self.atm_proj = nn.Sequential(
            nn.Linear(atm_embed_dim, atm_proj_dim),
            nn.GELU(),
        )

        # ── Location branch ──────────────────────────────────────────────────
        if use_location_embedding:
            self.loc_proj = nn.Sequential(
                nn.Linear(4, location_embed_dim),  # 4 = sin/cos for lat & lon
                nn.GELU(),
            )
        else:
            self.loc_proj = None
            location_embed_dim = 0

        # ── Cross-attention (optional, for future upgrade) ───────────────────
        if use_cross_attention:
            token_dim = sat_proj_dim  # all tokens must match
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=token_dim,
                num_heads=num_attention_heads,
                dropout=dropout,
                batch_first=True,
            )

        # ── Compute fusion input dimension ───────────────────────────────────
        if use_cross_attention:
            # After attention over (1 + num_horizons) tokens → pool to 1
            fusion_in = sat_proj_dim
        else:
            fusion_in = (
                sat_proj_dim
                + atm_proj_dim * num_horizons
                + location_embed_dim
            )

        # ── Shared MLP backbone ──────────────────────────────────────────────
        self.shared_mlp = nn.Sequential(
            nn.Linear(fusion_in, shared_hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_hidden_dim * 2, shared_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(shared_hidden_dim, shared_hidden_dim),
        )

        self.output_dim = shared_hidden_dim

        self._init_weights()
        logger.info(
            "MultimodalFusionModel: fusion_in=%d → shared_hidden=%d "
            "(cross_attention=%s)",
            fusion_in,
            shared_hidden_dim,
            use_cross_attention,
        )

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        satellite_embedding: torch.Tensor,
        cyclone_probability: torch.Tensor,
        atm_embeddings: List[torch.Tensor],
        location_encoding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fuse multimodal inputs into a shared representation.

        Parameters
        ----------
        satellite_embedding : torch.Tensor
            Shape (batch, 512) — from SatelliteWrapper.
        cyclone_probability : torch.Tensor
            Shape (batch,) or (batch, 1) — classifier probability.
        atm_embeddings : list of torch.Tensor
            Length 4 (one per horizon). Each: (batch, atm_embed_dim).
        location_encoding : torch.Tensor | None
            Shape (batch, 4) — sinusoidal location encoding from geo.py.
            Required if use_location_embedding=True.

        Returns
        -------
        torch.Tensor
            Shape (batch, shared_hidden_dim) — shared multimodal representation.
        """
        # ── Satellite projection ─────────────────────────────────────────────
        if cyclone_probability.dim() == 1:
            cyclone_probability = cyclone_probability.unsqueeze(-1)  # (batch, 1)

        sat_input = torch.cat([satellite_embedding, cyclone_probability], dim=-1)
        sat_proj = self.sat_proj(sat_input)  # (batch, sat_proj_dim)

        # ── Atmospheric projections ──────────────────────────────────────────
        atm_projs = [self.atm_proj(emb) for emb in atm_embeddings]  # list of (batch, atm_proj_dim)

        if self.use_cross_attention:
            # Stack as tokens: (batch, 1 + num_horizons, sat_proj_dim)
            # Note: atm_proj_dim must equal sat_proj_dim for this to work
            sat_tok = sat_proj.unsqueeze(1)       # (batch, 1, sat_proj_dim)
            atm_tok = torch.stack(atm_projs, dim=1)  # (batch, num_horizons, atm_proj_dim)
            tokens = torch.cat([sat_tok, atm_tok], dim=1)

            attn_out, _ = self.cross_attn(sat_tok, tokens, tokens)
            fused = attn_out.squeeze(1)  # (batch, sat_proj_dim)
        else:
            # Simple concatenation
            parts = [sat_proj] + atm_projs

            if self.use_location_embedding and location_encoding is not None:
                loc_proj = self.loc_proj(location_encoding)  # (batch, loc_embed_dim)
                parts.append(loc_proj)

            fused = torch.cat(parts, dim=-1)

        # ── Shared MLP ───────────────────────────────────────────────────────
        shared_repr = self.shared_mlp(fused)  # (batch, shared_hidden_dim)
        return shared_repr
