"""
models/track_head.py
======================
Prediction head for cyclone track (centre coordinates).

Output
------
    (batch, 4, 2)  →  4 horizons × (Δlat, Δlon)

The head predicts *deltas* from the current cyclone centre rather than
absolute coordinates. This makes training faster and predictions more stable.
The inference layer converts deltas back to absolute lat/lon.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

NUM_HORIZONS = 4  # +6h, +12h, +18h, +24h
COORDS_PER_HORIZON = 2  # (lat, lon)


class TrackHead(nn.Module):
    """
    MLP head that predicts cyclone track displacements.

    Parameters
    ----------
    input_dim : int
        Dimension of the shared multimodal representation from FusionModel.
    hidden_dim : int
        Hidden layer width. Default 128.
    num_horizons : int
        Number of forecast horizons. Default 4.

    Output convention
    -----------------
    Returns delta coordinates (Δlat, Δlon) relative to the current
    cyclone centre. Convert to absolute with:
        pred_lat = current_lat + delta_lat
        pred_lon = current_lon + delta_lon
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_horizons: int = NUM_HORIZONS,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_horizons = num_horizons
        self.output_dim = num_horizons * COORDS_PER_HORIZON

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, self.output_dim),
        )

        self._init_weights()
        logger.debug("TrackHead: input_dim=%d, output=(batch, %d, 2)", input_dim, num_horizons)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, shared_repr: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        shared_repr : torch.Tensor
            Shape (batch, input_dim) — output of FusionModel.

        Returns
        -------
        torch.Tensor
            Shape (batch, num_horizons, 2) — (Δlat, Δlon) per horizon.
        """
        out = self.head(shared_repr)  # (batch, 4 × 2)
        return out.view(-1, self.num_horizons, COORDS_PER_HORIZON)
