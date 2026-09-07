"""
models/atmospheric_encoder.py
================================
Encodes a flat atmospheric feature vector (from a single forecast horizon
crop) into a compact fixed-dimension embedding.

Architecture
------------
    Input: flat_features  (batch, flat_dim)
        ↓
    Linear(flat_dim → hidden_dim) + GELU + Dropout
        ↓
    Linear(hidden_dim → hidden_dim) + GELU + Dropout
        ↓
    Linear(hidden_dim → output_dim)         [embedding]

A shared encoder instance is used for all 4 forecast horizons.
The caller passes each horizon's feature vector independently.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AtmosphericEncoder(nn.Module):
    """
    Small MLP that maps a flat atmospheric crop feature vector to an embedding.

    Parameters
    ----------
    input_dim : int
        Dimension of the flat input feature vector.
        Compute via: AtmosphericFeatureExtractor.feature_dim(spatial_size).
    hidden_dim : int
        Hidden layer width. Default 256.
    output_dim : int
        Embedding dimension. Default 128.
    dropout : float
        Dropout probability. Default 0.2.

    Notes
    -----
    This module is trainable — weights are NOT frozen.
    The satellite classifier and GraphCast runner remain frozen; only this
    encoder, the fusion model, and prediction heads are trained.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self._init_weights()
        logger.debug(
            "AtmosphericEncoder: input=%d → hidden=%d → output=%d",
            input_dim,
            hidden_dim,
            output_dim,
        )

    def _init_weights(self) -> None:
        """Kaiming uniform initialisation for Linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a flat atmospheric feature vector.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Shape (batch, output_dim).
        """
        return self.encoder(x)

    def encode_horizons(
        self, horizon_features: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Convenience method — encode all forecast horizon feature vectors
        using the same shared encoder weights.

        Parameters
        ----------
        horizon_features : list of torch.Tensor
            Each tensor has shape (batch, input_dim).
            Length must be 4 (for +6h/+12h/+18h/+24h).

        Returns
        -------
        list of torch.Tensor
            Each tensor has shape (batch, output_dim).
        """
        return [self.forward(feat) for feat in horizon_features]
