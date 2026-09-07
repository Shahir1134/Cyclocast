"""
models/confidence_head.py
===========================
Prediction head for the overall multimodal forecast confidence score.

Output
------
    (batch, 1)  →  sigmoid → [0.0, 1.0]

This score represents the model's self-assessed confidence in the
combined track + intensity forecast. It is distinct from the satellite
classifier's cyclone probability.

Self-supervised training proxy
---------------------------------
During training, the confidence target is derived from the model's own
track error: samples with low track error get confidence → 1.0, and
samples with high track error get confidence → 0.0.

    confidence_target = exp(−track_error_km / sigma)

where sigma ≈ 200 km (configurable). This avoids needing manual confidence
labels while ensuring the score is meaningful.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

DEFAULT_ERROR_SIGMA_KM = 200.0  # controls sharpness of the proxy


class ConfidenceHead(nn.Module):
    """
    MLP head that outputs a scalar forecast confidence in [0, 1].

    Parameters
    ----------
    input_dim : int
        Dimension of the shared multimodal representation.
    hidden_dim : int
        Hidden layer width. Default 64.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self._init_weights()
        logger.debug("ConfidenceHead: input_dim=%d → scalar [0, 1]", input_dim)

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
            Shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Shape (batch, 1) — forecast confidence in [0, 1].
        """
        return self.head(shared_repr)

    # -----------------------------------------------------------------------
    # Training utilities
    # -----------------------------------------------------------------------

    @staticmethod
    def compute_confidence_target(
        track_errors_km: torch.Tensor,
        sigma_km: float = DEFAULT_ERROR_SIGMA_KM,
    ) -> torch.Tensor:
        """
        Compute a self-supervised confidence target from track prediction errors.

        Parameters
        ----------
        track_errors_km : torch.Tensor
            Mean track error per sample. Shape (batch,).
        sigma_km : float
            Controls sharpness: small sigma → rapid confidence decay.

        Returns
        -------
        torch.Tensor
            Shape (batch, 1) — target confidence in [0, 1].
        """
        targets = torch.exp(-track_errors_km / sigma_km)
        return targets.unsqueeze(-1).clamp(0.0, 1.0)
