"""
models/intensity_head.py
==========================
Prediction head for cyclone intensity.

Output
------
    (batch, 4, 2)  →  4 horizons × (wind_speed_kt, min_pressure_hPa)

Both targets are regression values normalised during training.
The inference layer denormalises using configurable statistics.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

NUM_HORIZONS = 4
INTENSITY_FEATURES = 2  # (wind_speed, pressure)

# Normalisation constants (approximate typical tropical cyclone ranges)
WIND_MEAN   = 60.0    # knots
WIND_STD    = 40.0    # knots
PRES_MEAN   = 990.0   # hPa
PRES_STD    = 30.0    # hPa


class IntensityHead(nn.Module):
    """
    MLP head that predicts future cyclone intensity.

    Parameters
    ----------
    input_dim : int
        Dimension of the shared multimodal representation.
    hidden_dim : int
        Hidden layer width. Default 128.
    num_horizons : int
        Number of forecast horizons. Default 4.
    denormalise_output : bool
        If True, apply denormalisation to return physical units.
        Set False during training (use normalised targets). Default False.

    Outputs
    -------
    (batch, num_horizons, 2):
        - ``[:, :, 0]`` — maximum sustained wind speed (knots)
        - ``[:, :, 1]`` — minimum central pressure (hPa)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_horizons: int = NUM_HORIZONS,
        denormalise_output: bool = False,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_horizons = num_horizons
        self.denormalise_output = denormalise_output
        self.output_dim = num_horizons * INTENSITY_FEATURES

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, self.output_dim),
        )

        self._init_weights()
        logger.debug(
            "IntensityHead: input_dim=%d, output=(batch, %d, 2)", input_dim, num_horizons
        )

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
            Shape (batch, num_horizons, 2) — (wind_kt, pressure_hPa).
            Values are normalised unless ``denormalise_output=True``.
        """
        out = self.head(shared_repr)  # (batch, 4 × 2)
        out = out.view(-1, self.num_horizons, INTENSITY_FEATURES)

        if self.denormalise_output:
            out = self.denormalise(out)

        return out

    @staticmethod
    def denormalise(x: torch.Tensor) -> torch.Tensor:
        """
        Convert normalised predictions back to physical units.

        Parameters
        ----------
        x : torch.Tensor
            Shape (..., 2) — normalised (wind, pressure).

        Returns
        -------
        torch.Tensor
            Shape (..., 2) — (knots, hPa).
        """
        wind = x[..., 0] * WIND_STD + WIND_MEAN
        pres = x[..., 1] * PRES_STD + PRES_MEAN
        return torch.stack([wind, pres], dim=-1)

    @staticmethod
    def normalise_targets(wind_kt: float, pressure_hPa: float):
        """Normalise physical targets for use as training labels."""
        return (wind_kt - WIND_MEAN) / WIND_STD, (pressure_hPa - PRES_MEAN) / PRES_STD
