"""
training/metrics.py
=====================
Evaluation metrics for the multimodal fusion pipeline.

Track metric  : Mean track error in km (great-circle distance) per horizon
                and aggregated across all horizons.
Intensity     : MAE on wind speed (knots) and minimum pressure (hPa).
Confidence    : Mean confidence score and calibration proxy.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch

from utils.geo import haversine_batch
from models.intensity_head import WIND_MEAN, WIND_STD, PRES_MEAN, PRES_STD

logger = logging.getLogger(__name__)

HORIZON_LABELS = ["6h", "12h", "18h", "24h"]


# ---------------------------------------------------------------------------
# Track metrics
# ---------------------------------------------------------------------------

def track_error_km(
    pred_track: torch.Tensor,
    true_track: torch.Tensor,
    current_lat: torch.Tensor,
    current_lon: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute mean track error (km) per forecast horizon.

    Parameters
    ----------
    pred_track : torch.Tensor
        Shape (batch, 4, 2) — predicted (Δlat, Δlon).
    true_track : torch.Tensor
        Shape (batch, 4, 2) — true (Δlat, Δlon).
    current_lat : torch.Tensor
        Shape (batch,) — cyclone centre latitude.
    current_lon : torch.Tensor
        Shape (batch,) — cyclone centre longitude.

    Returns
    -------
    dict
        Keys: ``"6h"``, ``"12h"``, ``"18h"``, ``"24h"``, ``"mean"``.
        Values: mean track error in km.
    """
    # Convert deltas to absolute coordinates
    pred_abs = torch.stack([
        current_lat.unsqueeze(-1) + pred_track[..., 0],
        current_lon.unsqueeze(-1) + pred_track[..., 1],
    ], dim=-1)  # (batch, 4, 2)

    true_abs = torch.stack([
        current_lat.unsqueeze(-1) + true_track[..., 0],
        current_lon.unsqueeze(-1) + true_track[..., 1],
    ], dim=-1)

    errors = haversine_batch(pred_abs.detach().cpu(), true_abs.cpu())  # (batch, 4)

    result: Dict[str, float] = {}
    for i, label in enumerate(HORIZON_LABELS):
        result[label] = errors[:, i].mean().item()
    result["mean"] = errors.mean().item()
    return result


# ---------------------------------------------------------------------------
# Intensity metrics
# ---------------------------------------------------------------------------

def intensity_mae(
    pred_intensity: torch.Tensor,
    true_intensity: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute MAE for wind speed (kt) and pressure (hPa).

    Parameters
    ----------
    pred_intensity : torch.Tensor
        Shape (batch, 4, 2) — normalised predicted intensity.
    true_intensity : torch.Tensor
        Shape (batch, 4, 2) — normalised true intensity.

    Returns
    -------
    dict
        Keys: ``"wind_mae_<h>"``, ``"pressure_mae_<h>"``, ``"wind_mae_mean"``,
              ``"pressure_mae_mean"``.
        Values in physical units (kt, hPa).
    """
    pred_d = pred_intensity.detach().cpu()
    true_d = true_intensity.cpu()

    # Denormalise
    pred_wind = pred_d[..., 0] * WIND_STD + WIND_MEAN
    pred_pres = pred_d[..., 1] * PRES_STD + PRES_MEAN
    true_wind = true_d[..., 0] * WIND_STD + WIND_MEAN
    true_pres = true_d[..., 1] * PRES_STD + PRES_MEAN

    wind_abs_err = (pred_wind - true_wind).abs()  # (batch, 4)
    pres_abs_err = (pred_pres - true_pres).abs()

    result: Dict[str, float] = {}
    for i, label in enumerate(HORIZON_LABELS):
        result[f"wind_mae_{label}"] = wind_abs_err[:, i].mean().item()
        result[f"pressure_mae_{label}"] = pres_abs_err[:, i].mean().item()

    result["wind_mae_mean"] = wind_abs_err.mean().item()
    result["pressure_mae_mean"] = pres_abs_err.mean().item()
    return result


# ---------------------------------------------------------------------------
# Confidence metrics
# ---------------------------------------------------------------------------

def confidence_stats(pred_confidence: torch.Tensor) -> Dict[str, float]:
    """
    Summarise confidence score distribution over a batch.

    Returns
    -------
    dict
        Keys: ``"mean"``, ``"std"``, ``"min"``, ``"max"``.
    """
    c = pred_confidence.detach().cpu().squeeze(-1)
    return {
        "mean": c.mean().item(),
        "std":  c.std().item(),
        "min":  c.min().item(),
        "max":  c.max().item(),
    }


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------

class MetricsAccumulator:
    """
    Running accumulator for computing epoch-level metrics from batches.

    Usage
    -----
    .. code-block:: python

        acc = MetricsAccumulator()
        for batch in dataloader:
            ...
            acc.update(pred_track, pred_intensity, pred_conf,
                       true_track, true_intensity, current_lat, current_lon)
        epoch_metrics = acc.compute()
        acc.reset()
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._track_errors: List[Dict[str, float]] = []
        self._intensity_maes: List[Dict[str, float]] = []
        self._conf_scores: List[float] = []
        self._total_loss: float = 0.0
        self._n_batches: int = 0

    def update(
        self,
        pred_track: torch.Tensor,
        pred_intensity: torch.Tensor,
        pred_confidence: torch.Tensor,
        true_track: torch.Tensor,
        true_intensity: torch.Tensor,
        current_lat: torch.Tensor,
        current_lon: torch.Tensor,
        batch_loss: Optional[float] = None,
    ) -> None:
        self._track_errors.append(
            track_error_km(pred_track, true_track, current_lat, current_lon)
        )
        self._intensity_maes.append(
            intensity_mae(pred_intensity, true_intensity)
        )
        self._conf_scores.append(
            confidence_stats(pred_confidence)["mean"]
        )
        if batch_loss is not None:
            self._total_loss += batch_loss
        self._n_batches += 1

    def compute(self) -> Dict[str, float]:
        """Return averaged metrics across all accumulated batches."""
        result: Dict[str, float] = {}

        # Track
        for key in self._track_errors[0]:
            result[f"track_error_{key}"] = float(
                np.mean([d[key] for d in self._track_errors])
            )

        # Intensity
        for key in self._intensity_maes[0]:
            result[key] = float(
                np.mean([d[key] for d in self._intensity_maes])
            )

        # Confidence
        result["mean_confidence"] = float(np.mean(self._conf_scores))

        # Loss
        if self._n_batches > 0:
            result["mean_loss"] = self._total_loss / self._n_batches

        return result
