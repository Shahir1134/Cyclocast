"""
training/losses.py
====================
Loss functions for the multimodal fusion training pipeline.

Three task losses are combined into a single weighted total:

    L_total = w_track × L_track
            + w_intensity × L_intensity
            + w_confidence × L_confidence

Loss design
-----------
L_track      : HuberLoss on (Δlat, Δlon) per horizon.
               Robust to the occasional large-error sample during early training.
L_intensity  : HuberLoss on normalised (wind, pressure) per horizon.
L_confidence : BCELoss between predicted confidence and self-supervised
               target derived from track error magnitude.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.geo import haversine_batch
from models.confidence_head import ConfidenceHead

logger = logging.getLogger(__name__)


class TrackLoss(nn.Module):
    """
    Huber loss on predicted vs. true track deltas (Δlat, Δlon).

    Parameters
    ----------
    delta : float
        Huber delta threshold. Default 1.0 degree.
    """

    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        self.loss_fn = nn.HuberLoss(delta=delta, reduction="mean")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : torch.Tensor
            Shape (batch, 4, 2) — predicted (Δlat, Δlon).
        target : torch.Tensor
            Shape (batch, 4, 2) — true (Δlat, Δlon).

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        return self.loss_fn(pred, target)


class IntensityLoss(nn.Module):
    """
    Huber loss on predicted vs. true normalised intensity (wind, pressure).

    Parameters
    ----------
    delta : float
        Huber delta. Default 1.0 (normalised units).
    wind_weight : float
        Relative weight for wind speed error. Default 1.0.
    pressure_weight : float
        Relative weight for pressure error. Default 1.0.
    """

    def __init__(
        self,
        delta: float = 1.0,
        wind_weight: float = 1.0,
        pressure_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.loss_fn = nn.HuberLoss(delta=delta, reduction="mean")
        self.wind_weight = wind_weight
        self.pressure_weight = pressure_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : torch.Tensor
            Shape (batch, 4, 2) — predicted (wind, pressure) normalised.
        target : torch.Tensor
            Shape (batch, 4, 2) — true (wind, pressure) normalised.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        wind_loss = self.loss_fn(pred[..., 0], target[..., 0])
        pres_loss = self.loss_fn(pred[..., 1], target[..., 1])
        return self.wind_weight * wind_loss + self.pressure_weight * pres_loss


class ConfidenceLoss(nn.Module):
    """
    BCELoss between predicted confidence and self-supervised target.

    The target is computed from the current-batch track error:
        target = exp(−mean_track_error_km / sigma)

    Parameters
    ----------
    sigma_km : float
        Error sigma for confidence target computation. Default 200 km.
    """

    def __init__(self, sigma_km: float = 200.0) -> None:
        super().__init__()
        self.sigma_km = sigma_km
        self.loss_fn = nn.BCELoss(reduction="mean")

    def forward(
        self,
        pred_confidence: torch.Tensor,
        pred_track: torch.Tensor,
        true_track: torch.Tensor,
        current_lat: torch.Tensor,
        current_lon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred_confidence : torch.Tensor
            Shape (batch, 1) — model's confidence output.
        pred_track : torch.Tensor
            Shape (batch, 4, 2) — predicted Δlat/Δlon.
        true_track : torch.Tensor
            Shape (batch, 4, 2) — true Δlat/Δlon.
        current_lat : torch.Tensor
            Shape (batch,) — cyclone centre latitude.
        current_lon : torch.Tensor
            Shape (batch,) — cyclone centre longitude.

        Returns
        -------
        torch.Tensor
            Scalar BCELoss.
        """
        # Convert deltas to absolute coords for haversine calculation
        pred_abs = torch.stack([
            current_lat.unsqueeze(-1) + pred_track[..., 0],
            current_lon.unsqueeze(-1) + pred_track[..., 1],
        ], dim=-1)  # (batch, 4, 2)

        true_abs = torch.stack([
            current_lat.unsqueeze(-1) + true_track[..., 0],
            current_lon.unsqueeze(-1) + true_track[..., 1],
        ], dim=-1)

        track_errors_km = haversine_batch(pred_abs.detach(), true_abs)  # (batch, 4)
        mean_errors = track_errors_km.mean(dim=-1)  # (batch,)

        conf_targets = ConfidenceHead.compute_confidence_target(
            mean_errors, self.sigma_km
        )  # (batch, 1)

        return self.loss_fn(pred_confidence, conf_targets.to(pred_confidence.device))


class FusionLoss(nn.Module):
    """
    Combined weighted loss for the fusion pipeline.

    Parameters
    ----------
    track_weight : float
        Weight for track loss. Default 1.0.
    intensity_weight : float
        Weight for intensity loss. Default 0.5.
    confidence_weight : float
        Weight for confidence loss. Default 0.2.
    track_delta : float
        Huber delta for TrackLoss.
    intensity_delta : float
        Huber delta for IntensityLoss.
    sigma_km : float
        Confidence target sigma.
    """

    def __init__(
        self,
        track_weight: float = 1.0,
        intensity_weight: float = 0.5,
        confidence_weight: float = 0.2,
        track_delta: float = 1.0,
        intensity_delta: float = 1.0,
        sigma_km: float = 200.0,
    ) -> None:
        super().__init__()
        self.track_weight = track_weight
        self.intensity_weight = intensity_weight
        self.confidence_weight = confidence_weight

        self.track_loss_fn = TrackLoss(delta=track_delta)
        self.intensity_loss_fn = IntensityLoss(delta=intensity_delta)
        self.confidence_loss_fn = ConfidenceLoss(sigma_km=sigma_km)

    def forward(
        self,
        pred_track: torch.Tensor,
        pred_intensity: torch.Tensor,
        pred_confidence: torch.Tensor,
        target_track: torch.Tensor,
        target_intensity: torch.Tensor,
        current_lat: torch.Tensor,
        current_lon: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all component losses and return both components and total.

        Returns
        -------
        dict with keys:
            ``loss_track``, ``loss_intensity``, ``loss_confidence``, ``loss_total``
        """
        l_track = self.track_loss_fn(pred_track, target_track)
        l_intensity = self.intensity_loss_fn(pred_intensity, target_intensity)
        l_confidence = self.confidence_loss_fn(
            pred_confidence, pred_track, target_track, current_lat, current_lon
        )

        l_total = (
            self.track_weight * l_track
            + self.intensity_weight * l_intensity
            + self.confidence_weight * l_confidence
        )

        return {
            "loss_track":      l_track,
            "loss_intensity":  l_intensity,
            "loss_confidence": l_confidence,
            "loss_total":      l_total,
        }
