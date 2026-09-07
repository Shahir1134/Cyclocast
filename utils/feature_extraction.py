"""
utils/feature_extraction.py
==============================
Convert cropped xarray atmospheric slices into flat PyTorch tensors
suitable for the AtmosphericEncoder.

The extractor:
  1. Selects a configurable set of variables from the crop.
  2. For 3-D variables (lat, lon, pressure_level), selects specific levels.
  3. Normalises each variable to zero-mean / unit-variance using
     pre-computed climatological statistics (or per-batch normalisation
     as a fallback).
  4. Flattens and concatenates all fields into a 1-D feature vector.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    import xarray as xr
    _XARRAY_AVAILABLE = True
except ImportError:
    _XARRAY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Per-variable rough climatological statistics (mean, std)
# Used for normalisation when pre-computed stats are not provided.
# Sources: ERA5 global annual climatology approximations.
# ---------------------------------------------------------------------------
VARIABLE_CLIMO: Dict[str, Tuple[float, float]] = {
    "mean_sea_level_pressure": (101325.0, 1500.0),    # Pa
    "10m_u_component_of_wind": (0.0, 7.0),            # m/s
    "10m_v_component_of_wind": (0.0, 7.0),            # m/s
    "2m_temperature":          (288.0, 15.0),         # K
    "specific_humidity":       (0.005, 0.008),        # kg/kg
    "geopotential":            (50000.0, 30000.0),    # m²/s²
    "temperature":             (260.0, 30.0),         # K
    "u_component_of_wind":     (0.0, 15.0),           # m/s
    "v_component_of_wind":     (0.0, 15.0),           # m/s
}

# Pressure levels to use for 3-D variables (must match config)
DEFAULT_PRESSURE_LEVELS: List[int] = [850, 500, 250]

# Surface-only variables (no pressure dimension)
SURFACE_VARIABLES: List[str] = [
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
]

# Pressure-level variables
PRESSURE_VARIABLES: List[str] = [
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
    "specific_humidity",
]


class AtmosphericFeatureExtractor:
    """
    Converts a cropped xarray forecast slice into a flat 1-D numpy array
    (or torch.Tensor) ready for the AtmosphericEncoder.

    Parameters
    ----------
    surface_vars : list[str]
        Surface-level variables to include.
    pressure_vars : list[str]
        Pressure-level variables to include.
    pressure_levels : list[int]
        Pressure levels (hPa) to extract for pressure variables.
    norm_stats : dict | None
        Optional per-variable (mean, std) override. Falls back to VARIABLE_CLIMO.
    spatial_size : int | None
        If set, crops/pads the spatial dimensions to this size × size.
        If None, uses whatever size comes from the crop window.
    """

    def __init__(
        self,
        surface_vars: Optional[List[str]] = None,
        pressure_vars: Optional[List[str]] = None,
        pressure_levels: Optional[List[int]] = None,
        norm_stats: Optional[Dict[str, Tuple[float, float]]] = None,
        spatial_size: Optional[int] = None,
    ) -> None:
        self.surface_vars = surface_vars or SURFACE_VARIABLES
        self.pressure_vars = pressure_vars or PRESSURE_VARIABLES
        self.pressure_levels = pressure_levels or DEFAULT_PRESSURE_LEVELS
        self.norm_stats = norm_stats or VARIABLE_CLIMO
        self.spatial_size = spatial_size

        # Compute expected flat feature dimension
        n_surf = len(self.surface_vars)
        n_pres = len(self.pressure_vars) * len(self.pressure_levels)
        spatial = (spatial_size * spatial_size) if spatial_size else None
        self._n_channels = n_surf + n_pres
        self.spatial_size_computed = spatial
        logger.debug(
            "AtmosphericFeatureExtractor: %d channels, spatial=%s",
            self._n_channels,
            spatial,
        )

    # -----------------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------------

    def extract(self, crop_slice) -> np.ndarray:
        """
        Extract and normalise features from a single forecast horizon slice.

        Parameters
        ----------
        crop_slice : xr.Dataset | dict
            Cropped atmospheric dataset for one horizon.

        Returns
        -------
        np.ndarray
            Flat 1-D array of shape (n_channels × spatial_h × spatial_w,).
        """
        channels = []

        for var in self.surface_vars:
            arr = self._get_variable(crop_slice, var, level=None)
            if arr is not None:
                channels.append(self._normalise(arr, var))

        for var in self.pressure_vars:
            for level in self.pressure_levels:
                arr = self._get_variable(crop_slice, var, level=level)
                if arr is not None:
                    channels.append(self._normalise(arr, var))

        if not channels:
            raise ValueError("No valid atmospheric channels could be extracted.")

        # Stack → (C, H, W), optionally resize, then flatten
        stacked = np.stack(channels, axis=0)  # (C, H, W)
        if self.spatial_size is not None:
            stacked = self._resize_spatial(stacked, self.spatial_size)

        return stacked.flatten().astype(np.float32)

    def extract_tensor(self, crop_slice) -> torch.Tensor:
        """Extract and return as torch.Tensor."""
        arr = self.extract(crop_slice)
        return torch.from_numpy(arr)

    def feature_dim(self, spatial_h: int, spatial_w: Optional[int] = None) -> int:
        """Compute the flat feature dimension given spatial grid size."""
        w = spatial_w or spatial_h
        return self._n_channels * spatial_h * w

    # -----------------------------------------------------------------------
    # Private
    # -----------------------------------------------------------------------

    def _get_variable(self, ds, var: str, level: Optional[int]) -> Optional[np.ndarray]:
        """Extract a single variable (optionally at a pressure level)."""
        try:
            if _XARRAY_AVAILABLE and hasattr(ds, "data_vars"):
                if var not in ds.data_vars:
                    logger.debug("Variable '%s' not in crop slice — skipping.", var)
                    return None
                da = ds[var]
                if level is not None and "level" in da.dims:
                    da = da.sel(level=level, method="nearest")
                return da.values.astype(np.float32)
            elif isinstance(ds, dict):
                if var not in ds:
                    return None
                arr = np.array(ds[var], dtype=np.float32)
                # For pressure-level variables in dict format: shape (level, lat, lon)
                if level is not None and arr.ndim == 3:
                    level_idx = 0  # fallback
                    arr = arr[level_idx]
                return arr
            else:
                return None
        except Exception as exc:
            logger.warning("Failed to extract variable '%s': %s", var, exc)
            return None

    def _normalise(self, arr: np.ndarray, var: str) -> np.ndarray:
        """Normalise array using climatological statistics."""
        mean, std = self.norm_stats.get(var, (0.0, 1.0))
        std = std if std > 0 else 1.0
        # Handle NaN/Inf from GraphCast
        arr = np.nan_to_num(arr, nan=mean, posinf=mean + 5 * std, neginf=mean - 5 * std)
        return (arr - mean) / std

    def _resize_spatial(self, arr: np.ndarray, size: int) -> np.ndarray:
        """
        Crop or pad the spatial dimensions (H, W) to ``size × size``.
        arr shape: (C, H, W).
        """
        _, h, w = arr.shape
        # Crop if larger
        h_start = max(0, (h - size) // 2)
        w_start = max(0, (w - size) // 2)
        arr = arr[:, h_start:h_start + size, w_start:w_start + size]

        # Pad if smaller
        _, h2, w2 = arr.shape
        pad_h = max(0, size - h2)
        pad_w = max(0, size - w2)
        if pad_h > 0 or pad_w > 0:
            arr = np.pad(
                arr,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=0,
            )
        return arr
