"""
utils/geo.py
=============
Geospatial utility functions used across the pipeline.

Functions
---------
haversine_distance   – great-circle distance (km) between two points
crop_atmospheric_window – extract a lat/lon box from an xarray Dataset
lat_lon_to_tensor    – encode location as a small embedding tensor
"""

from __future__ import annotations

import logging
import math
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
# Constants
# ---------------------------------------------------------------------------
EARTH_RADIUS_KM = 6371.0


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------

def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """
    Compute the great-circle distance between two points using the
    Haversine formula.

    Parameters
    ----------
    lat1, lon1 : float
        Origin coordinates (degrees).
    lat2, lon2 : float
        Destination coordinates (degrees).

    Returns
    -------
    float
        Distance in kilometres.

    Examples
    --------
    >>> haversine_distance(20.0, 88.0, 21.0, 89.0)
    154.26...
    """
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_KM * c


def haversine_batch(
    pred_latlons: torch.Tensor, true_latlons: torch.Tensor
) -> torch.Tensor:
    """
    Vectorised Haversine distance for batched track predictions.

    Parameters
    ----------
    pred_latlons : torch.Tensor
        Shape (batch, 4, 2) — predicted (lat, lon) per horizon.
    true_latlons : torch.Tensor
        Shape (batch, 4, 2) — true (lat, lon) per horizon.

    Returns
    -------
    torch.Tensor
        Shape (batch, 4) — distances in km.
    """
    pred_latlons_r = torch.deg2rad(pred_latlons)
    true_latlons_r = torch.deg2rad(true_latlons)

    dlat = true_latlons_r[..., 0] - pred_latlons_r[..., 0]
    dlon = true_latlons_r[..., 1] - pred_latlons_r[..., 1]

    lat1 = pred_latlons_r[..., 0]
    lat2 = true_latlons_r[..., 0]

    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.asin(torch.clamp(torch.sqrt(a), max=1.0))
    return EARTH_RADIUS_KM * c


# ---------------------------------------------------------------------------
# Atmospheric window cropping
# ---------------------------------------------------------------------------

def crop_atmospheric_window(
    ds,
    center_lat: float,
    center_lon: float,
    window_deg: float = 10.0,
) -> object:
    """
    Crop a square lat/lon window from an xarray Dataset.

    Handles both ascending (0→360 or -180→180) and descending latitude axes.

    Parameters
    ----------
    ds : xr.Dataset
        Atmospheric forecast dataset.
    center_lat : float
        Cyclone centre latitude in degrees.
    center_lon : float
        Cyclone centre longitude in degrees.
    window_deg : float
        Half-width of the crop window in degrees. Default 10°.

    Returns
    -------
    xr.Dataset
        Cropped dataset. May be smaller than requested at domain boundaries.
    """
    if not _XARRAY_AVAILABLE or not hasattr(ds, "sel"):
        logger.warning("xarray not available — returning raw dataset unchanged.")
        return ds

    lat_min = center_lat - window_deg
    lat_max = center_lat + window_deg
    lon_min = center_lon - window_deg
    lon_max = center_lon + window_deg

    # Detect descending latitude axis (GraphCast uses descending)
    lats = ds.lat.values
    if lats[0] > lats[-1]:
        # Descending — swap slice arguments
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)

    lon_slice = slice(lon_min, lon_max)

    try:
        cropped = ds.sel(lat=lat_slice, lon=lon_slice)
    except Exception as exc:
        logger.error("Crop failed: %s. Returning full dataset.", exc)
        return ds

    return cropped


def compute_crop_shape(window_deg: float, grid_resolution: float) -> Tuple[int, int]:
    """
    Compute the expected spatial dimensions of a cropped window.

    Parameters
    ----------
    window_deg : float
        Half-width of window in degrees.
    grid_resolution : float
        Grid spacing in degrees.

    Returns
    -------
    (n_lat, n_lon) : tuple[int, int]
    """
    n = int(round(2 * window_deg / grid_resolution)) + 1
    return n, n


# ---------------------------------------------------------------------------
# Location encoding
# ---------------------------------------------------------------------------

def encode_location(lat: float, lon: float) -> torch.Tensor:
    """
    Encode (lat, lon) as a 4-element sinusoidal tensor for use as metadata
    in the fusion model.

    Returns
    -------
    torch.Tensor
        Shape (4,) — [sin(lat_r), cos(lat_r), sin(lon_r), cos(lon_r)]
    """
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    return torch.tensor(
        [math.sin(lat_r), math.cos(lat_r), math.sin(lon_r), math.cos(lon_r)],
        dtype=torch.float32,
    )


def encode_location_batch(
    lats: torch.Tensor, lons: torch.Tensor
) -> torch.Tensor:
    """
    Vectorised location encoding.

    Parameters
    ----------
    lats, lons : torch.Tensor
        Shape (batch,).

    Returns
    -------
    torch.Tensor
        Shape (batch, 4).
    """
    lats_r = torch.deg2rad(lats)
    lons_r = torch.deg2rad(lons)
    return torch.stack(
        [torch.sin(lats_r), torch.cos(lats_r), torch.sin(lons_r), torch.cos(lons_r)],
        dim=-1,
    )
