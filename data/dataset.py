"""
data/dataset.py
================
Generic PyTorch Dataset for the multimodal fusion training pipeline.

Each sample aligns:
  - A satellite image (INSAT/IR)
  - Pre-computed GraphCast outputs (xarray .nc files or pickled dicts)
  - Current cyclone centre (lat, lon)
  - Ground-truth track for +6h/+12h/+18h/+24h
  - Ground-truth intensity (wind speed, pressure) per horizon

Dataset CSV format
------------------
The ``labels_file`` CSV must have these columns:

    image_path       | path to satellite image (absolute or relative to root_dir)
    gc_6h_path       | path to GraphCast +6h xarray .nc file
    gc_12h_path      | path to GraphCast +12h xarray .nc file
    gc_18h_path      | path to GraphCast +18h xarray .nc file
    gc_24h_path      | path to GraphCast +24h xarray .nc file
    current_lat      | cyclone centre latitude at t=0
    current_lon      | cyclone centre longitude at t=0
    lat_6h           | cyclone centre latitude at +6h
    lon_6h           | cyclone centre longitude at +6h
    lat_12h          | ...
    lon_12h          | ...
    lat_18h          | ...
    lon_18h          | ...
    lat_24h          | ...
    lon_24h          | ...
    wind_6h          | max sustained wind speed (kt) at +6h
    pressure_6h      | min central pressure (hPa) at +6h
    wind_12h...24h   | (same pattern)
    pressure_12h...24h

Note: The dataset is intentionally generic. If you change the GraphCast
output format or label schema, only update the CSV and this loader.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split

from data.preprocessing import get_inference_transform
from utils.feature_extraction import AtmosphericFeatureExtractor
from utils.geo import encode_location

logger = logging.getLogger(__name__)

HORIZONS = ["6h", "12h", "18h", "24h"]


class CycloneFusionDataset(Dataset):
    """
    Multimodal dataset that returns aligned satellite + atmospheric samples.

    Parameters
    ----------
    labels_csv : str | Path
        Path to the CSV labels file (see module docstring for schema).
    root_dir : str | Path
        Root directory for resolving relative paths in the CSV.
    feature_extractor : AtmosphericFeatureExtractor
        Converts atmospheric xarray slices to flat tensors.
    image_transform : callable | None
        Torchvision transform for satellite images. Defaults to inference transform.
    window_deg : float
        Atmospheric crop window half-width in degrees. Default 10.0.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``. For logging only.
    """

    def __init__(
        self,
        labels_csv: str | Path,
        root_dir: str | Path,
        feature_extractor: AtmosphericFeatureExtractor,
        image_transform=None,
        window_deg: float = 10.0,
        split: str = "train",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.window_deg = window_deg
        self.split = split
        self.feature_extractor = feature_extractor
        self.image_transform = image_transform or get_inference_transform()

        csv_path = Path(labels_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"Labels CSV not found: {csv_path}")
        self.df = pd.read_csv(csv_path).reset_index(drop=True)

        self._validate_columns()
        logger.info("CycloneFusionDataset [%s]: %d samples", split, len(self.df))

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def _validate_columns(self) -> None:
        required = [
            "image_path",
            "current_lat", "current_lon",
        ] + [
            f"{col}_{h}" for h in HORIZONS
            for col in ["gc_path", "lat", "lon", "wind", "pressure"]
        ]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            logger.warning(
                "Labels CSV missing columns: %s. "
                "These will be filled with zeros during training.",
                missing,
            )

    # -----------------------------------------------------------------------
    # Dataset interface
    # -----------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # ── Satellite image ──────────────────────────────────────────────────
        img_path = self._resolve_path(row["image_path"])
        image = Image.open(img_path).convert("RGB")
        satellite_image = self.image_transform(image)  # (3, 224, 224)

        # ── Current location ─────────────────────────────────────────────────
        current_lat = float(row.get("current_lat", 0.0))
        current_lon = float(row.get("current_lon", 0.0))
        location_encoding = encode_location(current_lat, current_lon)  # (4,)

        # ── Atmospheric features per horizon ─────────────────────────────────
        atm_features = {}
        for h in HORIZONS:
            col = f"gc_path_{h}"
            gc_path = self._resolve_path(row.get(col, ""))
            crop = self._load_atmospheric_crop(gc_path, current_lat, current_lon)
            try:
                atm_features[h] = self.feature_extractor.extract_tensor(crop)
            except Exception as exc:
                logger.warning("Feature extraction failed for %s at %s: %s", h, gc_path, exc)
                # Fallback: zero tensor
                atm_features[h] = torch.zeros(self._get_fallback_feat_dim())

        # ── Track targets ────────────────────────────────────────────────────
        # Store as deltas from current position for training stability
        track_targets = []
        for h in HORIZONS:
            lat_h = float(row.get(f"lat_{h}", current_lat))
            lon_h = float(row.get(f"lon_{h}", current_lon))
            delta_lat = lat_h - current_lat
            delta_lon = lon_h - current_lon
            track_targets.append([delta_lat, delta_lon])
        target_track = torch.tensor(track_targets, dtype=torch.float32)  # (4, 2)

        # ── Intensity targets ────────────────────────────────────────────────
        from models.intensity_head import IntensityHead
        intensity_targets = []
        for h in HORIZONS:
            wind = float(row.get(f"wind_{h}", 0.0))
            pressure = float(row.get(f"pressure_{h}", 1013.0))
            norm_wind, norm_pres = IntensityHead.normalise_targets(wind, pressure)
            intensity_targets.append([norm_wind, norm_pres])
        target_intensity = torch.tensor(intensity_targets, dtype=torch.float32)  # (4, 2)

        return {
            "satellite_image":   satellite_image,
            "current_lat":       torch.tensor(current_lat, dtype=torch.float32),
            "current_lon":       torch.tensor(current_lon, dtype=torch.float32),
            "location_encoding": location_encoding,
            "atm_6h":            atm_features["6h"],
            "atm_12h":           atm_features["12h"],
            "atm_18h":           atm_features["18h"],
            "atm_24h":           atm_features["24h"],
            "target_track":      target_track,
            "target_intensity":  target_intensity,
        }

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _resolve_path(self, path_str: str) -> Path:
        p = Path(str(path_str))
        if not p.is_absolute():
            p = self.root_dir / p
        return p

    def _load_atmospheric_crop(self, gc_path: Path, lat: float, lon: float):
        """Load a GraphCast slice from disk and crop around cyclone centre."""
        from utils.geo import crop_atmospheric_window

        if not gc_path.exists():
            logger.debug("GraphCast file missing: %s — using zeros.", gc_path)
            return {}

        try:
            import xarray as xr
            ds = xr.open_dataset(str(gc_path))
            crop = crop_atmospheric_window(ds, lat, lon, self.window_deg)
            return crop
        except Exception as exc:
            logger.warning("Failed to load %s: %s", gc_path, exc)
            return {}

    def _get_fallback_feat_dim(self) -> int:
        """Return the expected flat feature dim (used for fallback zero tensors)."""
        try:
            return self.feature_extractor.feature_dim(
                self.feature_extractor.spatial_size or 81
            )
        except Exception:
            return 4 * 81 * 81  # 4 surface channels × 81 × 81


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    labels_csv: str | Path,
    root_dir: str | Path,
    feature_extractor: AtmosphericFeatureExtractor,
    batch_size: int = 16,
    num_workers: int = 2,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test DataLoaders with random splits.

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    full_dataset = CycloneFusionDataset(
        labels_csv=labels_csv,
        root_dir=root_dir,
        feature_extractor=feature_extractor,
    )

    n = len(full_dataset)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    n_test  = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    # Apply augmented transform to training set
    from data.preprocessing import get_train_transform
    train_ds.dataset.image_transform = get_train_transform()

    def make_loader(ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=shuffle,
        )

    return (
        make_loader(train_ds, shuffle=True),
        make_loader(val_ds,   shuffle=False),
        make_loader(test_ds,  shuffle=False),
    )
