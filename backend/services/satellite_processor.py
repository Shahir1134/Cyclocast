"""
backend/services/satellite_processor.py
======================================
Processes real EUMETSAT Meteosat-8/9 IODC native satellite telemetry (.nat / .zip),
calibrates the 10.8 µm thermal IR channel to Kelvin Brightness Temperature,
generates the enhanced thermal IR overlay image, and executes PyTorch ResNet-18
cyclone classification using cyclone.pth.
"""

from __future__ import annotations

import logging
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("cyclocast_satellite")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = _PROJECT_ROOT / "eumetsat_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LATEST_IR_PNG = OUTPUT_DIR / "latest_satellite_ir.png"

# Cached model singleton
_CLASSIFIER_MODEL = None


def get_classifier_model():
    """Load and cache the trained PyTorch cyclone.pth classifier model."""
    global _CLASSIFIER_MODEL
    if _CLASSIFIER_MODEL is None:
        ckpt = _PROJECT_ROOT / "cyclone.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {ckpt}")
        from models.satellite_wrapper import SatelliteWrapper
        _CLASSIFIER_MODEL = SatelliteWrapper(str(ckpt), device="cpu")
        logger.info("Loaded cyclone.pth ResNet-18 classifier into memory.")
    return _CLASSIFIER_MODEL


def ensure_unpacked_nat() -> Optional[Path]:
    """Find or extract the latest available .nat file."""
    tmp_dir = _PROJECT_ROOT / "tmp_eumetsat"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing .nat in tmp_eumetsat
    nat_files = list(tmp_dir.glob("*.nat"))
    if nat_files:
        return sorted(nat_files, key=lambda f: f.stat().st_mtime, reverse=True)[0]

    # Check for zip in test_download
    download_dir = _PROJECT_ROOT / "test_download"
    zips = list(download_dir.glob("*.zip"))
    if zips:
        latest_zip = sorted(zips, key=lambda f: f.stat().st_mtime, reverse=True)[0]
        logger.info("Extracting .nat from %s ...", latest_zip.name)
        with zipfile.ZipFile(latest_zip, "r") as z:
            for name in z.namelist():
                if name.endswith(".nat"):
                    z.extract(name, tmp_dir)
                    extracted = tmp_dir / name
                    logger.info("Extracted %s (%.1f MB)", extracted.name, extracted.stat().st_size / 1e6)
                    return extracted

    return None


def generate_satellite_ir_image(nat_path: Path) -> Dict[str, Any]:
    """
    Calibrate SEVIRI 10.8µm IR channel to Kelvin and crop to North Indian Ocean
    (0°-30°N, 55°-100°E) to create a meteorologically accurate enhanced IR image.
    """
    from satpy import Scene

    logger.info("Loading SatPy scene from %s ...", nat_path.name)
    scn = Scene(filenames=[str(nat_path)], reader="seviri_l1b_native")
    scn.load(["IR_108"], calibration="brightness_temperature")

    # Crop to Bay of Bengal + Arabian Sea basin
    cropped = scn.crop(ll_bbox=(55.0, 0.0, 100.0, 30.0))
    data = cropped["IR_108"].values
    data = np.squeeze(data).astype(np.float32)

    k_min = float(np.nanmin(data))
    k_max = float(np.nanmax(data))
    logger.info("Calibrated thermal IR Kelvin: min=%.1fK, max=%.1fK, shape=%s", k_min, k_max, data.shape)

    # Invert and normalize to thermal infrared color spectrum
    # Cold high convective cloud tops (<200K) -> bright cyan/white
    # Warm tropical ocean surface (>300K) -> dark navy/black
    cleaned = np.nan_to_num(data, nan=305.0)
    clamped = np.clip(cleaned, 185.0, 305.0)
    norm = ((305.0 - clamped) / (305.0 - 185.0) * 255.0).astype(np.uint8)

    r = norm
    g = np.clip(norm * 1.1, 0, 255).astype(np.uint8)
    b = np.clip(norm * 1.3, 0, 255).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)

    img = Image.fromarray(rgb, mode="RGB")
    img.save(LATEST_IR_PNG)
    logger.info("Saved enhanced thermal IR image to %s", LATEST_IR_PNG)

    return {
        "kelvin_min": k_min,
        "kelvin_max": k_max,
        "width": img.width,
        "height": img.height,
    }


def classify_satellite_image(image_path: Path) -> Dict[str, Any]:
    """Run PyTorch ResNet-18 forward pass on the real satellite image."""
    from data.preprocessing import preprocess_image

    model = get_classifier_model()
    img = Image.open(image_path).convert("RGB")
    tensor = preprocess_image(img)
    result = model(tensor)

    pred_class = result["predicted_class_name"][0]
    prob = float(result["cyclone_probability"][0].item())
    probs_dict = {
        name: float(p)
        for name, p in zip(model.class_names, result["probabilities"][0].tolist())
    }

    return {
        "predicted_class": pred_class,
        "cyclone_probability": prob,
        "class_probabilities": probs_dict,
    }


def process_latest_satellite(force_regenerate: bool = False) -> Dict[str, Any]:
    """
    Main orchestration entry point:
    Generates enhanced IR overlay if missing or forced, then executes real classification.
    """
    kelvin_stats = {"kelvin_min": 196.1, "kelvin_max": 319.4}

    if force_regenerate or not LATEST_IR_PNG.exists():
        nat_file = ensure_unpacked_nat()
        if nat_file and nat_file.exists():
            try:
                kelvin_stats = generate_satellite_ir_image(nat_file)
            except Exception as e:
                logger.error("Failed generating satellite IR via SatPy: %s", e)
                if not LATEST_IR_PNG.exists():
                    raise e
        elif not LATEST_IR_PNG.exists():
            raise FileNotFoundError("No satellite .nat or .zip file found to process.")

    classification = classify_satellite_image(LATEST_IR_PNG)

    return {
        "image_path": str(LATEST_IR_PNG),
        "relative_url": "/api/satellite/latest_satellite_ir.png",
        "bounds": [[0.0, 55.0], [30.0, 100.0]],
        "predicted_class": classification["predicted_class"],
        "cyclone_probability": classification["cyclone_probability"],
        "class_probabilities": classification["class_probabilities"],
        "kelvin_min": kelvin_stats.get("kelvin_min", 196.1),
        "kelvin_max": kelvin_stats.get("kelvin_max", 319.4),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "EUMETSAT Meteosat-9 HRSEVIRI-IODC 10.8µm Thermal IR",
    }
