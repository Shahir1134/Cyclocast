"""
scripts/eumetsat_live_pipeline.py
==================================
Live cyclone-intensity classification pipeline for EUMETSAT HRSEVIRI-IODC data.

Pipeline Architecture (Sequential & Dependency-Chained):
  Step 1: Authenticate & Download latest .nat file from EUMETSAT Data Store (eumdac)
          - WHY: Raw satellite telemetry must be acquired from the live 15-min feed.
  Step 2: Read & Calibrate to Brightness Temperature in Kelvin, Crop Region (satpy)
          - WHY: Raw digital counts/radiance are non-linear; the enhancement LUT expects
            true physical temperature in Kelvin. Uncalibrated data makes colorization meaningless.
  Step 3: Apply Dvorak Enhanced-IR Colorization LUT (Stub)
          - WHY: The ResNet-18 model was trained exclusively on a specific Dvorak colorized
            style. Grayscale or standard colormaps (jet/viridis) cause silent prediction failures.
  Step 4: ImageNet Normalization & Resize (Exact Training Match)
          - WHY: The backbone requires 224x224 RGB tensors normalized with ImageNet mean/std.
            Any deviation silently degrades classification accuracy.
  Step 5: Run PyTorch ResNet-18 Inference
          - WHY: Generates real-time cyclone intensity class and confidence score.

Installation:
  pip install eumdac satpy pyresample pillow torchvision torch numpy
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# Ensure project root is available on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("eumetsat_pipeline")


# ===========================================================================
# STEP 1: Authentication and Download (eumdac)
# ===========================================================================

def download_latest_eumetsat(
    consumer_key: Optional[str] = None,
    consumer_secret: Optional[str] = None,
    download_dir: str | Path = "tmp_eumetsat",
    window_minutes: int = 45,
    latency_hours: float = 0.0,
    collection_id: str = "EO:EUM:DAT:MSG:HRSEVIRI-IODC",
) -> Path:
    """
    Search and download the most recent HRSEVIRI-IODC .nat product.

    WHY THIS STEP EXISTS:
    The EUMETSAT Data Store delivers full-disk Meteosat-8 observations at 41.5°E
    every 15 minutes. This step acquires the newest available observation pass
    to ensure the cyclone intensity analysis is live.

    Parameters
    ----------
    consumer_key : str, optional
        EUMETSAT API consumer key. If omitted, reads from EUMETSAT_CONSUMER_KEY env var.
    consumer_secret : str, optional
        EUMETSAT API consumer secret. If omitted, reads from EUMETSAT_CONSUMER_SECRET env var.
    download_dir : str | Path
        Local directory to store downloaded .nat files.
    window_minutes : int
        Lookback duration in minutes from current UTC time.
    collection_id : str
        EUMETSAT collection identifier.

    Returns
    -------
    Path
        Path to the downloaded .nat product file.
    """
    try:
        import eumdac
    except ImportError as exc:
        raise ImportError(
            "eumdac library is required for Step 1. Install via: pip install eumdac"
        ) from exc

    # Retrieve credentials securely without printing or leaking them
    key = consumer_key or os.environ.get("EUMETSAT_CONSUMER_KEY")
    secret = consumer_secret or os.environ.get("EUMETSAT_CONSUMER_SECRET")

    if not key or not secret:
        try:
            from eumdac.cli import load_credentials as _eumdac_load_creds
            loaded = _eumdac_load_creds()
            if loaded and len(loaded) >= 2:
                key, secret = loaded[0], loaded[1]
        except Exception:
            pass

    if not key or not secret:
        raise ValueError(
            "EUMETSAT credentials not found. Provide them via arguments, set the "
            "EUMETSAT_CONSUMER_KEY and EUMETSAT_CONSUMER_SECRET environment variables, "
            "or run 'eumdac set-credentials <key> <secret>'."
        )

    download_path = Path(download_dir)
    download_path.mkdir(parents=True, exist_ok=True)

    logger.info("Authenticating with EUMETSAT Data Store API...")
    try:
        token = eumdac.AccessToken((key, secret))
        datastore = eumdac.DataStore(token)
        selected_collection = datastore.get_collection(collection_id)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to authenticate with EUMETSAT API: {exc}. Please verify your credentials."
        ) from exc

    now_utc = datetime.now(timezone.utc)
    # EUMETSAT Open Data Policy allows free automated downloads for products >1hr or >2hr latency.
    # To avoid 403 Unauthorised on restricted <1hr real-time data, we query products in the open window.
    end_time = now_utc - timedelta(hours=latency_hours)
    start_time = end_time - timedelta(minutes=window_minutes)
    logger.info(
        "Searching collection '%s' for products between %s and %s UTC (latency offset: %sh)...",
        collection_id,
        start_time.strftime("%Y-%m-%d %H:%M"),
        end_time.strftime("%Y-%m-%d %H:%M"),
        latency_hours,
    )

    try:
        products = selected_collection.search(dtstart=start_time, dtend=end_time)
        product_list = list(products)
    except Exception as exc:
        raise RuntimeError(
            f"Error querying EUMETSAT collection '{collection_id}': {exc}"
        ) from exc

    if not product_list:
        raise FileNotFoundError(
            f"No products found in collection '{collection_id}' between "
            f"{start_time.strftime('%H:%M')} and {end_time.strftime('%H:%M')} UTC. "
            f"Try increasing the lookback window using --window-minutes."
        )

    # Sort to select the most recent product in the open window
    latest_product = sorted(product_list, key=lambda p: p.sensing_start, reverse=True)[0]
    logger.info(
        "Found latest product: %s (Sensing start: %s UTC)",
        latest_product._id,
        latest_product.sensing_start,
    )

    # Locate the native .nat entry
    target_entry = None
    for entry in latest_product.entries:
        if str(entry).endswith(".nat") or "native" in str(entry).lower():
            target_entry = entry
            break
    if target_entry is None:
        target_entry = latest_product.entries[0]

    out_file = download_path / str(target_entry)
    if out_file.exists():
        logger.info("File already exists locally: %s", out_file)
        return out_file

    logger.info("Downloading product entry '%s' to '%s'...", target_entry, out_file)
    try:
        with latest_product.open(entry=target_entry) as f_src, open(out_file, "wb") as f_dst:
            chunk_size = 1024 * 1024
            while True:
                chunk = f_src.read(chunk_size)
                if not chunk:
                    break
                f_dst.write(chunk)
    except Exception as exc:
        if out_file.exists():
            out_file.unlink()
        raise IOError(f"Failed to complete download of '{target_entry}': {exc}") from exc

    logger.info("Download completed successfully: %s (%.2f MB)", out_file, out_file.stat().st_size / 1e6)
    return out_file


# ===========================================================================
# STEP 2: Read, Calibrate to Brightness Temperature & Crop (satpy)
# ===========================================================================

def read_and_calibrate_bt(
    file_path: str | Path,
    channel: str = "IR_108",
    crop_bounds: Tuple[float, float, float, float] = (55.0, 0.0, 100.0, 30.0),
) -> np.ndarray:
    """
    Load SEVIRI .nat file, calibrate IR channel to Brightness Temperature (Kelvin),
    and crop to the Arabian Sea / Bay of Bengal region.

    WHY THIS STEP EXISTS:
    1. Channel Selection: The 10.8 µm infrared window channel penetrates atmospheric
       water vapor to record cloud-top temperatures, which is the foundational signal
       for tropical cyclone intensity estimation (Dvorak technique).
    2. Calibration: Raw sensor digital numbers or radiances are not physically linear.
       The enhancement color LUT is calibrated strictly against temperature in Kelvin.
       Omitting calibration produces distorted colors and incorrect predictions.
    3. Spatial Crop: Crops out the full disk to focus on the storm basin
       (0°-30°N, 55°-100°E) without cartographic distortion.

    Parameters
    ----------
    file_path : str | Path
        Path to downloaded SEVIRI .nat file.
    channel : str
        Target IR channel (default: "IR_108").
    crop_bounds : tuple of float
        (min_lon, min_lat, max_lon, max_lat) in degrees.
        Default: (55.0, 0.0, 100.0, 30.0) covers the North Indian Ocean basin.

    Returns
    -------
    np.ndarray
        2D array of brightness temperatures in Kelvin.
    """
    try:
        from satpy import Scene
    except ImportError as exc:
        raise ImportError(
            "satpy library is required for Step 2. Install via: pip install satpy pyresample"
        ) from exc

    file_p = Path(file_path)
    if not file_p.exists():
        raise FileNotFoundError(f"Input file not found at: {file_p}")

    logger.info("Loading scene from '%s' using 'seviri_l1b_native' reader...", file_p.name)
    try:
        scn = Scene(filenames=[str(file_p)], reader="seviri_l1b_native")
        logger.info("Calibrating channel '%s' to Brightness Temperature (Kelvin)...", channel)
        scn.load([channel], calibration="brightness_temperature")
    except Exception as exc:
        raise RuntimeError(f"Satpy failed to decode and calibrate '{file_p}': {exc}") from exc

    # Crop to Arabian Sea / Bay of Bengal region
    min_lon, min_lat, max_lon, max_lat = crop_bounds
    logger.info(
        "Cropping region to bounds: Lon [%.1f, %.1f], Lat [%.1f, %.1f]...",
        min_lon, max_lon, min_lat, max_lat
    )
    try:
        cropped_scn = scn.crop(ll_bbox=(min_lon, min_lat, max_lon, max_lat))
        data_array = cropped_scn[channel].values
    except Exception as exc:
        logger.warning(
            "Coordinate crop via bounding box failed (%s). Attempting coordinate indexing fallback...",
            exc,
        )
        try:
            # Fallback: compute coordinates explicitly
            dataset = scn[channel]
            lons, lats = dataset.attrs["area"].get_lonlats()
            mask = (lats >= min_lat) & (lats <= max_lat) & (lons >= min_lon) & (lons <= max_lon)
            y_indices, x_indices = np.where(mask)
            if len(y_indices) == 0:
                raise ValueError("No pixels found within the specified geographic bounding box.")
            ymin, ymax = y_indices.min(), y_indices.max() + 1
            xmin, xmax = x_indices.min(), x_indices.max() + 1
            data_array = dataset.values[ymin:ymax, xmin:xmax]
        except Exception as fallback_exc:
            raise RuntimeError(
                f"Failed to crop satellite image to geographic bounds: {fallback_exc}"
            ) from fallback_exc

    # Remove singleton dimensions if present
    data_array = np.squeeze(data_array).astype(np.float32)

    # Sanity check: Brightness temperatures in the troposphere typically range 170K - 330K
    valid_mask = ~np.isnan(data_array)
    if not np.any(valid_mask):
        raise ValueError("Calibrated brightness temperature array contains only NaN values.")

    t_min = float(np.nanmin(data_array))
    t_max = float(np.nanmax(data_array))
    logger.info("Brightness Temperature calibrated: Min=%.1f K, Max=%.1f K, Shape=%s", t_min, t_max, data_array.shape)

    return data_array


# ===========================================================================
# STEP 3: Apply Dvorak Colorization LUT (Stub)
# ===========================================================================

def apply_cyclone_colormap(temp_array_kelvin: np.ndarray) -> Image.Image:
    """
    Apply Dvorak-style enhanced IR color lookup table (LUT) to brightness temperature data.

    WHY THIS STEP EXISTS:
    The cyclone intensity classifier was trained ONLY on a specific colorized
    enhanced-infrared image style (Dvorak-technique color curves where specific
    cloud-top temperature intervals are mapped to distinct RGB bands: cold cloud tops,
    CDO, eye temperature).
    
    CRITICAL WARNING:
    If raw grayscale or generic false-color schemes (such as matplotlib 'jet' or 'viridis')
    are used, the convolutional filters will extract mismatched spatial features and
    silently output incorrect intensity predictions without throwing any error.

    Parameters
    ----------
    temp_array_kelvin : np.ndarray
        2D float array of brightness temperatures in Kelvin.

    Returns
    -------
    PIL.Image.Image
        Colorized RGB image ready for model transformation.
    """
    # -----------------------------------------------------------------------
    # TODO: REPLACE WITH EXACT USER-DEFINED DVORAK COLOR CURVE / LOOKUP TABLE
    # -----------------------------------------------------------------------
    # Note: This is an unfinalized stub function. Do NOT deploy to production
    # until the exact training color curve mapping (Kelvin -> RGB) is provided.
    logger.warning(
        "Using placeholder colormap stub in Step 3. "
        "Supply the exact Dvorak colorization LUT before running final predictions."
    )

    # Temporary placeholder: Handle NaNs from space/unobserved pixels cleanly
    cleaned = np.nan_to_num(temp_array_kelvin, nan=310.0)
    clamped = np.clip(cleaned, 190.0, 310.0)
    normalized = ((310.0 - clamped) / (310.0 - 190.0) * 255.0).astype(np.uint8)
    
    # Replicate into 3-channel RGB
    rgb_array = np.stack([normalized, normalized, normalized], axis=-1)
    
    return Image.fromarray(rgb_array, mode="RGB")


# ===========================================================================
# STEP 4: Model Preprocessing (Exact Match to Training)
# ===========================================================================

def preprocess_for_model(rgb_image: Image.Image) -> torch.Tensor:
    """
    Apply exact torchvision transforms matching training preprocessing.

    WHY THIS STEP EXISTS:
    The model backbone is an ImageNet-pretrained ResNet-18 fine-tuned on 224x224
    colorized IR images. The normalization constants (mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]) match the exact distribution used during training.
    Any mismatch in resizing interpolation or missing normalization shifts feature
    activations and silently degrades classification accuracy.

    Parameters
    ----------
    rgb_image : PIL.Image.Image
        Input RGB satellite image.

    Returns
    -------
    torch.Tensor
        Batch tensor of shape (1, 3, 224, 224) ready for model forward pass.
    """
    inference_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = rgb_image.convert("RGB")
    tensor = inference_transform(img)
    # Add batch dimension: (3, 224, 224) -> (1, 3, 224, 224)
    return tensor.unsqueeze(0)


# ===========================================================================
# STEP 5: Model Loading and Inference
# ===========================================================================

CLASS_NAMES = ["Low", "Moderate", "Severe", "Very Severe", "Extreme"]

def load_trained_model(
    checkpoint_path: str | Path = "cyclone.pth",
    device: Optional[torch.device | str] = None,
) -> torch.nn.Module:
    """
    Load the trained ResNet-18 cyclone classifier checkpoint.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to saved PyTorch state_dict (.pth file).
    device : torch.device or str, optional
        Target device ('cuda' or 'cpu').

    Returns
    -------
    torch.nn.Module
        Loaded and frozen PyTorch evaluation model.
    """
    from torchvision.models import resnet18, ResNet18_Weights

    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {ckpt_p}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    logger.info("Instantiating ResNet-18 architecture on device '%s'...", device)
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # Match the fine-tuned architecture from Cyclone.ipynb
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.4),
        torch.nn.Linear(512, len(CLASS_NAMES))
    )

    logger.info("Loading checkpoint weights from '%s'...", ckpt_p.name)
    state_dict = torch.load(ckpt_p, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    # Freeze weights for inference
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    model.to(device)
    return model


def run_inference(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: Optional[torch.device | str] = None,
) -> Dict[str, Any]:
    """
    Execute forward pass and compute cyclone intensity classification.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded PyTorch model in evaluation mode.
    input_tensor : torch.Tensor
        Batch tensor of shape (1, 3, 224, 224).
    device : torch.device or str, optional
        Computation device.

    Returns
    -------
    dict
        Prediction results containing class name, confidence, and class probabilities.
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    tensor = input_tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().squeeze(0).numpy()

    predicted_idx = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(probs[predicted_idx])

    # Cyclone probability proxy: sum of classes > 'Low'
    cyclone_prob = float(np.sum(probs[1:]))

    results = {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "cyclone_probability": round(cyclone_prob, 4),
        "class_probabilities": {
            cls_name: round(float(p), 4) for cls_name, p in zip(CLASS_NAMES, probs)
        },
    }
    return results


# ===========================================================================
# CLI Orchestrator
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EUMETSAT Live Cyclone Intensity Inference Pipeline"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute steps 1 and 2 only; save calibrated Kelvin array as .npy for calibration inspection.",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to an existing local .nat file. If provided, skips Step 1 (download).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="cyclone.pth",
        help="Path to trained PyTorch model checkpoint (.pth).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eumetsat_output",
        help="Directory to save downloaded files and dry-run .npy arrays.",
    )
    parser.add_argument(
        "--latency-hours",
        type=float,
        default=0.0,
        help="Latency offset in hours (default: 0.0 for real-time live data).",
    )
    parser.add_argument(
        "--window-minutes",
        type=int,
        default=45,
        help="Lookback search window in minutes (default: 45).",
    )
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help="EUMETSAT Consumer Key (optional, defaults to EUMETSAT_CONSUMER_KEY env var).",
    )
    parser.add_argument(
        "--secret",
        type=str,
        default=None,
        help="EUMETSAT Consumer Secret (optional, defaults to EUMETSAT_CONSUMER_SECRET env var).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  LIVE CYCLONE INTENSITY CLASSIFICATION PIPELINE (EUMETSAT IODC)")
    print("=" * 65)

    # ── STEP 1: Acquisition ────────────────────────────────────────────────
    if args.file:
        nat_file = Path(args.file)
        logger.info("[Step 1/5] Using local file provided via --file: %s", nat_file)
    else:
        logger.info("[Step 1/5] Downloading latest pass from EUMETSAT Data Store...")
        try:
            nat_file = download_latest_eumetsat(
                consumer_key=args.key,
                consumer_secret=args.secret,
                download_dir=out_dir,
                window_minutes=args.window_minutes,
                latency_hours=args.latency_hours,
            )
        except Exception as exc:
            logger.error("Step 1 failed: %s", exc)
            sys.exit(1)

    # ── STEP 2: Calibration to Kelvin & Crop ──────────────────────────────
    logger.info("[Step 2/5] Reading, calibrating to Kelvin (IR 10.8µm) & cropping...")
    try:
        bt_kelvin = read_and_calibrate_bt(nat_file)
    except Exception as exc:
        logger.error("Step 2 failed: %s", exc)
        sys.exit(1)

    if args.dry_run:
        dry_run_file = out_dir / "cropped_bt_kelvin.npy"
        np.save(dry_run_file, bt_kelvin)
        logger.info(
            "[DRY-RUN] Step 2 complete. Calibrated brightness-temperature array saved to: %s",
            dry_run_file,
        )
        logger.info(
            "[DRY-RUN] Array stats -> Shape: %s | Min: %.2f K | Max: %.2f K | Mean: %.2f K",
            bt_kelvin.shape,
            np.nanmin(bt_kelvin),
            np.nanmax(bt_kelvin),
            np.nanmean(bt_kelvin),
        )
        print("=" * 65)
        print(f"DRY-RUN SUCCESSFUL: Calibrated array saved to {dry_run_file}")
        print("=" * 65)
        return

    # ── STEP 3: Colorization LUT ───────────────────────────────────────────
    logger.info("[Step 3/5] Applying Dvorak colorization LUT (stub)...")
    rgb_image = apply_cyclone_colormap(bt_kelvin)

    # ── STEP 4: Preprocessing Transform ───────────────────────────────────
    logger.info("[Step 4/5] Preprocessing tensor (224x224, ImageNet normalized)...")
    input_tensor = preprocess_for_model(rgb_image)
    logger.info("Tensor ready for model: shape=%s, dtype=%s", input_tensor.shape, input_tensor.dtype)

    # ── STEP 5: Run Inference ─────────────────────────────────────────────
    logger.info("[Step 5/5] Loading model checkpoint and executing inference...")
    try:
        model = load_trained_model(args.model_path)
        inference_result = run_inference(model, input_tensor)
    except Exception as exc:
        logger.error("Step 5 failed: %s", exc)
        sys.exit(1)

    print("\n" + "-" * 65)
    print("LIVE CLASSIFICATION RESULT:")
    print("-" * 65)
    print(f"  Predicted Intensity Class : {inference_result['predicted_class']}")
    print(f"  Class Confidence          : {inference_result['confidence'] * 100:.1f}%")
    print(f"  Cyclone Probability       : {inference_result['cyclone_probability'] * 100:.1f}%")
    print("  Full Class Probabilities  :")
    for cls_name, prob in inference_result["class_probabilities"].items():
        print(f"    - {cls_name:12s}: {prob * 100:.2f}%")
    print("-" * 65)
    print("=" * 65)


if __name__ == "__main__":
    main()
