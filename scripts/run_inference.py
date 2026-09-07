"""
scripts/run_inference.py
========================
Single entry point for end-to-end cyclone multimodal forecast inference.

Supports:
  1. Satellite image input (e.g. INSAT-3D, jpg, png, webp)
  2. Automatic GraphCast forecast input via NetCDF (--nc argument or default)
  3. Optional trained fusion model checkpoint (--checkpoint)

Usage:
  # With real image and automatic NetCDF forecast:
  python scripts/run_inference.py --image cyclone.webp --lat 20.4 --lon 88.1

  # Pointing to custom GraphCast netcdf forecast:
  python scripts/run_inference.py --image cyclone.webp --nc path/to/forecast.nc --lat 20.4 --lon 88.1
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.forecast import CycloneForecaster
from utils.visualization import build_full_dashboard_payload

try:
    import xarray as xr
    _XARRAY_AVAILABLE = True
except ImportError:
    _XARRAY_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser(description="Run cyclone multimodal forecast")
    parser.add_argument("--image", default=None,
                        help="Path to satellite image (jpg, png, webp).")
    parser.add_argument("--kelvin", default=None,
                        help="Path to calibrated Kelvin brightness-temperature .npy array.")
    parser.add_argument("--nat", default=None,
                        help="Path to raw EUMETSAT .nat file.")
    parser.add_argument("--live-eumetsat", action="store_true",
                        help="Automatically download and process the newest live EUMETSAT pass.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override cyclone detection gate threshold (e.g. 0.4 or 0.0 to force forecast).")
    parser.add_argument("--nc", default="graphcast_forecast.nc",
                        help="Path to GraphCast forecast .nc file.")
    parser.add_argument("--lat", type=float, default=20.4, help="Cyclone centre latitude")
    parser.add_argument("--lon", type=float, default=88.1, help="Cyclone centre longitude")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to trained fusion checkpoint (optional)")
    parser.add_argument("--config", default="config/config.yaml", help="Config path")
    args = parser.parse_args()

    print("=" * 60)
    print("  CYCLONE MULTIMODAL INFERENCE")
    print("=" * 60)

    # 1. Satellite image preparation (Default: Automatic live EUMETSAT pipeline)
    img = None
    if args.image and os.path.exists(args.image):
        img = Image.open(args.image).convert("RGB")
        print(f"[1/3] Satellite Image : Manual file override: {args.image} (Loaded)")

    elif args.kelvin and os.path.exists(args.kelvin):
        from scripts.eumetsat_live_pipeline import apply_cyclone_colormap
        print(f"[1/3] Satellite Image : Loading calibrated Kelvin array: {args.kelvin}")
        bt_kelvin = np.load(args.kelvin)
        img = apply_cyclone_colormap(bt_kelvin)
        print("      Calibrated Kelvin array colorized and ready for model.")

    elif args.nat and os.path.exists(args.nat):
        from scripts.eumetsat_live_pipeline import (
            read_and_calibrate_bt,
            apply_cyclone_colormap,
        )
        print(f"[1/3] Satellite Image : Processing EUMETSAT .nat file: {args.nat}")
        bt_kelvin = read_and_calibrate_bt(args.nat)
        img = apply_cyclone_colormap(bt_kelvin)
        print("      Raw .nat calibrated to Kelvin and colorized for model.")

    else:
        # Default: Automatic live EUMETSAT pipeline
        from scripts.eumetsat_live_pipeline import (
            download_latest_eumetsat,
            read_and_calibrate_bt,
            apply_cyclone_colormap,
        )
        # Check if we already have a recent .nat or .npy in eumetsat_output to avoid redundant re-downloading
        cached_npy = Path("eumetsat_output/cropped_bt_kelvin.npy")
        cached_nats = list(Path("eumetsat_output").glob("*.nat")) if Path("eumetsat_output").exists() else []

        if not args.live_eumetsat and cached_npy.exists():
            print(f"[1/3] Satellite Image : Using latest calibrated EUMETSAT array from cache ({cached_npy})")
            bt_kelvin = np.load(cached_npy)
            img = apply_cyclone_colormap(bt_kelvin)
        elif not args.live_eumetsat and cached_nats:
            latest_nat = max(cached_nats, key=lambda p: p.stat().st_mtime)
            print(f"[1/3] Satellite Image : Processing cached EUMETSAT pass: {latest_nat.name}")
            bt_kelvin = read_and_calibrate_bt(latest_nat)
            img = apply_cyclone_colormap(bt_kelvin)
        else:
            print("[1/3] Satellite Image : Pulling live 15-min pass from EUMETSAT Data Store...")
            nat_path = download_latest_eumetsat(download_dir="eumetsat_output")
            bt_kelvin = read_and_calibrate_bt(nat_path)
            img = apply_cyclone_colormap(bt_kelvin)
        print("      EUMETSAT satellite observation calibrated and ready.")

    # Always save the generated image to eumetsat_output for the UI and hackathon display
    if img is not None:
        from datetime import datetime, timezone
        out_dir = Path("eumetsat_output")
        archive_dir = out_dir / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)

        latest_path = out_dir / "latest_satellite_ir.png"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        archive_path = archive_dir / f"satellite_{timestamp}.png"

        img.save(latest_path)
        img.save(archive_path)
        print(f"      [SAVED] Latest satellite image : {latest_path}")
        print(f"      [SAVED] Archived pass image    : {archive_path}")

    # 2. Atmospheric / GraphCast runner preparation
    runner_fn = None
    if args.nc and os.path.exists(args.nc) and _XARRAY_AVAILABLE:
        print(f"[2/3] Atmospheric Data: {args.nc} (Loading GraphCast NetCDF)")
        try:
            nc_dataset = xr.open_dataset(args.nc)
            # Inject runner returning this dataset directly
            runner_fn = lambda inputs: nc_dataset
            print("      GraphCast runner configured with NetCDF forecast.")
        except Exception as e:
            print(f"      [Warning] Failed reading NetCDF '{args.nc}': {e}. Falling back to mock runner.")
    else:
        print(f"[2/3] Atmospheric Data: Mock mode (NetCDF '{args.nc}' not found)")

    # 3. Instantiate Forecaster
    forecaster = CycloneForecaster(
        config_path=args.config,
        fusion_checkpoint=args.checkpoint,
        graphcast_runner_fn=runner_fn,
    )

    if args.threshold is not None:
        forecaster.detection_threshold = args.threshold
        print(f"      Detection threshold overridden to: {args.threshold}")

    print(f"[3/3] Cyclone Center  : ({args.lat} N, {args.lon} E)")
    print("\nRunning inference pipeline...")

    result = forecaster.forecast(
        satellite_image=img,
        graphcast_input=None,
        cyclone_lat=args.lat,
        cyclone_lon=args.lon,
    )

    print("\n" + "-" * 60)
    print("FORECAST RESULT:")
    print("-" * 60)
    print(json.dumps(result, indent=2))
    print("-" * 60)

    if result.get("cyclone_detected"):
        payload = build_full_dashboard_payload(result)
        print("\nINTENSITY CARDS:")
        for card in payload["atmospheric_cards"]:
            print(f"  {card['label']:12s}: wind={card['wind_kt']:.1f} kt | "
                  f"pressure={card['pressure_hPa']:.1f} hPa | {card['category']}")
        print(f"\nGeoJSON track features: {len(payload['track_geojson']['features'])}")

        # Save dashboard payload & GeoJSON for frontend / hackathon presentation
        out_json = Path("eumetsat_output/dashboard_payload.json")
        out_geojson = Path("eumetsat_output/cyclone_track.geojson")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        with open(out_geojson, "w", encoding="utf-8") as f:
            json.dump(payload["track_geojson"], f, indent=2)
        print(f"\n[SAVED] Dashboard JSON : {out_json}")
        print(f"[SAVED] GeoJSON Track  : {out_geojson}")
    else:
        print("\nNo cyclone detected (probability below detection threshold).")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
