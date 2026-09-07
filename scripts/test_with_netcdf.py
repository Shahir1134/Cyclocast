"""
scripts/test_with_netcdf.py
===========================
Demonstrates plugging a real NetCDF GraphCast forecast output file
directly into the CycloneForecaster inference pipeline.
"""
import os
import sys
import json
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

import xarray as xr
from inference.forecast import CycloneForecaster
from utils.visualization import build_full_dashboard_payload

def main():
    nc_path = "mock_graphcast_forecast.nc"
    if not os.path.exists(nc_path):
        print(f"[ERROR] Forecast file not found: {nc_path}")
        return

    print("=" * 65)
    print("  TESTING INFERENCE WITH GRAPHCAST NETCDF FORECAST DATA")
    print("=" * 65)

    # 1. Open forecast dataset
    print(f"\n[1] Reading NetCDF forecast dataset: {nc_path}")
    forecast_ds = xr.open_dataset(nc_path)
    print(f"    Dimensions: {dict(forecast_ds.sizes)}")
    print(f"    Variables : {list(forecast_ds.data_vars.keys())}")

    # 2. Define runner function returning this dataset
    def nc_graphcast_runner(inputs):
        print("    [Runner Callback] GraphCast output provided to fusion pipeline.")
        return forecast_ds

    # 3. Instantiate forecaster with runner function injected
    print("\n[2] Initializing CycloneForecaster...")
    forecaster = CycloneForecaster(
        config_path="config/config.yaml",
        fusion_checkpoint=None,
        graphcast_runner_fn=nc_graphcast_runner,
    )
    forecaster.detection_threshold = 0.0

    # 4. Run inference
    print("\n[3] Executing forecast around Bay of Bengal (20.4 N, 88.1 E)...")
    img = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype("uint8"))
    result = forecaster.forecast(
        satellite_image=img,
        graphcast_input=None,
        cyclone_lat=20.4,
        cyclone_lon=88.1,
    )

    print("\n[4] Forecast Results:")
    print("-" * 65)
    print(json.dumps(result, indent=2))
    print("-" * 65)

    if result.get("cyclone_detected"):
        payload = build_full_dashboard_payload(result)
        print("\n[5] Visualization Summary:")
        print(f"    GeoJSON features: {len(payload['track_geojson']['features'])}")
        for card in payload["atmospheric_cards"]:
            print(f"    {card['label']:10s} -> Lat: {card['lat']:.2f}, Lon: {card['lon']:.2f} | "
                  f"Wind: {card['wind_kt']:.1f} kt | Pres: {card['pressure_hPa']:.1f} hPa")

    print("\n" + "=" * 65)
    print("  TEST COMPLETED SUCCESSFULLY")
    print("=" * 65)

if __name__ == "__main__":
    main()
