"""
Force-demo: lowers the detection threshold so we can see the full
forecast output even with a synthetic (non-cyclone) image.

In real usage, a genuine INSAT cyclone image will give probability > 0.5
automatically.

Usage:
    python scripts/run_demo_forced.py
"""
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
from config.config_loader import load_config, get_device
from inference.forecast import CycloneForecaster
from utils.visualization import build_full_dashboard_payload

SEPARATOR = "=" * 62


def main():
    print(SEPARATOR)
    print("  FULL FORECAST DEMO (threshold=0.0 to force fusion output)")
    print(SEPARATOR)
    print()

    forecaster = CycloneForecaster(
        config_path="config/config.yaml",
        fusion_checkpoint=None,
        graphcast_runner_fn=None,
    )

    # Lower the threshold to 0.0 so any image triggers the fusion
    forecaster.detection_threshold = 0.0
    print("Detection threshold overridden to 0.0 (to demo full output)")
    print("(With a real cyclone image this is not needed)")
    print()

    img = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype("uint8"))

    CYCLONE_LAT = 20.4
    CYCLONE_LON = 88.1
    print(f"Cyclone centre: ({CYCLONE_LAT} N, {CYCLONE_LON} E)")
    print()

    print("Running pipeline...")
    result = forecaster.forecast(
        satellite_image=img,
        graphcast_input=None,
        cyclone_lat=CYCLONE_LAT,
        cyclone_lon=CYCLONE_LON,
    )
    print()

    print("-" * 62)
    print(json.dumps(result, indent=2))
    print("-" * 62)
    print()

    if result.get("cyclone_detected"):
        payload = build_full_dashboard_payload(result)

        print("Track prediction (from current centre):")
        for h, fcast in result["forecast"].items():
            dlat = round(fcast["lat"] - CYCLONE_LAT, 4)
            dlon = round(fcast["lon"] - CYCLONE_LON, 4)
            print(f"  +{h:>3s}: ({fcast['lat']:.4f} N, {fcast['lon']:.4f} E)  "
                  f"delta=({dlat:+.4f}, {dlon:+.4f})")
        print()

        print("Intensity forecast:")
        for card in payload["atmospheric_cards"]:
            w = card["wind_kt"]
            p = card["pressure_hPa"]
            c = card["category"]
            label = card["label"]
            print(f"  {label:12s}: {w:.1f} kt | {p:.1f} hPa | {c}")
        print()

        print(f"Forecast confidence : {result['forecast_confidence']:.4f}")
        print(f"Satellite class     : {result['predicted_class']}")
        print(f"Cyclone probability : {result['cyclone_probability']:.4f}")
        print()
        print(f"GeoJSON features    : {len(payload['track_geojson']['features'])}")
        print(f"  (1 LineString track + 5 Point markers)")

    print()
    print(SEPARATOR)
    print("  NOTE: Fusion weights are random (untrained).")
    print("  After training with real data, predictions will be accurate.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
