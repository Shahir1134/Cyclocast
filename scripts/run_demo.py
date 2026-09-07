"""
End-to-end pipeline demo on the terminal.
Runs: Satellite -> GraphCast (mock) -> Atmospheric features -> Fusion -> Forecast JSON

Usage:
    python scripts/run_demo.py
"""
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
from inference.forecast import CycloneForecaster
from utils.visualization import build_full_dashboard_payload

SEPARATOR = "=" * 62

def main():
    print(SEPARATOR)
    print("  CYCLONE MULTIMODAL FUSION - END-TO-END TERMINAL DEMO")
    print(SEPARATOR)
    print()

    # Step 1 — load forecaster
    print("[1/5] Loading CycloneForecaster (real cyclone.pth + mock GraphCast)...")
    forecaster = CycloneForecaster(
        config_path="config/config.yaml",
        fusion_checkpoint=None,       # swap in your checkpoint after training
        graphcast_runner_fn=None,     # swap in your JAX runner
    )
    print("      Done.")
    print()

    # Step 2 — satellite image
    print("[2/5] Preparing satellite image (synthetic - replace with real INSAT image)...")
    img = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype("uint8"))
    print("      224x224 RGB image ready.")
    print()

    # Step 3 — cyclone centre
    CYCLONE_LAT = 20.4   # Bay of Bengal
    CYCLONE_LON = 88.1
    print(f"[3/5] Cyclone centre: ({CYCLONE_LAT} N, {CYCLONE_LON} E)")
    print()

    # Step 4 — run pipeline
    print("[4/5] Running full pipeline (Satellite -> GraphCast -> Fusion)...")
    result = forecaster.forecast(
        satellite_image=img,
        graphcast_input=None,
        cyclone_lat=CYCLONE_LAT,
        cyclone_lon=CYCLONE_LON,
    )
    print("      Pipeline complete.")
    print()

    # Step 5 — output
    print("[5/5] Structured forecast output:")
    print("-" * 62)
    print(json.dumps(result, indent=2))
    print("-" * 62)
    print()

    if result.get("cyclone_detected"):
        payload = build_full_dashboard_payload(result)
        print("Dashboard intensity cards:")
        for card in payload["atmospheric_cards"]:
            w = card["wind_kt"]
            p = card["pressure_hPa"]
            c = card["category"]
            label = card["label"]
            print(f"  {label:12s}: wind={w:.1f} kt | pressure={p:.1f} hPa | {c}")
        print()
        print(f"GeoJSON features  : {len(payload['track_geojson']['features'])} (1 track + 5 position markers)")
        ts = payload["intensity_timeseries"]
        print(f"Wind series (kt)  : {[str(v) for v in ts['wind_kt']]}")
        print(f"Press series (hPa): {[str(v) for v in ts['pressure_hPa']]}")
    else:
        print("No cyclone detected.")
        print("Message:", result.get("message"))

    print()
    print(SEPARATOR)
    print("  DEMO COMPLETE")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
