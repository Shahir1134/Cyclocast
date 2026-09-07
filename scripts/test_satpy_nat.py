from pathlib import Path
import numpy as np
from PIL import Image
from satpy import Scene

nat_path = Path("tmp_eumetsat/MSG2-SEVI-MSG15-0100-NA-20260905052739.278000000Z-NA.nat")
print(f"Loading Scene from {nat_path}...")
scn = Scene(filenames=[str(nat_path)], reader="seviri_l1b_native")

print("Available datasets:", scn.available_dataset_names())
scn.load(["IR_108"], calibration="brightness_temperature")

print("Calibrated IR_108 loaded. Resampling/cropping to Bay of Bengal & Arabian Sea...")
# Area covering Indian Ocean: 0-30N, 55-100E
cropped = scn.crop(ll_bbox=(55.0, 0.0, 100.0, 30.0))
data = cropped["IR_108"].values
data = np.squeeze(data).astype(np.float32)

print("Data shape:", data.shape)
print(f"Kelvin range: min={np.nanmin(data):.1f}K, max={np.nanmax(data):.1f}K")

# Enhanced Dvorak/Thermal IR colorization (standard meteorological IR curve)
# Invert & scale: Cold clouds (<200K) bright white/cyan/pink; warm ocean (>300K) dark
cleaned = np.nan_to_num(data, nan=305.0)
clamped = np.clip(cleaned, 185.0, 305.0)
norm = ((305.0 - clamped) / (305.0 - 185.0) * 255.0).astype(np.uint8)

# Colorize with thermal tint
r = norm
g = np.clip(norm * 1.1, 0, 255).astype(np.uint8)
b = np.clip(norm * 1.3, 0, 255).astype(np.uint8)
rgb = np.stack([r, g, b], axis=-1)

img = Image.fromarray(rgb, mode="RGB")
out_img = Path("eumetsat_output/latest_satellite_ir.png")
out_img.parent.mkdir(parents=True, exist_ok=True)
img.save(out_img)
print(f"Successfully generated real satellite image: {out_img} ({out_img.stat().st_size / 1024:.1f} KB)")
