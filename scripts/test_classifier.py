import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
import numpy as np

from models.satellite_wrapper import SatelliteWrapper
from data.preprocessing import preprocess_image

# ── Step 1: Load real cyclone.pth ─────────────────────────────────────────────
model = SatelliteWrapper("cyclone.pth", device="cpu")
print("=== Real cyclone.pth loaded ===")
print("Classes:", model.class_names)
print()

# ── Step 2: Find a real image or use synthetic ────────────────────────────────
img_path = None
for f in os.listdir("."):
    if f.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = f
        break

if img_path:
    img = Image.open(img_path).convert("RGB")
    print(f"Using real image: {img_path}")
else:
    arr = (np.random.rand(256, 256, 3) * 255).astype("uint8")
    img = Image.fromarray(arr)
    print("No image found in folder — using synthetic random image")

# ── Step 3: Run classifier ────────────────────────────────────────────────────
tensor = preprocess_image(img)
result = model(tensor)

print()
print(f"Predicted class     : {result['predicted_class_name'][0]}")
print(f"Cyclone probability : {result['cyclone_probability'][0].item():.4f}")
print(f"All class probs     : {[round(p, 4) for p in result['probabilities'][0].tolist()]}")
print(f"Embedding shape     : {result['satellite_embedding'].shape}")
print(f"Embedding norm      : {result['satellite_embedding'].norm().item():.4f}")
