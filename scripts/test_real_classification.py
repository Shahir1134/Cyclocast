import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image
from models.satellite_wrapper import SatelliteWrapper
from data.preprocessing import preprocess_image

model = SatelliteWrapper("cyclone.pth", device="cpu")
img = Image.open("eumetsat_output/latest_satellite_ir.png").convert("RGB")
tensor = preprocess_image(img)
result = model(tensor)

print("=== INFERENCE ON REAL METEOSAT SATELLITE PASS ===")
print("Predicted Class:", result["predicted_class_name"][0])
print(f"Cyclone Probability: {result['cyclone_probability'][0].item()*100:.2f}%")
print("Class Probabilities:")
for name, prob in zip(model.class_names, result["probabilities"][0].tolist()):
    print(f"  - {name:15s}: {prob*100:.2f}%")
