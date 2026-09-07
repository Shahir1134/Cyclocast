"""
scripts/test_eumetsat_pipeline.py
==================================
Unit verification for steps 3-5 of the EUMETSAT pipeline
using synthetic Kelvin brightness-temperature data and the real cyclone.pth.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from scripts.eumetsat_live_pipeline import (
    apply_cyclone_colormap,
    preprocess_for_model,
    load_trained_model,
    run_inference
)

def test_pipeline():
    print("=" * 60)
    print("  VERIFYING PIPELINE STEPS 3 -> 4 -> 5")
    print("=" * 60)

    # 1. Step 3 test
    print("\n[Step 3] Testing apply_cyclone_colormap with 400x400 Kelvin array...")
    mock_kelvin = np.linspace(190.0, 310.0, 400 * 400, dtype=np.float32).reshape(400, 400)
    rgb_img = apply_cyclone_colormap(mock_kelvin)
    assert isinstance(rgb_img, Image.Image), "Output must be PIL Image"
    assert rgb_img.mode == "RGB", f"Expected RGB mode, got {rgb_img.mode}"
    print(f"         Output PIL Image: {rgb_img.size}, mode: {rgb_img.mode} (OK)")

    # 2. Step 4 test
    print("\n[Step 4] Testing preprocess_for_model (Resize 224x224 + ImageNet Normalization)...")
    tensor = preprocess_for_model(rgb_img)
    assert tensor.shape == (1, 3, 224, 224), f"Expected shape (1, 3, 224, 224), got {tensor.shape}"
    assert tensor.dtype == torch.float32, f"Expected float32, got {tensor.dtype}"
    print(f"         Output Tensor: {tensor.shape}, {tensor.dtype} (OK)")

    # 3. Step 5 test
    print("\n[Step 5] Testing model load and inference with cyclone.pth...")
    model = load_trained_model("cyclone.pth", device="cpu")
    result = run_inference(model, tensor, device="cpu")
    print(f"         Prediction: {result['predicted_class']} (Confidence: {result['confidence']*100:.1f}%)")
    print(f"         Cyclone Probability: {result['cyclone_probability']*100:.1f}%")
    print("         Probabilities:")
    for c, p in result["class_probabilities"].items():
        print(f"           - {c:12s}: {p*100:.2f}%")

    print("\n" + "=" * 60)
    print("  PIPELINE UNIT TEST PASSED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    import torch
    test_pipeline()
