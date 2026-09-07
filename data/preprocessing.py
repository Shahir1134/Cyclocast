"""
data/preprocessing.py
=======================
Image pre-processing transforms that exactly match the training notebook
(Cyclone.ipynb Cell 12) so the satellite wrapper receives correctly
normalised tensors at inference time.

Also contains utilities for normalising atmospheric data.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torchvision import transforms

# ── Image normalisation parameters (must match training) ─────────────────────
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD:  List[float] = [0.229, 0.224, 0.225]

INPUT_SIZE: int = 224  # pixels


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_inference_transform(input_size: int = INPUT_SIZE) -> transforms.Compose:
    """
    Return the inference-time image transform.

    Matches `test_transform_v4` from the training notebook exactly:
        Resize → ToTensor → Normalize

    Parameters
    ----------
    input_size : int
        Target image size. Default 224.

    Returns
    -------
    torchvision.transforms.Compose
    """
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_train_transform(input_size: int = INPUT_SIZE) -> transforms.Compose:
    """
    Return the training-time image transform with augmentation.

    Matches `train_transform_v4` from the training notebook.
    Used only if satellite classifier is unfrozen for fine-tuning.
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(input_size, scale=(0.88, 1.0)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_image(pil_image, train: bool = False, input_size: int = INPUT_SIZE) -> torch.Tensor:
    """
    Preprocess a single PIL image to a (1, 3, H, W) tensor.

    Parameters
    ----------
    pil_image : PIL.Image
        Input image. Will be converted to RGB internally.
    train : bool
        If True, apply training augmentation. Default False.
    input_size : int
        Target size in pixels.

    Returns
    -------
    torch.Tensor
        Shape (1, 3, input_size, input_size).
    """
    from PIL import Image as PilImageModule
    if hasattr(pil_image, "convert"):
        pil_image = pil_image.convert("RGB")

    transform = get_train_transform(input_size) if train else get_inference_transform(input_size)
    tensor = transform(pil_image)
    return tensor.unsqueeze(0)  # add batch dimension


def denormalise_image(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reverse ImageNet normalisation for visualisation.

    Parameters
    ----------
    tensor : torch.Tensor
        Shape (3, H, W) or (batch, 3, H, W).

    Returns
    -------
    torch.Tensor
        Values clipped to [0, 1].
    """
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device)
    std  = torch.tensor(IMAGENET_STD,  dtype=tensor.dtype, device=tensor.device)

    if tensor.dim() == 3:
        mean = mean.view(3, 1, 1)
        std  = std.view(3, 1, 1)
    else:
        mean = mean.view(1, 3, 1, 1)
        std  = std.view(1, 3, 1, 1)

    return (tensor * std + mean).clamp(0, 1)
