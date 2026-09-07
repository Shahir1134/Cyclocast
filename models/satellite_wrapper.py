"""
models/satellite_wrapper.py
============================
Non-invasive wrapper around the trained ResNet18 cyclone classifier.

Architecture (matches Cyclone.ipynb exactly):
    ResNet18 (ImageNet pre-trained backbone)
    └── layer4  (fine-tuned)
    └── avgpool → 512-d vector  ← penultimate embedding extracted here
    └── fc: Sequential(Dropout(0.4), Linear(512, 5))

The wrapper:
  1. Reconstructs the identical architecture.
  2. Loads weights from `cyclone.pth` without modifying them.
  3. Registers a forward hook on `avgpool` to capture the 512-d embedding.
  4. Exposes `cyclone_probability`, `predicted_class`, and `satellite_embedding`
     through a single `forward()` call.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class labels (must match training notebook exactly)
# ---------------------------------------------------------------------------
CLASS_NAMES: List[str] = [
    "Low",
    "Moderate",
    "Severe",
    "Very Severe",
    "Extreme",
]


class SatelliteWrapper(nn.Module):
    """
    Frozen ResNet18 cyclone classifier with penultimate embedding extraction.

    Parameters
    ----------
    model_path : str | Path
        Path to the saved `cyclone.pth` state_dict file.
    num_classes : int
        Number of output classes. Default 5 (matches training).
    device : torch.device | str | None
        Computation device. Pass None to auto-detect.

    Attributes
    ----------
    embedding_dim : int
        Dimension of the satellite embedding (512 for ResNet18).
    class_names : list[str]
        Human-readable class labels.
    """

    embedding_dim: int = 512

    def __init__(
        self,
        model_path: str | Path,
        num_classes: int = 5,
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__()

        self.model_path = Path(model_path)
        self.num_classes = num_classes
        self.class_names = CLASS_NAMES[:num_classes]

        # ── Device ──────────────────────────────────────────────────────────
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # ── Build identical architecture to the training notebook ────────────
        self._model = self._build_model()

        # ── Load weights ─────────────────────────────────────────────────────
        self._load_weights()

        # ── Freeze all parameters ────────────────────────────────────────────
        for param in self._model.parameters():
            param.requires_grad = False

        self._model.eval()
        self._model.to(self.device)

        # ── Forward hook storage ─────────────────────────────────────────────
        self._embedding_buffer: Optional[torch.Tensor] = None
        self._hook_handle = self._model.avgpool.register_forward_hook(
            self._avgpool_hook
        )

        logger.info(
            "SatelliteWrapper loaded from '%s' on device '%s'.",
            self.model_path,
            self.device,
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _build_model(self) -> nn.Module:
        """Reconstruct the exact architecture used during training."""
        model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Replace classifier head (matches notebook Cell 18)
        num_features = model.fc.in_features  # 512
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, self.num_classes),
        )
        return model

    def _load_weights(self) -> None:
        """Load state_dict from .pth file."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Satellite model weights not found at '{self.model_path}'."
            )

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=True,
        )

        # Handle both raw state_dict and wrapped checkpoint dicts
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        self._model.load_state_dict(state_dict, strict=True)
        logger.info("Satellite classifier weights loaded successfully.")

    def _avgpool_hook(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        """Captures the avgpool output (512-d embedding) during forward pass."""
        # avgpool output shape: (batch, 512, 1, 1)  → flatten to (batch, 512)
        self._embedding_buffer = output.detach().flatten(start_dim=1)

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def forward(
        self, image_tensor: torch.Tensor
    ) -> Dict[str, torch.Tensor | List[str]]:
        """
        Run the satellite classifier and return classification outputs
        together with the penultimate embedding.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Pre-processed image tensor of shape ``(batch, 3, 224, 224)``.
            Use ``data.preprocessing.get_inference_transform()`` to prepare
            raw PIL images.

        Returns
        -------
        dict with keys:
            ``logits``          – raw classifier logits   (batch, num_classes)
            ``probabilities``   – softmax probabilities   (batch, num_classes)
            ``cyclone_probability`` – max class prob      (batch,)
            ``predicted_class_id``  – argmax class index  (batch,)
            ``predicted_class_name`` – list[str]
            ``satellite_embedding``  – 512-d embedding    (batch, 512)
        """
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            logits = self._model(image_tensor)  # also fires the hook

        probs = F.softmax(logits, dim=-1)  # (batch, 5)
        cyclone_prob, pred_class_id = probs.max(dim=-1)

        pred_names = [self.class_names[idx.item()] for idx in pred_class_id]

        # Hook populates self._embedding_buffer
        embedding = self._embedding_buffer  # (batch, 512)

        return {
            "logits": logits,
            "probabilities": probs,
            "cyclone_probability": cyclone_prob,
            "predicted_class_id": pred_class_id,
            "predicted_class_name": pred_names,
            "satellite_embedding": embedding,
        }

    def get_embedding(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Convenience method — returns only the satellite embedding tensor."""
        result = self.forward(image_tensor)
        return result["satellite_embedding"]

    def remove_hook(self) -> None:
        """Remove the forward hook (call when done to avoid memory leaks)."""
        self._hook_handle.remove()
        logger.debug("Forward hook removed from avgpool.")

    def __del__(self) -> None:
        try:
            self.remove_hook()
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Unfreeze support (for optional fine-tuning later)
    # -----------------------------------------------------------------------

    def unfreeze_head(self) -> None:
        """Unfreeze the FC head for optional fine-tuning."""
        for param in self._model.fc.parameters():
            param.requires_grad = True
        logger.info("Satellite wrapper FC head unfrozen.")

    def unfreeze_layer4(self) -> None:
        """Unfreeze layer4 + FC for deeper fine-tuning."""
        for param in self._model.layer4.parameters():
            param.requires_grad = True
        self.unfreeze_head()
        logger.info("Satellite wrapper layer4 + FC unfrozen.")
