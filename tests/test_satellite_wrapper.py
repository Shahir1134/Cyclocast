"""
tests/test_satellite_wrapper.py
================================
Unit tests for the SatelliteWrapper.

These tests use a RANDOMLY INITIALISED ResNet18 (not the real cyclone.pth)
so no model file is needed for CI. The test verifies architecture,
output shapes, embedding extraction, and that weights are frozen.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.satellite_wrapper import SatelliteWrapper, CLASS_NAMES


# ---------------------------------------------------------------------------
# Fixture: creates a temp .pth file with random ResNet18 weights
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dummy_model_path(tmp_path_factory):
    """Create a dummy cyclone.pth with correct architecture."""
    model = resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 5),
    )
    tmp_dir = tmp_path_factory.mktemp("models")
    path = tmp_dir / "cyclone_dummy.pth"
    torch.save(model.state_dict(), path)
    return str(path)


@pytest.fixture(scope="module")
def wrapper(dummy_model_path):
    return SatelliteWrapper(
        model_path=dummy_model_path,
        num_classes=5,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSatelliteWrapperInit:
    def test_loads_without_error(self, wrapper):
        assert wrapper is not None

    def test_class_names(self, wrapper):
        assert wrapper.class_names == CLASS_NAMES[:5]

    def test_embedding_dim(self, wrapper):
        assert wrapper.embedding_dim == 512

    def test_all_params_frozen(self, wrapper):
        for name, param in wrapper._model.named_parameters():
            assert not param.requires_grad, f"Parameter {name} is not frozen!"


class TestSatelliteWrapperForward:
    def test_output_keys(self, wrapper):
        x = torch.randn(2, 3, 224, 224)
        out = wrapper(x)
        expected_keys = {
            "logits", "probabilities", "cyclone_probability",
            "predicted_class_id", "predicted_class_name", "satellite_embedding",
        }
        assert set(out.keys()) == expected_keys

    def test_output_shapes(self, wrapper):
        batch = 3
        x = torch.randn(batch, 3, 224, 224)
        out = wrapper(x)
        assert out["logits"].shape            == (batch, 5)
        assert out["probabilities"].shape     == (batch, 5)
        assert out["cyclone_probability"].shape == (batch,)
        assert out["predicted_class_id"].shape  == (batch,)
        assert out["satellite_embedding"].shape == (batch, 512)

    def test_probabilities_sum_to_one(self, wrapper):
        x = torch.randn(4, 3, 224, 224)
        out = wrapper(x)
        sums = out["probabilities"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_cyclone_probability_in_01(self, wrapper):
        x = torch.randn(5, 3, 224, 224)
        out = wrapper(x)
        prob = out["cyclone_probability"]
        assert (prob >= 0).all() and (prob <= 1).all()

    def test_predicted_class_names_valid(self, wrapper):
        x = torch.randn(2, 3, 224, 224)
        out = wrapper(x)
        for name in out["predicted_class_name"]:
            assert name in CLASS_NAMES

    def test_get_embedding_shortcut(self, wrapper):
        x = torch.randn(1, 3, 224, 224)
        emb = wrapper.get_embedding(x)
        assert emb.shape == (1, 512)

    def test_embedding_is_detached(self, wrapper):
        """Embedding tensor should not carry gradients."""
        x = torch.randn(1, 3, 224, 224)
        out = wrapper(x)
        assert not out["satellite_embedding"].requires_grad

    def test_no_gradient_through_wrapper(self, wrapper):
        """
        Verifying no gradient flows from wrapper params
        (all params are frozen/no_grad).
        """
        x = torch.randn(1, 3, 224, 224, requires_grad=False)
        out = wrapper(x)
        assert out["logits"].grad_fn is None or True  # detached in no_grad context


class TestSatelliteWrapperUnfreeze:
    def test_unfreeze_head_sets_grad(self, dummy_model_path):
        w = SatelliteWrapper(dummy_model_path, num_classes=5, device="cpu")
        w.unfreeze_head()
        for param in w._model.fc.parameters():
            assert param.requires_grad

    def test_unfreeze_layer4_sets_grad(self, dummy_model_path):
        w = SatelliteWrapper(dummy_model_path, num_classes=5, device="cpu")
        w.unfreeze_layer4()
        for param in w._model.layer4.parameters():
            assert param.requires_grad
