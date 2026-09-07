"""
tests/test_fusion_pipeline.py
==============================
Unit tests for the end-to-end fusion pipeline.

Tests each component's output shape and verifies the full pipeline
forward pass using entirely synthetic / random data.
No real model files or GraphCast installation required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.atmospheric_encoder import AtmosphericEncoder
from models.fusion_model import MultimodalFusionModel
from models.track_head import TrackHead
from models.intensity_head import IntensityHead, WIND_MEAN, WIND_STD, PRES_MEAN, PRES_STD
from models.confidence_head import ConfidenceHead
from training.losses import FusionLoss, TrackLoss, IntensityLoss, ConfidenceLoss
from training.metrics import track_error_km, intensity_mae, MetricsAccumulator
from utils.geo import haversine_distance, haversine_batch, encode_location
from utils.feature_extraction import AtmosphericFeatureExtractor
from utils.visualization import (
    build_track_geojson,
    build_intensity_timeseries,
    build_atmospheric_summary,
    build_full_dashboard_payload,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BATCH = 4
ATM_EMBED_DIM = 128
SAT_EMBED_DIM = 512
SHARED_DIM = 256
N_HORIZONS = 4


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def atm_encoder():
    return AtmosphericEncoder(input_dim=1024, hidden_dim=256, output_dim=ATM_EMBED_DIM)


@pytest.fixture
def fusion_model():
    return MultimodalFusionModel(
        satellite_embed_dim=SAT_EMBED_DIM,
        atm_embed_dim=ATM_EMBED_DIM,
        sat_proj_dim=128,
        atm_proj_dim=128,
        shared_hidden_dim=SHARED_DIM,
        use_location_embedding=True,
        location_embed_dim=16,
    )


@pytest.fixture
def track_head(fusion_model):
    return TrackHead(input_dim=fusion_model.output_dim)


@pytest.fixture
def intensity_head(fusion_model):
    return IntensityHead(input_dim=fusion_model.output_dim)


@pytest.fixture
def confidence_head(fusion_model):
    return ConfidenceHead(input_dim=fusion_model.output_dim)


@pytest.fixture
def fake_inputs():
    return {
        "sat_emb": torch.randn(BATCH, SAT_EMBED_DIM),
        "sat_prob": torch.rand(BATCH, 1),
        "atm_embeds": [torch.randn(BATCH, ATM_EMBED_DIM) for _ in range(N_HORIZONS)],
        "loc_enc": torch.randn(BATCH, 4),
    }


# ---------------------------------------------------------------------------
# AtmosphericEncoder
# ---------------------------------------------------------------------------

class TestAtmosphericEncoder:
    def test_output_shape(self, atm_encoder):
        x = torch.randn(BATCH, 1024)
        out = atm_encoder(x)
        assert out.shape == (BATCH, ATM_EMBED_DIM)

    def test_encode_horizons_length(self, atm_encoder):
        feats = [torch.randn(BATCH, 1024) for _ in range(4)]
        embeds = atm_encoder.encode_horizons(feats)
        assert len(embeds) == 4
        for e in embeds:
            assert e.shape == (BATCH, ATM_EMBED_DIM)

    def test_trainable_params(self, atm_encoder):
        n = sum(1 for p in atm_encoder.parameters() if p.requires_grad)
        assert n > 0


# ---------------------------------------------------------------------------
# MultimodalFusionModel
# ---------------------------------------------------------------------------

class TestFusionModel:
    def test_output_shape(self, fusion_model, fake_inputs):
        out = fusion_model(
            satellite_embedding=fake_inputs["sat_emb"],
            cyclone_probability=fake_inputs["sat_prob"],
            atm_embeddings=fake_inputs["atm_embeds"],
            location_encoding=fake_inputs["loc_enc"],
        )
        assert out.shape == (BATCH, SHARED_DIM)

    def test_output_shape_no_location(self):
        model = MultimodalFusionModel(
            satellite_embed_dim=SAT_EMBED_DIM,
            atm_embed_dim=ATM_EMBED_DIM,
            use_location_embedding=False,
        )
        sat_emb  = torch.randn(BATCH, SAT_EMBED_DIM)
        sat_prob = torch.rand(BATCH, 1)
        atm_embeds = [torch.randn(BATCH, ATM_EMBED_DIM) for _ in range(4)]
        out = model(sat_emb, sat_prob, atm_embeds, location_encoding=None)
        assert out.shape == (BATCH, model.output_dim)

    def test_accepts_1d_cyclone_prob(self, fusion_model, fake_inputs):
        sat_prob_1d = fake_inputs["sat_prob"].squeeze(-1)  # (batch,)
        out = fusion_model(
            satellite_embedding=fake_inputs["sat_emb"],
            cyclone_probability=sat_prob_1d,
            atm_embeddings=fake_inputs["atm_embeds"],
            location_encoding=fake_inputs["loc_enc"],
        )
        assert out.shape == (BATCH, SHARED_DIM)


# ---------------------------------------------------------------------------
# Prediction Heads
# ---------------------------------------------------------------------------

class TestTrackHead:
    def test_output_shape(self, track_head):
        x = torch.randn(BATCH, SHARED_DIM)
        out = track_head(x)
        assert out.shape == (BATCH, 4, 2)

    def test_output_is_finite(self, track_head):
        x = torch.randn(BATCH, SHARED_DIM)
        out = track_head(x)
        assert torch.isfinite(out).all()


class TestIntensityHead:
    def test_output_shape(self, intensity_head):
        x = torch.randn(BATCH, SHARED_DIM)
        out = intensity_head(x)
        assert out.shape == (BATCH, 4, 2)

    def test_denormalise(self):
        x = torch.zeros(BATCH, 4, 2)
        head = IntensityHead(SHARED_DIM, denormalise_output=True)
        inp = torch.randn(BATCH, SHARED_DIM)
        out = head(inp)
        # After denorm, wind ≈ WIND_MEAN ± WIND_STD range
        # Just check finite
        assert torch.isfinite(out).all()

    def test_normalise_targets(self):
        nw, np_ = IntensityHead.normalise_targets(WIND_MEAN, PRES_MEAN)
        assert abs(nw) < 1e-5
        assert abs(np_) < 1e-5


class TestConfidenceHead:
    def test_output_shape(self, confidence_head):
        x = torch.randn(BATCH, SHARED_DIM)
        out = confidence_head(x)
        assert out.shape == (BATCH, 1)

    def test_output_in_01(self, confidence_head):
        x = torch.randn(BATCH, SHARED_DIM)
        out = confidence_head(x)
        assert (out >= 0).all() and (out <= 1).all()

    def test_confidence_target_proxy(self):
        errors = torch.tensor([0.0, 100.0, 200.0, 500.0])
        targets = ConfidenceHead.compute_confidence_target(errors, sigma_km=200.0)
        assert targets.shape == (4, 1)
        # Zero error → confidence = 1
        assert abs(targets[0, 0].item() - 1.0) < 1e-5
        # Large error → low confidence
        assert targets[-1, 0].item() < 0.1


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class TestLosses:
    def test_track_loss_zero_for_perfect_pred(self):
        t = torch.randn(BATCH, 4, 2)
        loss = TrackLoss()(t, t)
        assert loss.item() < 1e-6

    def test_intensity_loss_positive(self):
        pred = torch.randn(BATCH, 4, 2)
        true = torch.randn(BATCH, 4, 2)
        loss = IntensityLoss()(pred, true)
        assert loss.item() >= 0

    def test_fusion_loss_keys(self):
        loss_fn = FusionLoss()
        pred_track = torch.randn(BATCH, 4, 2)
        pred_int   = torch.randn(BATCH, 4, 2)
        pred_conf  = torch.rand(BATCH, 1)
        true_track = torch.randn(BATCH, 4, 2)
        true_int   = torch.randn(BATCH, 4, 2)
        lat = torch.full((BATCH,), 20.0)
        lon = torch.full((BATCH,), 88.0)
        losses = loss_fn(pred_track, pred_int, pred_conf, true_track, true_int, lat, lon)
        assert "loss_total" in losses
        assert "loss_track" in losses
        assert "loss_intensity" in losses
        assert "loss_confidence" in losses

    def test_fusion_loss_total_positive(self):
        loss_fn = FusionLoss()
        pred_track = torch.randn(BATCH, 4, 2)
        true_track = torch.randn(BATCH, 4, 2)
        pred_int   = torch.randn(BATCH, 4, 2)
        true_int   = torch.randn(BATCH, 4, 2)
        pred_conf  = torch.rand(BATCH, 1)
        lat = torch.full((BATCH,), 20.0)
        lon = torch.full((BATCH,), 88.0)
        losses = loss_fn(pred_track, pred_int, pred_conf, true_track, true_int, lat, lon)
        assert losses["loss_total"].item() >= 0


# ---------------------------------------------------------------------------
# Geo utilities
# ---------------------------------------------------------------------------

class TestGeoUtils:
    def test_haversine_distance_same_point(self):
        d = haversine_distance(20.0, 88.0, 20.0, 88.0)
        assert d < 1e-6

    def test_haversine_distance_known(self):
        # Mumbai (~18.97N, 72.82E) to Chennai (~13.09N, 80.27E) ≈ 1062 km
        d = haversine_distance(18.97, 72.82, 13.09, 80.27)
        assert 1000 < d < 1120

    def test_haversine_batch_shape(self):
        pred = torch.randn(BATCH, 4, 2)
        true = torch.randn(BATCH, 4, 2)
        errors = haversine_batch(pred, true)
        assert errors.shape == (BATCH, 4)
        assert (errors >= 0).all()

    def test_encode_location_shape(self):
        enc = encode_location(20.0, 88.0)
        assert enc.shape == (4,)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def _make_track_data(self):
        lat = torch.full((BATCH,), 20.0)
        lon = torch.full((BATCH,), 88.0)
        pred = torch.zeros(BATCH, 4, 2)
        true = torch.zeros(BATCH, 4, 2)
        return pred, true, lat, lon

    def test_track_error_perfect(self):
        pred, true, lat, lon = self._make_track_data()
        result = track_error_km(pred, true, lat, lon)
        for h in ["6h", "12h", "18h", "24h"]:
            assert result[h] < 1e-3
        assert result["mean"] < 1e-3

    def test_track_error_keys(self):
        pred, true, lat, lon = self._make_track_data()
        result = track_error_km(pred, true, lat, lon)
        for key in ["6h", "12h", "18h", "24h", "mean"]:
            assert key in result

    def test_intensity_mae_keys(self):
        pred = torch.randn(BATCH, 4, 2)
        true = torch.randn(BATCH, 4, 2)
        result = intensity_mae(pred, true)
        assert "wind_mae_mean" in result
        assert "pressure_mae_mean" in result

    def test_accumulator(self):
        acc = MetricsAccumulator()
        lat = torch.full((BATCH,), 20.0)
        lon = torch.full((BATCH,), 88.0)
        for _ in range(3):
            acc.update(
                pred_track=torch.zeros(BATCH, 4, 2),
                pred_intensity=torch.zeros(BATCH, 4, 2),
                pred_confidence=torch.full((BATCH, 1), 0.8),
                true_track=torch.zeros(BATCH, 4, 2),
                true_intensity=torch.zeros(BATCH, 4, 2),
                current_lat=lat,
                current_lon=lon,
                batch_loss=0.5,
            )
        metrics = acc.compute()
        assert "track_error_mean" in metrics
        assert abs(metrics["track_error_mean"]) < 1e-3


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_forecast_result():
    return {
        "cyclone_detected":    True,
        "cyclone_probability": 0.94,
        "predicted_class":     "Severe",
        "current_location":    {"lat": 20.4, "lon": 88.1},
        "forecast": {
            "6h":  {"lat": 20.8, "lon": 88.6, "wind": 85.0, "pressure": 982.0},
            "12h": {"lat": 21.3, "lon": 89.2, "wind": 92.0, "pressure": 975.0},
            "18h": {"lat": 21.9, "lon": 89.9, "wind": 100.0, "pressure": 968.0},
            "24h": {"lat": 22.6, "lon": 90.7, "wind": 108.0, "pressure": 960.0},
        },
        "forecast_confidence": 0.87,
        "error": None,
    }


class TestVisualization:
    def test_geojson_type(self, mock_forecast_result):
        gj = build_track_geojson(mock_forecast_result)
        assert gj["type"] == "FeatureCollection"
        assert len(gj["features"]) > 0

    def test_geojson_has_linestring(self, mock_forecast_result):
        gj = build_track_geojson(mock_forecast_result)
        types = [f["geometry"]["type"] for f in gj["features"]]
        assert "LineString" in types

    def test_geojson_no_cyclone_detected(self):
        result = {"cyclone_detected": False}
        gj = build_track_geojson(result)
        assert gj["features"] == []

    def test_intensity_timeseries_labels(self, mock_forecast_result):
        ts = build_intensity_timeseries(mock_forecast_result)
        assert ts["labels"] == ["0h", "6h", "12h", "18h", "24h"]
        assert len(ts["wind_kt"]) == 5

    def test_atmospheric_cards_count(self, mock_forecast_result):
        cards = build_atmospheric_summary(mock_forecast_result)
        assert len(cards) == 4
        assert cards[0]["horizon"] == "6h"

    def test_full_payload_keys(self, mock_forecast_result):
        payload = build_full_dashboard_payload(mock_forecast_result)
        assert "summary" in payload
        assert "track_geojson" in payload
        assert "intensity_timeseries" in payload
        assert "atmospheric_cards" in payload


# ---------------------------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------------------------

class TestAtmosphericFeatureExtractor:
    def test_extract_from_dict(self):
        import numpy as np
        crop = {
            "mean_sea_level_pressure": np.random.randn(81, 81).astype(np.float32),
            "10m_u_component_of_wind": np.random.randn(81, 81).astype(np.float32),
        }
        extractor = AtmosphericFeatureExtractor(
            surface_vars=["mean_sea_level_pressure", "10m_u_component_of_wind"],
            pressure_vars=[],
            spatial_size=81,
        )
        feat = extractor.extract(crop)
        assert feat.ndim == 1
        assert feat.shape[0] == 2 * 81 * 81

    def test_extract_tensor_type(self):
        import numpy as np
        import torch
        crop = {"mean_sea_level_pressure": np.random.randn(81, 81).astype(np.float32)}
        extractor = AtmosphericFeatureExtractor(
            surface_vars=["mean_sea_level_pressure"],
            pressure_vars=[],
            spatial_size=81,
        )
        t = extractor.extract_tensor(crop)
        assert isinstance(t, torch.Tensor)
        assert t.dtype == torch.float32
