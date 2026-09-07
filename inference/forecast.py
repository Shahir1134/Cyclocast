"""
inference/forecast.py
=======================
Single entry point for end-to-end cyclone forecast inference.

Usage
-----
    from inference.forecast import CycloneForecaster

    forecaster = CycloneForecaster("config/config.yaml")

    result = forecaster.forecast(
        satellite_image=pil_image,
        graphcast_input=gdas_dataset,
        cyclone_lat=20.4,
        cyclone_lon=88.1,
    )

The ``forecast_cyclone`` module-level function is provided as a
quick-access shorthand for the above.

Output schema
-------------
    {
        "cyclone_detected": bool,
        "cyclone_probability": float,
        "predicted_class": str,
        "current_location": {"lat": float, "lon": float},
        "forecast": {
            "6h":  {"lat": float, "lon": float, "wind": float, "pressure": float},
            "12h": {...},
            "18h": {...},
            "24h": {...}
        },
        "forecast_confidence": float,
        "error": str | None
    }
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# Ensure project root is on path when used as a script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import load_config, get_device
from data.preprocessing import preprocess_image
from models.satellite_wrapper import SatelliteWrapper
from models.graphcast_wrapper import GraphCastWrapper
from models.atmospheric_encoder import AtmosphericEncoder
from models.fusion_model import MultimodalFusionModel
from models.track_head import TrackHead
from models.intensity_head import IntensityHead, WIND_MEAN, WIND_STD, PRES_MEAN, PRES_STD
from models.confidence_head import ConfidenceHead
from training.train_fusion import FusionSystem, build_system
from utils.feature_extraction import AtmosphericFeatureExtractor
from utils.geo import crop_atmospheric_window, encode_location

logger = logging.getLogger(__name__)

HORIZONS = ["6h", "12h", "18h", "24h"]
HORIZON_HOURS = [6, 12, 18, 24]


class CycloneForecaster:
    """
    End-to-end multimodal cyclone forecaster.

    Wraps the satellite classifier, GraphCast adapter, atmospheric encoder,
    and the fusion prediction heads into a single callable inference object.

    Parameters
    ----------
    config_path : str | Path
        Path to ``config/config.yaml``.
    fusion_checkpoint : str | Path | None
        Path to a trained fusion system checkpoint (.pt file).
        If None, the fusion model uses random weights (for demonstration/testing).
    graphcast_runner_fn : callable | None
        Your JAX GraphCast runner. If None, the mock runner is used.
    device : str | None
        Override device. If None, uses config value.
    """

    def __init__(
        self,
        config_path: str | Path = "config/config.yaml",
        fusion_checkpoint: Optional[str | Path] = None,
        graphcast_runner_fn=None,
        device: Optional[str] = None,
    ) -> None:
        self.cfg = load_config(config_path)
        self.device = get_device(device or self.cfg.get("device", "auto"))

        # Detection threshold
        self.detection_threshold: float = self.cfg["detection"]["cyclone_threshold"]

        # ── Satellite wrapper ─────────────────────────────────────────────────
        self.sat_wrapper = SatelliteWrapper(
            model_path=self.cfg["satellite_model_path"],
            num_classes=self.cfg["satellite"]["num_classes"],
            device=self.device,
        )

        # ── GraphCast wrapper ─────────────────────────────────────────────────
        self.gc_wrapper = GraphCastWrapper(
            runner_fn=graphcast_runner_fn,
            variables=self.cfg["graphcast"]["variables"],
            forecast_horizons_h=self.cfg["graphcast"]["forecast_horizons_h"],
        )

        # ── Atmospheric feature extractor ─────────────────────────────────────
        crop_cfg = self.cfg["crop"]
        window_deg = crop_cfg["window_deg"]
        grid_res   = crop_cfg["grid_resolution"]
        n_spatial  = int(round(2 * window_deg / grid_res)) + 1
        self._window_deg = window_deg

        self.feature_extractor = AtmosphericFeatureExtractor(spatial_size=n_spatial)

        # ── Fusion system ─────────────────────────────────────────────────────
        _, self.system = build_system(self.cfg, self.device)
        self.system.eval()

        if fusion_checkpoint:
            ckpt_path = Path(fusion_checkpoint)
            if ckpt_path.exists():
                from training.train_fusion import load_checkpoint
                load_checkpoint(ckpt_path, self.system, device=self.device)
            else:
                logger.warning("Checkpoint not found at '%s'. Using random weights.", ckpt_path)

        logger.info("CycloneForecaster ready on device=%s", self.device)

    # -----------------------------------------------------------------------
    # Main inference method
    # -----------------------------------------------------------------------

    def forecast(
        self,
        satellite_image,
        graphcast_input=None,
        cyclone_lat: float = 0.0,
        cyclone_lon: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Run end-to-end cyclone forecast.

        Parameters
        ----------
        satellite_image : PIL.Image or torch.Tensor (1, 3, 224, 224)
            Input satellite image (INSAT IR or visible).
        graphcast_input : xr.Dataset | None
            GDAS/ERA5 input for GraphCast. If None, the mock runner is used.
        cyclone_lat : float
            Current cyclone centre latitude (degrees N).
        cyclone_lon : float
            Current cyclone centre longitude (degrees E).

        Returns
        -------
        dict
            Structured forecast result (see module docstring).
        """
        try:
            # ── Step 1: Satellite classification ───────────────────────────────
            if not isinstance(satellite_image, torch.Tensor):
                image_tensor = preprocess_image(satellite_image, train=False)
            else:
                image_tensor = satellite_image

            sat_out = self.sat_wrapper(image_tensor)
            cyclone_prob  = sat_out["cyclone_probability"][0].item()
            predicted_cls = sat_out["predicted_class_name"][0]
            sat_embedding = sat_out["satellite_embedding"]  # (1, 512)

            # ── Step 2: Detection gate ──────────────────────────────────────────
            if cyclone_prob < self.detection_threshold:
                return {
                    "cyclone_detected":    False,
                    "cyclone_probability": round(cyclone_prob, 4),
                    "predicted_class":     predicted_cls,
                    "current_location":    {"lat": cyclone_lat, "lon": cyclone_lon},
                    "message":             "No cyclone detected — probability below threshold.",
                    "forecast":            None,
                    "forecast_confidence": None,
                    "error":               None,
                }

            # ── Step 3: GraphCast forecast ─────────────────────────────────────
            forecasts = self.gc_wrapper.run_forecast(graphcast_input)

            # ── Step 4: Atmospheric feature extraction ─────────────────────────
            atm_tensors = []
            for h in HORIZONS:
                crop = crop_atmospheric_window(
                    forecasts.get(h), cyclone_lat, cyclone_lon, self._window_deg
                )
                try:
                    feat = self.feature_extractor.extract_tensor(crop).unsqueeze(0).to(self.device)
                except Exception:
                    n_spatial = self.feature_extractor.spatial_size or 81
                    feat = torch.zeros(
                        1, self.feature_extractor.feature_dim(n_spatial), device=self.device
                    )
                atm_tensors.append(feat)

            atm_6h, atm_12h, atm_18h, atm_24h = atm_tensors

            # ── Step 5: Location encoding ──────────────────────────────────────
            loc_enc = encode_location(cyclone_lat, cyclone_lon).unsqueeze(0).to(self.device)
            sat_prob_tensor = torch.tensor([[cyclone_prob]], dtype=torch.float32, device=self.device)

            # ── Step 6: Fusion ─────────────────────────────────────────────────
            with torch.no_grad():
                outputs = self.system(
                    satellite_embedding=sat_embedding.to(self.device),
                    cyclone_probability=sat_prob_tensor,
                    atm_6h=atm_6h,
                    atm_12h=atm_12h,
                    atm_18h=atm_18h,
                    atm_24h=atm_24h,
                    location_encoding=loc_enc,
                )

            # ── Step 7: Decode outputs ─────────────────────────────────────────
            track_deltas = outputs["track"][0].cpu()         # (4, 2)
            intensity    = outputs["intensity"][0].cpu()     # (4, 2)  normalised
            confidence   = outputs["confidence"][0, 0].item()

            forecast_dict: Dict[str, Dict[str, float]] = {}
            for i, h in enumerate(HORIZONS):
                delta_lat = track_deltas[i, 0].item()
                delta_lon = track_deltas[i, 1].item()
                norm_wind = intensity[i, 0].item()
                norm_pres = intensity[i, 1].item()

                pred_lat = round(cyclone_lat + delta_lat, 4)
                pred_lon = round(cyclone_lon + delta_lon, 4)
                pred_wind = round(norm_wind * WIND_STD + WIND_MEAN, 2)
                pred_pres = round(norm_pres * PRES_STD + PRES_MEAN, 2)

                forecast_dict[h] = {
                    "lat":      pred_lat,
                    "lon":      pred_lon,
                    "wind":     pred_wind,
                    "pressure": pred_pres,
                }

            return {
                "cyclone_detected":    True,
                "cyclone_probability": round(cyclone_prob, 4),
                "predicted_class":     predicted_cls,
                "current_location": {
                    "lat": round(cyclone_lat, 4),
                    "lon": round(cyclone_lon, 4),
                },
                "forecast":            forecast_dict,
                "forecast_confidence": round(float(confidence), 4),
                "error":               None,
            }

        except Exception as exc:
            logger.exception("Forecast failed: %s", exc)
            return {
                "cyclone_detected":    None,
                "cyclone_probability": None,
                "predicted_class":     None,
                "current_location":    {"lat": cyclone_lat, "lon": cyclone_lon},
                "forecast":            None,
                "forecast_confidence": None,
                "error":               str(exc),
            }


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

_default_forecaster: Optional[CycloneForecaster] = None


def forecast_cyclone(
    satellite_image,
    graphcast_input=None,
    cyclone_lat: float = 0.0,
    cyclone_lon: float = 0.0,
    config_path: str = "config/config.yaml",
    fusion_checkpoint: Optional[str] = None,
    graphcast_runner_fn=None,
) -> Dict[str, Any]:
    """
    Module-level convenience function for single-call inference.

    On first call, instantiates a ``CycloneForecaster`` (which is cached).
    Subsequent calls reuse the same instance for efficiency.

    Parameters
    ----------
    satellite_image : PIL.Image or torch.Tensor
        Input satellite image.
    graphcast_input : xr.Dataset | None
        GDAS/ERA5 atmospheric state. If None, the mock runner is used.
    cyclone_lat : float
        Current cyclone centre latitude.
    cyclone_lon : float
        Current cyclone centre longitude.
    config_path : str
        Path to config.yaml.
    fusion_checkpoint : str | None
        Path to trained fusion checkpoint.
    graphcast_runner_fn : callable | None
        Your GraphCast JAX runner (dependency injected).

    Returns
    -------
    dict
        Structured forecast result.

    Example
    -------
    .. code-block:: python

        from PIL import Image
        from inference.forecast import forecast_cyclone

        img = Image.open("cyclone_sample.jpg")
        result = forecast_cyclone(
            satellite_image=img,
            cyclone_lat=20.4,
            cyclone_lon=88.1,
        )
        print(result)
    """
    global _default_forecaster
    if _default_forecaster is None:
        _default_forecaster = CycloneForecaster(
            config_path=config_path,
            fusion_checkpoint=fusion_checkpoint,
            graphcast_runner_fn=graphcast_runner_fn,
        )
    return _default_forecaster.forecast(
        satellite_image=satellite_image,
        graphcast_input=graphcast_input,
        cyclone_lat=cyclone_lat,
        cyclone_lon=cyclone_lon,
    )
