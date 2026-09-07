"""
utils/visualization.py
========================
Visualisation utilities that produce dashboard-ready data structures.

All functions return plain Python dicts / lists — no matplotlib or
frontend dependency. Designed to be consumed directly by Streamlit or
a React/FastAPI backend.

Functions
---------
build_track_geojson          – GeoJSON LineString for map rendering
build_intensity_timeseries   – Chart-ready intensity data
build_atmospheric_summary    – Summary card data per horizon
build_full_dashboard_payload – Combines all of the above
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

HORIZONS = ["6h", "12h", "18h", "24h"]
HORIZON_LABELS = {
    "6h":  "+6 hours",
    "12h": "+12 hours",
    "18h": "+18 hours",
    "24h": "+24 hours",
}

# Saffir-Simpson–inspired intensity category mapping
INTENSITY_CATEGORIES = [
    ("Tropical Depression", 0,   33),
    ("Tropical Storm",      34,  63),
    ("Category 1",          64,  82),
    ("Category 2",          83,  95),
    ("Category 3",          96, 112),
    ("Category 4",         113, 136),
    ("Category 5",         137, 9999),
]


def _wind_category(wind_kt: float) -> str:
    for name, lo, hi in INTENSITY_CATEGORIES:
        if lo <= wind_kt <= hi:
            return name
    return "Unknown"


# ---------------------------------------------------------------------------
# GeoJSON
# ---------------------------------------------------------------------------

def build_track_geojson(forecast_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a GeoJSON FeatureCollection containing:
      - A LineString connecting current location → all forecast positions
      - Point markers at each forecast position with intensity metadata

    Parameters
    ----------
    forecast_result : dict
        Output from ``CycloneForecaster.forecast()``.

    Returns
    -------
    dict
        GeoJSON FeatureCollection.
    """
    if not forecast_result.get("cyclone_detected"):
        return _empty_geojson()

    current = forecast_result["current_location"]
    fcast   = forecast_result.get("forecast") or {}
    conf    = forecast_result.get("forecast_confidence", 0)
    cls_    = forecast_result.get("predicted_class", "Unknown")

    # Build coordinate sequence: current → 6h → 12h → 18h → 24h
    coords = [[current["lon"], current["lat"]]]
    for h in HORIZONS:
        if h in fcast:
            coords.append([fcast[h]["lon"], fcast[h]["lat"]])

    features = []

    # Track LineString
    features.append({
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": coords,
        },
        "properties": {
            "name": "Predicted Track",
            "forecast_confidence": conf,
            "satellite_class": cls_,
        },
    })

    # Current position marker
    features.append({
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [current["lon"], current["lat"]],
        },
        "properties": {
            "label": "Current",
            "time_offset": "0h",
            "cyclone_probability": forecast_result.get("cyclone_probability"),
        },
    })

    # Forecast position markers
    for h in HORIZONS:
        if h not in fcast:
            continue
        pt = fcast[h]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [pt["lon"], pt["lat"]],
            },
            "properties": {
                "label": HORIZON_LABELS.get(h, h),
                "time_offset": h,
                "lat": pt["lat"],
                "lon": pt["lon"],
                "wind_kt": pt.get("wind"),
                "pressure_hPa": pt.get("pressure"),
                "intensity_category": _wind_category(pt.get("wind", 0)),
            },
        })

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def _empty_geojson() -> Dict[str, Any]:
    return {"type": "FeatureCollection", "features": []}


# ---------------------------------------------------------------------------
# Intensity timeseries
# ---------------------------------------------------------------------------

def build_intensity_timeseries(forecast_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build chart-ready intensity data for wind speed and central pressure.

    Returns
    -------
    dict
        ``{
            "labels": ["0h", "6h", "12h", "18h", "24h"],
            "wind_kt": [current_wind?, 85, 92, 100, 108],
            "pressure_hPa": [current_pres?, 982, 975, 968, 960],
            "categories": ["Cat 2", "Cat 3", "Cat 3", "Cat 4"],
           }``
    """
    fcast = forecast_result.get("forecast") or {}

    labels = ["0h"] + HORIZONS
    wind_series: List[Optional[float]] = [None]  # current wind unknown without obs
    pres_series: List[Optional[float]] = [None]
    categories:  List[Optional[str]]   = [None]

    for h in HORIZONS:
        if h in fcast:
            w = fcast[h].get("wind")
            p = fcast[h].get("pressure")
            wind_series.append(w)
            pres_series.append(p)
            categories.append(_wind_category(w) if w is not None else None)
        else:
            wind_series.append(None)
            pres_series.append(None)
            categories.append(None)

    return {
        "labels":       labels,
        "wind_kt":      wind_series,
        "pressure_hPa": pres_series,
        "categories":   categories,
        "detected":     forecast_result.get("cyclone_detected", False),
    }


# ---------------------------------------------------------------------------
# Atmospheric summary cards
# ---------------------------------------------------------------------------

def build_atmospheric_summary(forecast_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build a list of summary card dicts — one per forecast horizon.

    Each card contains key values suitable for display in a dashboard grid.

    Returns
    -------
    list of dict
        E.g.::

            [
                {
                    "horizon": "6h",
                    "label": "+6 hours",
                    "lat": 20.8,
                    "lon": 88.6,
                    "wind_kt": 85.0,
                    "pressure_hPa": 982.0,
                    "category": "Category 2",
                },
                ...
            ]
    """
    fcast = forecast_result.get("forecast") or {}
    cards = []

    for h in HORIZONS:
        if h not in fcast:
            continue
        pt = fcast[h]
        wind = pt.get("wind")
        cards.append({
            "horizon":      h,
            "label":        HORIZON_LABELS.get(h, h),
            "lat":          pt.get("lat"),
            "lon":          pt.get("lon"),
            "wind_kt":      wind,
            "pressure_hPa": pt.get("pressure"),
            "category":     _wind_category(wind) if wind is not None else None,
        })

    return cards


# ---------------------------------------------------------------------------
# Full dashboard payload
# ---------------------------------------------------------------------------

def build_full_dashboard_payload(forecast_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine all visualisation components into a single JSON-serialisable dict.

    This is the recommended response format for a FastAPI/Streamlit backend.

    Parameters
    ----------
    forecast_result : dict
        Output from ``CycloneForecaster.forecast()``.

    Returns
    -------
    dict
        Keys: ``"summary"``, ``"track_geojson"``, ``"intensity_timeseries"``,
              ``"atmospheric_cards"``.
    """
    return {
        "summary": {
            "cyclone_detected":    forecast_result.get("cyclone_detected"),
            "cyclone_probability": forecast_result.get("cyclone_probability"),
            "predicted_class":     forecast_result.get("predicted_class"),
            "current_location":    forecast_result.get("current_location"),
            "forecast_confidence": forecast_result.get("forecast_confidence"),
        },
        "track_geojson":        build_track_geojson(forecast_result),
        "intensity_timeseries": build_intensity_timeseries(forecast_result),
        "atmospheric_cards":    build_atmospheric_summary(forecast_result),
    }
