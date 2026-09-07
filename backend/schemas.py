"""
backend/schemas.py
==================
Pydantic response and request schemas for the CycloCast API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class CycloneBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    basin: str
    season: int
    status: str
    category: str
    max_wind_kt: float
    min_pressure_hpa: float
    current_lat: float
    current_lon: float
    advisory: Optional[str] = None
    movement: Optional[str] = None
    last_updated: Optional[datetime] = None


class CycloneSummary(CycloneBase):
    pass


class TrackPointOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    horizon_hours: int
    timestamp: datetime
    lat: float
    lon: float
    wind_speed_kt: float
    pressure_hpa: float
    category: str


class TrackGeoJSON(BaseModel):
    type: str = "FeatureCollection"
    features: List[Dict[str, Any]]


class UncertaintyConeOut(BaseModel):
    forecast_run_id: int
    type: str = "Feature"
    geometry: Dict[str, Any]
    properties: Dict[str, Any] = Field(default_factory=dict)


class IntensityPointOut(BaseModel):
    horizon_hours: int
    label: str
    timestamp: datetime
    wind_speed_kt: float
    wind_speed_kmh: float
    pressure_hpa: float
    category: str


class IntensityTimelineOut(BaseModel):
    cyclone_id: str
    cyclone_name: str
    data_points: List[IntensityPointOut]


class ReplayStep(BaseModel):
    step_index: int
    timestamp: datetime
    actual_lat: float
    actual_lon: float
    actual_wind_kt: float
    actual_pressure_hpa: float
    actual_category: str
    predicted_lat: float
    predicted_lon: float
    predicted_wind_kt: float
    predicted_pressure_hpa: float
    predicted_category: str
    track_error_km: float
    intensity_error_kt: float


class ReplayDataOut(BaseModel):
    cyclone_id: str
    cyclone_name: str
    total_steps: int
    steps: List[ReplayStep]
    mean_track_error_km: float
    mean_intensity_error_kt: float


class CoastalVulnerability(BaseModel):
    district_or_port: str
    state_or_country: str
    estimated_population: int
    risk_level: str  # "High", "Extreme", "Moderate"
    distance_to_track_km: float


class ImpactSummaryOut(BaseModel):
    cyclone_id: str
    cyclone_name: str
    landfall_estimate: Optional[str] = None
    coastal_alert_level: str
    total_exposed_population: int
    vulnerable_districts: List[CoastalVulnerability]
    infrastructure_alerts: List[str]


class SatelliteOverlayOut(BaseModel):
    cyclone_id: str
    timestamp: datetime
    image_url: str
    bounds: List[List[float]]  # [[min_lat, min_lon], [max_lat, max_lon]]


class CycloneDetail(CycloneBase):
    latest_run_id: Optional[int] = None
    confidence_score: Optional[float] = None
    track_points: List[TrackPointOut] = []
    satellite_overlay: Optional[SatelliteOverlayOut] = None
