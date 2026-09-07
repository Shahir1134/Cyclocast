"""
backend/models.py
=================
SQLAlchemy ORM models for the CycloCast platform:
- Cyclone
- ForecastRun
- TrackPoint
- UncertaintyCone
- BestTrackPoint (for Replay & Ground-Truth validation)
- SatelliteOverlay
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    Text,
)
from sqlalchemy.orm import relationship
from backend.database import Base


class Cyclone(Base):
    __tablename__ = "cyclones"

    id = Column(String(64), primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    basin = Column(String(50), nullable=False)  # "Bay of Bengal", "Arabian Sea", etc.
    season = Column(Integer, nullable=False)     # Year, e.g. 2020, 2023, 2026
    status = Column(String(20), default="active", index=True)  # "active" or "historical"
    category = Column(String(50), default="Cyclonic Storm")
    max_wind_kt = Column(Float, default=0.0)
    min_pressure_hpa = Column(Float, default=1010.0)
    current_lat = Column(Float, nullable=False)
    current_lon = Column(Float, nullable=False)
    advisory = Column(Text, nullable=True)
    movement = Column(String(100), nullable=True)  # e.g. "Moving NNE at 15 km/h"
    last_updated = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    forecast_runs = relationship("ForecastRun", back_populates="cyclone", cascade="all, delete-orphan")
    best_track_points = relationship("BestTrackPoint", back_populates="cyclone", cascade="all, delete-orphan")
    satellite_overlays = relationship("SatelliteOverlay", back_populates="cyclone", cascade="all, delete-orphan")


class ForecastRun(Base):
    __tablename__ = "forecast_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cyclone_id = Column(String(64), ForeignKey("cyclones.id"), nullable=False, index=True)
    run_timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    model_version = Column(String(100), default="GraphCast-ResNet18-Fusion-v1.0")
    confidence_score = Column(Float, default=0.75)

    cyclone = relationship("Cyclone", back_populates="forecast_runs")
    track_points = relationship("TrackPoint", back_populates="forecast_run", cascade="all, delete-orphan")
    uncertainty_cone = relationship("UncertaintyCone", back_populates="forecast_run", uselist=False, cascade="all, delete-orphan")


class TrackPoint(Base):
    __tablename__ = "track_points"

    id = Column(Integer, primary_key=True, autoincrement=True)
    forecast_run_id = Column(Integer, ForeignKey("forecast_runs.id"), nullable=False, index=True)
    horizon_hours = Column(Integer, nullable=False)  # 0, 6, 12, 18, 24
    timestamp = Column(DateTime, nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    wind_speed_kt = Column(Float, nullable=False)
    pressure_hpa = Column(Float, nullable=False)
    category = Column(String(50), nullable=False)

    forecast_run = relationship("ForecastRun", back_populates="track_points")


class UncertaintyCone(Base):
    __tablename__ = "uncertainty_cones"

    id = Column(Integer, primary_key=True, autoincrement=True)
    forecast_run_id = Column(Integer, ForeignKey("forecast_runs.id"), nullable=False, unique=True)
    geometry_json = Column(Text, nullable=False)  # GeoJSON Polygon string

    forecast_run = relationship("ForecastRun", back_populates="uncertainty_cone")

    @property
    def geojson(self):
        return json.loads(self.geometry_json) if self.geometry_json else None


class BestTrackPoint(Base):
    __tablename__ = "best_track_points"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cyclone_id = Column(String(64), ForeignKey("cyclones.id"), nullable=False, index=True)
    step_index = Column(Integer, nullable=False)  # Sequential index for replay
    timestamp = Column(DateTime, nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    wind_speed_kt = Column(Float, nullable=False)
    pressure_hpa = Column(Float, nullable=False)
    category = Column(String(50), nullable=False)
    source = Column(String(50), default="IMD/IBTrACS")

    cyclone = relationship("Cyclone", back_populates="best_track_points")


class SatelliteOverlay(Base):
    __tablename__ = "satellite_overlays"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cyclone_id = Column(String(64), ForeignKey("cyclones.id"), nullable=False, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    image_url = Column(String(255), nullable=False)
    bounds_json = Column(Text, nullable=False)  # [[min_lat, min_lon], [max_lat, max_lon]]

    cyclone = relationship("Cyclone", back_populates="satellite_overlays")

    @property
    def bounds(self):
        return json.loads(self.bounds_json) if self.bounds_json else None
