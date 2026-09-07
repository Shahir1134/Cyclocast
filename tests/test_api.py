"""
tests/test_api.py
=================
Automated route and integration tests for CycloCast FastAPI backend.
"""

import sys
from pathlib import Path

# Ensure project root is available
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify backend health endpoint."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"


def test_list_cyclones():
    """Verify listing cyclones with filtering."""
    resp = client.get("/api/cyclones")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 3  # Amphan, Biparjoy, Live watch

    # Test filtering by status
    active_resp = client.get("/api/cyclones?status=active")
    assert active_resp.status_code == 200
    active_data = active_resp.json()
    assert all(c["status"] == "active" for c in active_data)

    hist_resp = client.get("/api/cyclones?status=historical")
    assert hist_resp.status_code == 200
    hist_data = hist_resp.json()
    assert all(c["status"] == "historical" for c in hist_data)


def test_cyclone_detail():
    """Verify cyclone detail endpoint."""
    resp = client.get("/api/cyclones/amphan-2020")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Cyclone Amphan"
    assert data["basin"] == "Bay of Bengal"
    assert len(data["track_points"]) > 0


def test_track_geojson():
    """Verify track endpoint returns valid GeoJSON FeatureCollection."""
    resp = client.get("/api/cyclones/amphan-2020/track")
    assert resp.status_code == 200
    data = resp.json()
    assert "geojson" in data
    geojson = data["geojson"]
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) >= 2
    # Ensure point and linestring exist
    geom_types = [f["geometry"]["type"] for f in geojson["features"]]
    assert "Point" in geom_types
    assert "LineString" in geom_types


def test_uncertainty_cone():
    """Verify uncertainty cone returns valid GeoJSON polygon."""
    resp = client.get("/api/cyclones/amphan-2020/uncertainty")
    assert resp.status_code == 200
    data = resp.json()
    assert data["type"] == "Feature"
    assert data["geometry"]["type"] in ("Polygon", "MultiPolygon")
    assert len(data["geometry"]["coordinates"]) > 0


def test_intensity_timeline():
    """Verify intensity timeline data structure for charts."""
    resp = client.get("/api/cyclones/amphan-2020/intensity")
    assert resp.status_code == 200
    data = resp.json()
    assert "data_points" in data
    assert len(data["data_points"]) > 0
    pt = data["data_points"][0]
    assert "wind_speed_kt" in pt
    assert "pressure_hpa" in pt
    assert "horizon_hours" in pt


def test_replay_mode_data():
    """Verify replay mode returns simultaneous actual and predicted steps with errors."""
    resp = client.get("/api/cyclones/amphan-2020/replay")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_steps"] > 0
    assert len(data["steps"]) == data["total_steps"]
    step0 = data["steps"][0]
    assert "actual_lat" in step0
    assert "predicted_lat" in step0
    assert "track_error_km" in step0
    assert "intensity_error_kt" in step0
    assert data["mean_track_error_km"] >= 0


def test_impact_exposure():
    """Verify population and infrastructure impact layer."""
    resp = client.get("/api/cyclones/amphan-2020/impact")
    assert resp.status_code == 200
    data = resp.json()
    assert "coastal_alert_level" in data
    assert "total_exposed_population" in data
    assert data["total_exposed_population"] > 0
    assert len(data["vulnerable_districts"]) > 0
    dist0 = data["vulnerable_districts"][0]
    assert "district_or_port" in dist0
    assert "risk_level" in dist0
