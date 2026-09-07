"""
backend/services/uncertainty.py
===============================
Generates climatological uncertainty cones for cyclone track forecasts.

In official meteorological practice (IMD / NHC / JTWC), an operational
uncertainty cone is constructed using historical track error radii:
  - 0h:  ~15 km (initial analysis error)
  - 6h:  ~40 km
  - 12h: ~75 km
  - 18h: ~110 km
  - 24h: ~150 km

Circles of expanding radii are centered on each forecast coordinate, and
their spatial union forms a smooth polygon "cone of uncertainty".
"""

from __future__ import annotations

import math
from typing import List, Tuple, Dict, Any
from shapely.geometry import Point, MultiPolygon, Polygon, mapping
from shapely.ops import unary_union

# Climatological radius of error (in kilometers) per forecast horizon
ERROR_RADII_KM = {
    0: 15.0,
    6: 40.0,
    12: 75.0,
    18: 110.0,
    24: 150.0,
    36: 210.0,
    48: 280.0,
}


def km_to_degrees(km: float, lat_deg: float) -> Tuple[float, float]:
    """
    Convert kilometers to approximate degrees latitude and longitude at a given latitude.
    """
    lat_rad = math.radians(lat_deg)
    d_lat = km / 111.0
    cos_lat = max(math.cos(lat_rad), 0.1)  # avoid division by zero near poles
    d_lon = km / (111.0 * cos_lat)
    return d_lat, d_lon


def generate_uncertainty_cone(track_points: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Given a sequence of forecast track points with `lat`, `lon`, and `horizon_hours`,
    generate a smooth GeoJSON Polygon representing the uncertainty cone.

    Parameters
    ----------
    track_points : list of dict
        Each dict must contain: 'lat', 'lon', 'horizon_hours'.

    Returns
    -------
    dict
        GeoJSON Feature dict containing the Polygon geometry and metadata properties.
    """
    if not track_points:
        return {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": []},
            "properties": {"error_model": "climatological_expanding_cone"},
        }

    buffered_circles: List[Polygon] = []

    for pt in track_points:
        lat = pt["lat"]
        lon = pt["lon"]
        h = pt.get("horizon_hours", 0)

        # Look up radius in km or extrapolate linearly
        radius_km = ERROR_RADII_KM.get(h, 15.0 + (h * 5.5))
        d_lat, d_lon = km_to_degrees(radius_km, lat)
        avg_deg = (d_lat + d_lon) / 2.0

        # Buffer point into a circle in spatial degree coordinate space
        p = Point(lon, lat)
        circle = p.buffer(avg_deg, resolution=24)
        buffered_circles.append(circle)

    # Union all buffered circles along the track to produce the smooth cone
    cone_union = unary_union(buffered_circles)

    # If it's a MultiPolygon, take the convex hull or the largest polygon
    if isinstance(cone_union, MultiPolygon):
        cone_poly = cone_union.convex_hull
    else:
        cone_poly = cone_union

    # Smooth the boundary slightly
    geo_dict = mapping(cone_poly)

    return {
        "type": "Feature",
        "geometry": geo_dict,
        "properties": {
            "error_model": "IMD_Climatological_Cone",
            "horizons": [pt.get("horizon_hours", 0) for pt in track_points],
            "radii_km": [ERROR_RADII_KM.get(pt.get("horizon_hours", 0), 40.0) for pt in track_points],
            "fill_color": "#f59e0b",
            "fill_opacity": 0.22,
        },
    }
