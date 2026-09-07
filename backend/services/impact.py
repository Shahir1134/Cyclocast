"""
backend/services/impact.py
==========================
Impact and population/infrastructure exposure assessment along the
predicted cyclone track and uncertainty cone.
"""

from __future__ import annotations

import math
from typing import List, Dict, Any

# Key coastal ports and districts across the North Indian Ocean (Bay of Bengal & Arabian Sea)
COASTAL_ZONES = [
    # Bay of Bengal (India & Bangladesh)
    {"name": "Puri & Konark", "state": "Odisha, India", "lat": 19.81, "lon": 85.83, "population": 1698000, "infra": "Puri Coastal Highway, Tourism Belt"},
    {"name": "Paradip Port", "state": "Odisha, India", "lat": 20.31, "lon": 86.61, "population": 420000, "infra": "Major Deepwater Port, Oil Refineries"},
    {"name": "Kendrapara & Jagatsinghpur", "state": "Odisha, India", "lat": 20.50, "lon": 86.42, "population": 2580000, "infra": "Agriculture, Coastal Embankments"},
    {"name": "Digha & Contai", "state": "West Bengal, India", "lat": 21.62, "lon": 87.51, "population": 980000, "infra": "Coastal Tourism, Fishing Fleet"},
    {"name": "Haldia Port", "state": "West Bengal, India", "lat": 22.06, "lon": 88.06, "population": 650000, "infra": "Petrochemical Hub, Industrial Dock"},
    {"name": "Sagar Island & Sundarbans", "state": "West Bengal, India", "lat": 21.75, "lon": 88.10, "population": 1200000, "infra": "Biosphere Reserve, Estuary Settlements"},
    {"name": "Chittagong Port", "state": "Chittagong, Bangladesh", "lat": 22.33, "lon": 91.80, "population": 5200000, "infra": "Primary Maritime Gateway, Shipyards"},
    {"name": "Cox's Bazar", "state": "Chittagong, Bangladesh", "lat": 21.43, "lon": 91.98, "population": 2300000, "infra": "Refugee Settlements, Fishing Ports"},
    {"name": "Visakhapatnam", "state": "Andhra Pradesh, India", "lat": 17.68, "lon": 83.21, "population": 2350000, "infra": "Naval Base, Steel Plant, Container Terminal"},
    {"name": "Machilipatnam", "state": "Andhra Pradesh, India", "lat": 16.18, "lon": 81.13, "population": 890000, "infra": "Aquaculture, Coastal Port"},
    {"name": "Chennai Port", "state": "Tamil Nadu, India", "lat": 13.08, "lon": 80.27, "population": 7100000, "infra": "Major Port, IT & Automotive Corridors"},
    
    # Arabian Sea (India)
    {"name": "Dwarka & Okha Port", "state": "Gujarat, India", "lat": 22.24, "lon": 68.96, "population": 380000, "infra": "Maritime Pilgrimage Route, Okha Pier"},
    {"name": "Kandla & Mundra Port", "state": "Gujarat, India", "lat": 23.00, "lon": 70.21, "population": 1150000, "infra": "India's Largest Commercial Cargo Port Complex"},
    {"name": "Porbandar", "state": "Gujarat, India", "lat": 21.64, "lon": 69.62, "population": 580000, "infra": "Fishing Port, Chemical Industries"},
    {"name": "Veraval & Somnath", "state": "Gujarat, India", "lat": 20.90, "lon": 70.36, "population": 720000, "infra": "Trawler Docks, Coastal Highway"},
    {"name": "Mumbai & JNPT", "state": "Maharashtra, India", "lat": 18.95, "lon": 72.95, "population": 12500000, "infra": "Financial Capital, JNPT Container Hub, Offshore Oil"},
]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points in kilometers."""
    r = 6371.0  # Earth's radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def evaluate_impact_exposure(track_points: List[Dict[str, Any]], max_distance_km: float = 250.0) -> Dict[str, Any]:
    """
    Evaluate coastal population and infrastructure exposure along the forecasted cyclone path.
    """
    if not track_points:
        return {
            "coastal_alert_level": "Normal",
            "total_exposed_population": 0,
            "vulnerable_districts": [],
            "infrastructure_alerts": [],
            "landfall_estimate": None,
        }

    vulnerable_districts = []
    total_exposed = 0
    max_wind = max((pt.get("wind_speed_kt", 0) for pt in track_points), default=0)

    for zone in COASTAL_ZONES:
        # Find minimum distance to any point along the forecast track
        min_dist = float("inf")
        closest_point = None

        for pt in track_points:
            dist = haversine_km(zone["lat"], zone["lon"], pt["lat"], pt["lon"])
            if dist < min_dist:
                min_dist = dist
                closest_point = pt

        if min_dist <= max_distance_km:
            # Determine risk based on distance and storm intensity
            if min_dist < 60 and max_wind >= 64:
                risk = "Extreme"
            elif min_dist < 120 and max_wind >= 45:
                risk = "High"
            else:
                risk = "Moderate"

            total_exposed += zone["population"]
            vulnerable_districts.append({
                "district_or_port": zone["name"],
                "state_or_country": zone["state"],
                "estimated_population": zone["population"],
                "risk_level": risk,
                "distance_to_track_km": round(min_dist, 1),
                "infra": zone["infra"],
            })

    # Sort by distance
    vulnerable_districts.sort(key=lambda x: x["distance_to_track_km"])

    # Determine overall coastal alert level
    if any(d["risk_level"] == "Extreme" for d in vulnerable_districts):
        alert_level = "Red Alert (Evacuation & Marine Standstill)"
    elif any(d["risk_level"] == "High" for d in vulnerable_districts):
        alert_level = "Orange Alert (High Wind & Storm Surge Warning)"
    elif vulnerable_districts:
        alert_level = "Yellow Alert (Fishermen Advisory & Watch)"
    else:
        alert_level = "Green (No Immediate Land Threat)"

    # Infrastructure alerts
    infra_alerts = []
    for d in vulnerable_districts[:4]:
        if d["risk_level"] in ("Extreme", "High"):
            infra_alerts.append(f"{d['district_or_port']} ({d['state_or_country']}): {d['infra']} within {d['distance_to_track_km']} km of projected path.")

    # Landfall estimate
    landfall = None
    if vulnerable_districts and vulnerable_districts[0]["distance_to_track_km"] < 80:
        closest = vulnerable_districts[0]
        landfall = f"Projected close coastal passage / landfall near {closest['district_or_port']} ({closest['state_or_country']}) within 18–24 hours."

    return {
        "coastal_alert_level": alert_level,
        "total_exposed_population": total_exposed,
        "vulnerable_districts": vulnerable_districts,
        "infrastructure_alerts": infra_alerts,
        "landfall_estimate": landfall,
    }
