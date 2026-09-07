import urllib.request
import json

# Test fetching real NOAA GFS 0.25 deg forecast for Bay of Bengal (20.4N, 88.1E)
url = "https://api.open-meteo.com/v1/gfs?latitude=20.4&longitude=88.1&hourly=surface_pressure,wind_speed_10m,wind_direction_10m,wind_speed_850hPa,wind_direction_850hPa&forecast_days=2"
print(f"Fetching real GFS forecast from {url}...")
try:
    req = urllib.request.Request(url, headers={'User-Agent': 'CycloCast/1.0'})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())
        print("[SUCCESS] NOAA GFS live data fetched!")
        hourly = data.get("hourly", {})
        print("Hours available:", len(hourly.get("time", [])))
        for i in [0, 6, 12, 18, 24]:
            t = hourly['time'][i]
            p = hourly['surface_pressure'][i]
            w = hourly['wind_speed_10m'][i]
            w850 = hourly['wind_speed_850hPa'][i]
            d = hourly['wind_direction_10m'][i]
            print(f"  +{i:02d}h ({t}): Pres={p:.1f} hPa, Wind10m={w:.1f} km/h (dir={d}°), Wind850hPa={w850:.1f} km/h")
except Exception as e:
    print("[FAIL]:", e)
