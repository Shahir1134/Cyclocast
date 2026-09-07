import urllib.request
import json

url = "http://localhost:8000/api/cyclones/amphan-2020/replay"
try:
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read().decode())
        print("Cyclone ID:", data.get("cyclone_id"))
        print("Mean Track Error:", data.get("mean_track_error_km"))
        print("Mean Intensity Error:", data.get("mean_intensity_error_kt"))
        print("Steps count:", len(data.get("steps", [])))
        for i, s in enumerate(data.get("steps", [])):
            print(f"Step {i}: actual=({s.get('actual_lat')}, {s.get('actual_lon')}), pred=({s.get('predicted_lat')}, {s.get('predicted_lon')}), actual_wind={s.get('actual_wind_kt')}, pred_wind={s.get('predicted_wind_kt')}")
except Exception as e:
    print("Error:", e)
