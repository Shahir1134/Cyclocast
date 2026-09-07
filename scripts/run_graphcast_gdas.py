"""
scripts/run_graphcast_gdas.py
==============================
Full pipeline to fetch latest GDAS/GFS data from AWS S3, format inputs for
GraphCast Small, run predictions, and optionally feed directly into the fusion pipeline.

Based on graph.ipynb workflow.
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import s3fs
    import xarray as xr
    import cfgrib
except ImportError as e:
    print(f"[ERROR] Missing required dependencies: {e}")
    print("Please install via: pip install s3fs xarray cfgrib eccodes netCDF4")
    sys.exit(1)

# GraphCast pressure levels
GRAPHCAST_LEVELS = [
    50, 100, 150, 200, 250, 300,
    400, 500, 600, 700, 850, 925, 1000
]

def gdas_path(t: datetime) -> str:
    return f"noaa-gfs-bdp-pds/gdas.{t:%Y%m%d}/{t.hour:02d}/atmos/gdas.t{t.hour:02d}z.pgrb2.0p25.f000"

def gdas_precip_path(t: datetime) -> str:
    return f"noaa-gfs-bdp-pds/gdas.{t:%Y%m%d}/{t.hour:02d}/atmos/gdas.t{t.hour:02d}z.pgrb2.0p25.f003"

def find_latest_available_gdas(fs: s3fs.S3FileSystem, max_lookback_cycles: int = 8) -> datetime:
    """Find the latest GDAS analysis available in NOAA's public AWS bucket."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    cycle_hour = (now.hour // 6) * 6
    candidate = now.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)

    for _ in range(max_lookback_cycles):
        if fs.exists(gdas_path(candidate)):
            return candidate
        candidate -= timedelta(hours=6)

    raise RuntimeError("No available GDAS cycle found in lookback window")

def load_and_rename(path: str, precip_path: str) -> xr.Dataset:
    """Convert GRIB2 variables to GraphCast naming conventions and downsample."""
    plev = xr.open_dataset(path, engine="cfgrib",
                           filter_by_keys={"typeOfLevel": "isobaricInhPa"})
    plev = plev.sel(isobaricInhPa=GRAPHCAST_LEVELS)
    plev = plev.rename({
        "t": "temperature", "u": "u_component_of_wind",
        "v": "v_component_of_wind", "q": "specific_humidity",
        "gh": "geopotential", "w": "vertical_velocity",
        "isobaricInhPa": "level"
    })
    plev["geopotential"] = plev["geopotential"] * 9.80665

    mslp = xr.open_dataset(path, engine="cfgrib",
                          filter_by_keys={"typeOfLevel": "meanSea"})
    mslp = mslp.rename({"prmsl": "mean_sea_level_pressure"})

    t2m = xr.open_dataset(path, engine="cfgrib",
                         filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 2})
    t2m = t2m.rename({"t2m": "2m_temperature"})

    wind10 = xr.open_dataset(path, engine="cfgrib",
                             filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 10})
    wind10 = wind10.rename({"u10": "10m_u_component_of_wind", "v10": "10m_v_component_of_wind"})

    precip = xr.open_dataset(precip_path, engine="cfgrib",
                             filter_by_keys={"typeOfLevel": "surface", "shortName": "tp"})
    precip = precip.rename({"tp": "total_precipitation_6hr"})
    precip["total_precipitation_6hr"] = precip["total_precipitation_6hr"] / 1000.0  # kg/m^2 -> m

    merged = xr.merge([plev, mslp, t2m, wind10, precip], compat="override")
    merged = merged.coarsen(latitude=4, longitude=4, boundary="trim").mean()
    return merged

def load_static_fields(path: str) -> xr.Dataset:
    """Load orography and land-sea mask."""
    orog = xr.open_dataset(path, engine="cfgrib",
                           filter_by_keys={"typeOfLevel": "surface", "shortName": "orog"})
    lsm = xr.open_dataset(path, engine="cfgrib",
                          filter_by_keys={"typeOfLevel": "surface", "shortName": "lsm"})

    orog = orog.rename({"orog": "geopotential_at_surface"})
    orog["geopotential_at_surface"] = orog["geopotential_at_surface"] * 9.80665
    lsm = lsm.rename({"lsm": "land_sea_mask"})

    static = xr.merge([orog, lsm], compat="override")
    static = static.rename({"latitude": "lat", "longitude": "lon"})
    static = static.coarsen(lat=4, lon=4, boundary="trim").mean()
    return static.drop_vars(["time", "step", "valid_time"], errors="ignore")

def prepare_gdas_batch(work_dir: str = "tmp_gdas") -> xr.Dataset:
    """Download the 2 consecutive cycles and format into GraphCast input tensor format."""
    os.makedirs(work_dir, exist_ok=True)
    fs = s3fs.S3FileSystem(anon=True)

    print("[1/3] Finding latest GDAS cycles from NOAA AWS S3...")
    t1 = find_latest_available_gdas(fs)
    t0 = t1 - timedelta(hours=6)
    print(f"      Selected t0 = {t0} UTC, t1 = {t1} UTC")

    for t in (t0, t1):
        p_path = os.path.join(work_dir, f"{t:%Y%m%d%H}.grib2")
        tp_path = os.path.join(work_dir, f"{t:%Y%m%d%H}_precip.grib2")
        if not os.path.exists(p_path):
            print(f"      Downloading {gdas_path(t)}...")
            fs.get(gdas_path(t), p_path)
        if not os.path.exists(tp_path):
            print(f"      Downloading {gdas_precip_path(t)}...")
            fs.get(gdas_precip_path(t), tp_path)

    print("[2/3] Parsing GRIB2 files and converting coordinates...")
    ds0 = load_and_rename(os.path.join(work_dir, f"{t0:%Y%m%d%H}.grib2"),
                          os.path.join(work_dir, f"{t0:%Y%m%d%H}_precip.grib2"))
    ds1 = load_and_rename(os.path.join(work_dir, f"{t1:%Y%m%d%H}.grib2"),
                          os.path.join(work_dir, f"{t1:%Y%m%d%H}_precip.grib2"))

    batch = xr.concat([ds0, ds1], dim="time").expand_dims("batch")
    batch = batch.rename({"latitude": "lat", "longitude": "lon"})

    absolute_times = batch["time"].values
    batch = batch.assign_coords(
        datetime=(("batch", "time"), absolute_times.reshape(1, -1))
    )
    time_deltas = absolute_times - absolute_times[-1]
    batch = batch.assign_coords(time=time_deltas)

    static_fields = load_static_fields(os.path.join(work_dir, f"{t1:%Y%m%d%H}.grib2"))
    batch = batch.assign(
        geopotential_at_surface=(("lat", "lon"), static_fields["geopotential_at_surface"].values),
        land_sea_mask=(("lat", "lon"), static_fields["land_sea_mask"].values),
    )

    print("[3/3] Extending placeholders for +6h, +12h, +18h, +24h targets...")
    future_offsets = np.array([6, 12, 18, 24], dtype="timedelta64[h]").astype("timedelta64[ns]")
    static_vars = ["geopotential_at_surface", "land_sea_mask"]

    placeholder = batch.drop_vars(static_vars).isel(time=[-1]) * np.nan
    placeholders = xr.concat(
        [placeholder.assign_coords(time=[batch["time"].values[-1] + off]) for off in future_offsets],
        dim="time",
        data_vars="minimal",
        coords="minimal",
    )

    full_batch = xr.concat(
        [batch.drop_vars(static_vars), placeholders],
        dim="time",
        data_vars="minimal",
        coords="minimal",
    )
    full_batch = full_batch.assign(
        geopotential_at_surface=batch["geopotential_at_surface"],
        land_sea_mask=batch["land_sea_mask"],
    )

    last_dt = batch["datetime"].values[0, -1]
    future_dts = np.array([last_dt + off for off in future_offsets])
    full_datetime = np.concatenate([batch["datetime"].values[0], future_dts]).reshape(1, -1)
    full_batch = full_batch.assign_coords(datetime=(("batch", "time"), full_datetime))

    return full_batch

def run_graphcast_prediction(full_batch: xr.Dataset, model=None) -> xr.Dataset:
    """Run GraphCast model prediction using JAX."""
    if model is None:
        try:
            # Check for DeepMind GraphCast
            from graphcast import graphcast
            from graphcast_wrapper import GraphCastModel
            model = GraphCastModel.from_gcs(
                "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - "
                "pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
            )
        except Exception as e:
            print(f"[WARNING] Native GraphCast runner cannot be loaded directly: {e}")
            raise e

    inputs, targets_template, forcings = model.extract_inputs_targets_forcings(
        full_batch, target_lead_times=slice("6h", "24h"),
    )
    preds = model.predict(inputs, targets_template, forcings)
    return preds

if __name__ == "__main__":
    print("GraphCast GDAS Pipeline")
    print("=" * 60)
    try:
        batch_input = prepare_gdas_batch()
        print("Batch successfully prepared with shape:")
        print(batch_input.sizes)
    except Exception as e:
        print(f"Error during execution: {e}")
