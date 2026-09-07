import xarray as xr
from pathlib import Path

grib_path = Path("tmp_gdas/2026090318.grib2")
print(f"Reading MSLP from {grib_path}...")
ds_mslp = xr.open_dataset(
    str(grib_path),
    engine="cfgrib",
    filter_by_keys={"typeOfLevel": "meanSea"},
    backend_kwargs={"indexpath": ""}
)
print("MSLP loaded. Variables:", list(ds_mslp.data_vars.keys()))

print("Reading 10m wind...")
ds_wind = xr.open_dataset(
    str(grib_path),
    engine="cfgrib",
    filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 10},
    backend_kwargs={"indexpath": ""}
)
print("10m Wind loaded. Variables:", list(ds_wind.data_vars.keys()))

# Slice to North Indian Ocean
sub_mslp = ds_mslp["prmsl"].sel(latitude=slice(30, 0), longitude=slice(60, 100)) / 100.0  # Pa to hPa
sub_u = ds_wind["u10"].sel(latitude=slice(30, 0), longitude=slice(60, 100))
sub_v = ds_wind["v10"].sel(latitude=slice(30, 0), longitude=slice(60, 100))
sub_wspd = (sub_u**2 + sub_v**2)**0.5 * 1.94384  # kt

min_p = float(sub_mslp.min())
max_w = float(sub_wspd.max())
print(f"North Indian Ocean GFS: Min MSLP={min_p:.1f} hPa, Max Wind={max_w:.1f} kt")
