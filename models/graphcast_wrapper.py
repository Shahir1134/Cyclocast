"""
models/graphcast_wrapper.py
============================
Thin adapter around your existing GraphCast JAX runner.

Design principle — Dependency Injection
----------------------------------------
This wrapper does NOT import JAX or graphcast directly. Instead you supply
a ``runner_fn`` callable at construction time. This means:

  - The fusion system runs end-to-end even if JAX is not installed.
  - You can swap in a mock runner during development / unit tests.
  - Your real GraphCast runner plugs in with a single argument.

Expected runner_fn signature
-----------------------------
    runner_fn(inputs: xr.Dataset) -> xr.Dataset

    The returned xr.Dataset must have a ``time`` dimension with (at least)
    4 time steps corresponding to +6h, +12h, +18h, +24h, and the variables
    listed in config → graphcast.variables.

Usage
------
    from models.graphcast_wrapper import GraphCastWrapper

    # Wire in your real runner (from graph.ipynb)
    def my_runner(inputs):
        # ... your JAX GraphCast call ...
        return preds   # xr.Dataset

    gc = GraphCastWrapper(runner_fn=my_runner)
    forecasts = gc.run_forecast(gdas_inputs)
    # forecasts["6h"]  → xr.Dataset for +6 h slice
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import xarray as xr
    _XARRAY_AVAILABLE = True
except ImportError:
    _XARRAY_AVAILABLE = False
    logger.warning("xarray not installed. GraphCastWrapper will use dict fallback.")


# Default variable list (override via config)
DEFAULT_VARIABLES: List[str] = [
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "specific_humidity",
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
]

FORECAST_HORIZONS_H: List[int] = [6, 12, 18, 24]


class GraphCastWrapper:
    """
    Adapter between GraphCast JAX runner and the PyTorch fusion pipeline.

    Parameters
    ----------
    runner_fn : callable
        Your GraphCast runner. Signature: ``(inputs: xr.Dataset) → xr.Dataset``.
        The output dataset must contain a ``time`` dimension with at least 4
        steps, each separated by 6 hours.
    variables : list[str] | None
        Atmospheric variables to extract. Defaults to DEFAULT_VARIABLES.
    forecast_horizons_h : list[int]
        Forecast lead times in hours. Default [6, 12, 18, 24].
    """

    def __init__(
        self,
        runner_fn: Optional[Callable] = None,
        variables: Optional[List[str]] = None,
        forecast_horizons_h: Optional[List[int]] = None,
    ) -> None:
        self.runner_fn = runner_fn or self._mock_runner
        self.variables = variables or DEFAULT_VARIABLES
        self.forecast_horizons_h = forecast_horizons_h or FORECAST_HORIZONS_H
        self._last_raw_output = None  # cache for debugging

        logger.info(
            "GraphCastWrapper initialised (runner=%s).",
            getattr(self.runner_fn, "__name__", repr(self.runner_fn)),
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run_forecast(self, inputs) -> Dict[str, object]:
        """
        Run the GraphCast forecast and return per-horizon slices.

        Parameters
        ----------
        inputs : xr.Dataset
            GDAS/ERA5 atmospheric state compatible with GraphCast Small.

        Returns
        -------
        dict
            Keys: ``"6h"``, ``"12h"``, ``"18h"``, ``"24h"``.
            Values: xr.Dataset (or dict if xarray unavailable) containing
                    the atmospheric variables at that forecast horizon.
        """
        logger.info("Running GraphCast forecast …")
        raw_preds = self.runner_fn(inputs)
        self._last_raw_output = raw_preds
        logger.info("GraphCast forecast complete.")

        return self._slice_horizons(raw_preds)

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _slice_horizons(self, preds) -> Dict[str, object]:
        """
        Slice the raw GraphCast output by forecast horizon.

        GraphCast outputs a dataset with a ``time`` dimension.
        Step 0 → +6h, step 1 → +12h, step 2 → +18h, step 3 → +24h.
        """
        horizons: Dict[str, object] = {}

        for step_idx, hours in enumerate(self.forecast_horizons_h):
            key = f"{hours}h"
            try:
                if _XARRAY_AVAILABLE and hasattr(preds, "isel"):
                    # Standard xarray path
                    # GraphCast uses batch + time dims: isel(batch=0, time=step_idx)
                    if "batch" in preds.dims:
                        slc = preds.isel(batch=0, time=step_idx)
                    else:
                        slc = preds.isel(time=step_idx)

                    # Keep only the variables we care about (intersection)
                    available = [v for v in self.variables if v in slc.data_vars]
                    horizons[key] = slc[available]
                else:
                    # Fallback: assume preds is a list of dicts
                    horizons[key] = preds[step_idx]
            except (IndexError, KeyError) as exc:
                logger.warning(
                    "Could not slice horizon %s (step %d): %s", key, step_idx, exc
                )
                horizons[key] = None

        return horizons

    @staticmethod
    def _mock_runner(inputs) -> object:
        """
        Mock GraphCast runner for testing / offline development.
        Returns a synthetic xr.Dataset with the expected structure.
        """
        import numpy as np

        if not _XARRAY_AVAILABLE:
            # Return a simple list of dicts
            import math
            return [
                {var: np.random.randn(81, 81).astype(np.float32)
                 for var in DEFAULT_VARIABLES}
                for _ in range(4)
            ]

        import xarray as xr
        from datetime import datetime, timedelta

        n_lat, n_lon = 81, 81  # ±10° at 0.25° resolution
        n_time = 4
        n_pressure = 3

        lats = np.linspace(10, 30, n_lat)
        lons = np.linspace(78, 98, n_lon)
        times = [datetime(2024, 10, 1) + timedelta(hours=6 * (i + 1))
                 for i in range(n_time)]
        pressure_levels = [250, 500, 850]

        data_vars = {}
        # Surface variables (time, lat, lon)
        for var in ["mean_sea_level_pressure", "10m_u_component_of_wind",
                    "10m_v_component_of_wind", "2m_temperature"]:
            data_vars[var] = xr.DataArray(
                np.random.randn(1, n_time, n_lat, n_lon).astype(np.float32),
                dims=["batch", "time", "lat", "lon"],
            )
        # Pressure-level variables (time, level, lat, lon)
        for var in ["temperature", "u_component_of_wind", "v_component_of_wind",
                    "geopotential", "specific_humidity"]:
            data_vars[var] = xr.DataArray(
                np.random.randn(1, n_time, n_pressure, n_lat, n_lon).astype(np.float32),
                dims=["batch", "time", "level", "lat", "lon"],
            )

        ds = xr.Dataset(
            data_vars,
            coords={
                "batch": [0],
                "time": times,
                "lat": lats,
                "lon": lons,
                "level": pressure_levels,
            },
        )
        logger.warning(
            "GraphCastWrapper is using MOCK runner — real forecasts disabled."
        )
        return ds

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def get_available_variables(self, forecast: Dict[str, object]) -> List[str]:
        """Return list of variables present in a forecast slice."""
        first = next(iter(forecast.values()), None)
        if first is None:
            return []
        if _XARRAY_AVAILABLE and hasattr(first, "data_vars"):
            return list(first.data_vars)
        if isinstance(first, dict):
            return list(first.keys())
        return []
