"""Open-Meteo Historical Archive backend for ERA5 retrieval.

A second source for the same ERA5 data the CDS API exposes via
`scripts/download_era5.py`. Both backends produce datasets that downstream
`calculate_daily_metrics()` consumes identically.

Trade-offs vs CDS:

* No authentication, no per-user quota.
* JSON over HTTP (~200 ms per call) instead of NetCDF in a queued retrieve.
* Single point snapped to the ~0.25° ERA5 grid centroid, not a full bbox raster.
* Cloud cover lives in the same response as land variables — no separate
  retrieval needed.

The adapter writes a NetCDF using the same variable names and shape an
ERA5-Land file would carry, with unit conversions applied (°C → K, mm/h
→ m/h, W/m² → J/m²/h, % → fraction), so callers don't have to branch on
which backend produced the file.
"""
from __future__ import annotations

import calendar
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from utils.config import AREA, OUTPUT_DIR


OPEN_METEO_DEFAULT_HOURLY = [
    "temperature_2m",
    "precipitation",
    "shortwave_radiation",
    "cloud_cover",
    "cloud_cover_low",
    "soil_temperature_0_to_7cm",
    "soil_moisture_0_to_7cm",
    "snow_depth",
    "dew_point_2m",
]

# Open-Meteo name → (target name, conversion to CDS-equivalent units).
_VAR_MAPPING = {
    "temperature_2m":            ("t2m",   lambda v: v + 273.15),
    "precipitation":             ("tp",    lambda v: v / 1000.0),
    "shortwave_radiation":       ("ssrd",  lambda v: v * 3600.0),
    "cloud_cover":               ("tcc",   lambda v: v / 100.0),
    "cloud_cover_low":           ("lcc",   lambda v: v / 100.0),
    "soil_temperature_0_to_7cm": ("stl1",  lambda v: v + 273.15),
    "soil_moisture_0_to_7cm":    ("swvl1", lambda v: v),
    "snow_depth":                ("sd",    lambda v: v),
    "dew_point_2m":              ("d2m",   lambda v: v + 273.15),
}


def _snap_to_era5_grid(area, grid_deg=0.25):
    lat = round(((area["south"] + area["north"]) / 2) / grid_deg) * grid_deg
    lon = round(((area["west"] + area["east"]) / 2) / grid_deg) * grid_deg
    return lat, lon


def open_meteo_json_to_dataset(hourly_payload, lat, lon):
    """Convert an Open-Meteo /v1/archive `hourly` payload to an xarray.Dataset
    that has the same variable names + dimension shape as an ERA5-Land NetCDF."""
    times = pd.to_datetime(hourly_payload["time"])
    lats = np.array([lat], dtype=np.float32)
    lons = np.array([lon], dtype=np.float32)

    data_vars = {}
    for om_name, (cds_name, convert) in _VAR_MAPPING.items():
        if om_name not in hourly_payload:
            continue
        raw = np.asarray(hourly_payload[om_name], dtype=np.float64)
        converted = convert(raw).astype(np.float32).reshape(-1, 1, 1)
        data_vars[cds_name] = (("time", "latitude", "longitude"), converted)

    return xr.Dataset(
        data_vars,
        coords={"time": times, "latitude": lats, "longitude": lons},
    )


def fetch_open_meteo_archive(year, month, *, area=None, variables=None, timeout=60):
    """Hit the Open-Meteo Historical Archive for one month over `area`.

    Returns `(hourly_payload_dict, snapped_lat, snapped_lon)`. Pure HTTP —
    no NetCDF I/O — so it's straightforward to mock in tests.
    """
    import requests

    area = area or AREA
    variables = variables or OPEN_METEO_DEFAULT_HOURLY

    lat, lon = _snap_to_era5_grid(area)
    last_day = calendar.monthrange(year, month)[1]
    start_date = f"{year}-{month:02d}-01"
    end_date = f"{year}-{month:02d}-{last_day:02d}"

    response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(variables),
            "timezone": "UTC",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if "hourly" not in payload:
        raise RuntimeError(
            f"Open-Meteo response missing 'hourly' block: keys={list(payload.keys())}"
        )
    return payload["hourly"], lat, lon


def download_open_meteo_land(year=2024, month=8, area=None, variables=None):
    """Fetch one month from Open-Meteo and persist as a NetCDF whose variable
    names and dimensions match an ERA5-Land file. Returns the path."""
    hourly, lat, lon = fetch_open_meteo_archive(year, month, area=area, variables=variables)
    ds = open_meteo_json_to_dataset(hourly, lat, lon)

    out_path = Path(OUTPUT_DIR) / "era5" / f"openmeteo_land_{year}_{month:02d}.nc"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(
        out_path,
        engine="scipy",
        encoding={"time": {"units": "seconds since 1970-01-01", "dtype": "float64"}},
    )
    return str(out_path)


if __name__ == "__main__":
    download_open_meteo_land()
