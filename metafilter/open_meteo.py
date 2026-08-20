"""Open-Meteo Historical Archive backend for ERA5 retrieval.

This is the second supported source for the same ERA5 data that
`scripts/download_era5.py` fetches from the CDS API. The two backends are
interchangeable from the filter-rule perspective — both produce datasets
that `calculate_daily_metrics()` can consume identically.

Differences from the CDS path:

* **Wire format:** JSON over HTTP (no auth, no quota) versus binary NetCDF.
* **Spatial:** A single point snapped to the ~0.25° ERA5 grid centroid of the
  requested AOI, versus the full bbox raster.
* **Latency:** ~200 ms per call versus minutes/hours in the CDS queue.
* **Cloud cover:** Returned in the same response as land variables (no separate
  retrieval needed) — `download_period(backend="open-meteo")` returns
  `cloud=[]` and the cloud signal lives in the land NetCDF.

The adapter writes the JSON response to a NetCDF file with the same variable
names, dimensions, and coordinate conventions that ERA5-Land emits, so the
downstream code path is bit-identical to the CDS case.
"""
from __future__ import annotations

import calendar
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .config import AREA, OUTPUT_DIR


# Open-Meteo hourly variables that map to the columns in calculate_daily_metrics.
# Order matters: we request all of them in one HTTP call.
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


# Mapping: Open-Meteo variable name → (CDS-equivalent variable name, conversion function).
# The conversion function takes the raw Open-Meteo values (already a numpy array)
# and returns the array in the same units the CDS-side NetCDF would carry, so
# calculate_daily_metrics() doesn't need to know which backend produced the data.
#
# Notes on each conversion:
#   * temperature_2m: Open-Meteo gives °C, ERA5-Land gives Kelvin
#   * precipitation: Open-Meteo gives mm/h, ERA5-Land gives m/h (we want m/h)
#   * shortwave_radiation: Open-Meteo gives W/m² (instantaneous), ERA5-Land gives
#     J/m² (accumulated over the hour). For a constant-W reading, J/m² = W * 3600 s.
#   * cloud_cover, cloud_cover_low: Open-Meteo gives percentage 0–100,
#     ERA5 single-levels gives fraction 0–1.
#   * soil_moisture/snow_depth: same units (m³/m³, m respectively).
#   * soil_temperature: Open-Meteo °C, ERA5-Land Kelvin.
_VAR_MAPPING = {
    "temperature_2m":             ("t2m",   lambda v: v + 273.15),
    "precipitation":              ("tp",    lambda v: v / 1000.0),
    "shortwave_radiation":        ("ssrd",  lambda v: v * 3600.0),
    "cloud_cover":                ("tcc",   lambda v: v / 100.0),
    "cloud_cover_low":            ("lcc",   lambda v: v / 100.0),
    "soil_temperature_0_to_7cm":  ("stl1",  lambda v: v + 273.15),
    "soil_moisture_0_to_7cm":     ("swvl1", lambda v: v),
    "snow_depth":                 ("sd",    lambda v: v),
    "dew_point_2m":               ("d2m",   lambda v: v + 273.15),
}


def _snap_to_era5_grid(area, grid_deg=0.25):
    """Snap area centroid to the nearest ~0.25° cell so neighbouring AOIs share cache."""
    lat = round(((area["south"] + area["north"]) / 2) / grid_deg) * grid_deg
    lon = round(((area["west"] + area["east"]) / 2) / grid_deg) * grid_deg
    return lat, lon


def open_meteo_json_to_dataset(hourly_payload, lat, lon):
    """Convert an Open-Meteo /v1/archive 'hourly' payload to an xarray.Dataset.

    The returned dataset has the same variable names and structure as the
    NetCDF that scripts/download_era5.py produces, with a single 1×1 spatial
    grid (the snapped centroid). Downstream code consumes it identically.
    """
    times = pd.to_datetime(hourly_payload["time"])
    lats = np.array([lat], dtype=np.float32)
    lons = np.array([lon], dtype=np.float32)

    data_vars = {}
    for om_name, (cds_name, convert) in _VAR_MAPPING.items():
        if om_name not in hourly_payload:
            continue
        raw = np.asarray(hourly_payload[om_name], dtype=np.float64)
        # Open-Meteo returns None for missing values; replace with NaN.
        # np.asarray on a list with None gives dtype=object, but with the
        # explicit dtype=float64 above, None becomes NaN automatically.
        converted = convert(raw).astype(np.float32)
        # Reshape to (n_time, n_lat=1, n_lon=1) to match ERA5-Land grid shape.
        reshaped = converted.reshape(-1, 1, 1)
        data_vars[cds_name] = (("time", "latitude", "longitude"), reshaped)

    return xr.Dataset(
        data_vars,
        coords={"time": times, "latitude": lats, "longitude": lons},
    )


def fetch_open_meteo_archive(
    year,
    month,
    *,
    area=None,
    variables=None,
    timeout=60,
):
    """Hit the Open-Meteo Historical Archive endpoint for one month + AOI.

    Returns the parsed JSON 'hourly' block. Pure HTTP — no NetCDF I/O — so
    this function is straightforward to mock in tests.
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
            f"Open-Meteo response did not contain 'hourly' block: keys={list(payload.keys())}"
        )
    return payload["hourly"], lat, lon


def download_open_meteo_land(
    year=2024,
    month=8,
    area=None,
    variables=None,
):
    """Fetch one month from Open-Meteo and persist as NetCDF.

    The output file has the same shape downstream code expects from a CDS
    `reanalysis-era5-land` retrieve, with the same variable names. The
    spatial grid is 1×1 (centroid only) — small AOIs see the same value
    everywhere because Open-Meteo serves one ERA5 cell.

    Returns the absolute path of the written .nc file.
    """
    hourly, lat, lon = fetch_open_meteo_archive(
        year, month, area=area, variables=variables,
    )
    ds = open_meteo_json_to_dataset(hourly, lat, lon)

    out_path = Path(OUTPUT_DIR) / "era5" / f"openmeteo_land_{year}_{month:02d}.nc"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # scipy engine so we don't depend on h5netcdf/netCDF4 — same as tests.
    ds.to_netcdf(
        out_path,
        engine="scipy",
        encoding={"time": {"units": "seconds since 1970-01-01", "dtype": "float64"}},
    )
    return str(out_path)


if __name__ == "__main__":
    download_open_meteo_land()
