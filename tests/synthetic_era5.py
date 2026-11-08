"""Synthetic ERA5-Land files, so the pipeline can be tested without CDS access."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

TEST_AREA: dict[str, float] = {"west": 18.0, "east": 18.2, "south": 59.2, "north": 59.4}

# A 0.1° grid one cell wider than TEST_AREA on every side. Latitude runs
# north to south, as in the files CDS delivers.
GRID_LATITUDES = (59.5, 59.4, 59.3, 59.2, 59.1)
GRID_LONGITUDES = (17.9, 18.0, 18.1, 18.2, 18.3)

OUTSIDE_AREA_OFFSET_K = 30.0


def hourly_times(start: str, days: int) -> pd.DatetimeIndex:
    """Every hour of `days` whole days, starting at `start` 00:00."""
    return pd.date_range(start, periods=days * 24, freq="h")


def era5_land_dataset(
    times: pd.DatetimeIndex,
    daily_mean_temp_c: Sequence[float],
    *,
    latitudes: Sequence[float] = GRID_LATITUDES,
    longitudes: Sequence[float] = GRID_LONGITUDES,
    time_name: str = "valid_time",
) -> xr.Dataset:
    """Hourly t2m and tp on the grid, one temperature per calendar day.

    Inside TEST_AREA a whole day of t2m averages to that day's value: the
    diurnal cycle and the latitude and longitude gradients all cancel, so
    dropping an edge row or column shifts the mean. Cells outside TEST_AREA
    are OUTSIDE_AREA_OFFSET_K warmer, so a missing subset shows up as well.
    tp is zero throughout: no day is wet.
    """
    day_index = (times.normalize() - times[0].normalize()).days.to_numpy()
    base_k = np.asarray(daily_mean_temp_c, dtype=float)[day_index] + 273.15
    diurnal_k = 4.0 * np.sin(2 * np.pi * times.hour.to_numpy() / 24)

    lat = np.asarray(latitudes, dtype=float)[:, None]
    lon = np.asarray(longitudes, dtype=float)[None, :]
    centre_lat = (TEST_AREA["south"] + TEST_AREA["north"]) / 2
    centre_lon = (TEST_AREA["west"] + TEST_AREA["east"]) / 2
    inside = (
        (lat >= TEST_AREA["south"])
        & (lat <= TEST_AREA["north"])
        & (lon >= TEST_AREA["west"])
        & (lon <= TEST_AREA["east"])
    )
    cell_offset_k = np.where(
        inside,
        10.0 * (lat - centre_lat) + 3.0 * (lon - centre_lon),
        OUTSIDE_AREA_OFFSET_K,
    )

    t2m = (base_k + diurnal_k)[:, None, None] + cell_offset_k[None, :, :]
    dims = (time_name, "latitude", "longitude")
    return xr.Dataset(
        {
            "t2m": (dims, t2m.astype(np.float32), {"units": "K"}),
            "tp": (dims, np.zeros_like(t2m, dtype=np.float32), {"units": "m"}),
        },
        coords={time_name: times, "latitude": lat[:, 0], "longitude": lon[0, :]},
    )


def write_netcdf(directory: Path, name: str, dataset: xr.Dataset) -> Path:
    """Write `dataset` to `directory/<name>.nc` and return the path."""
    path = directory / f"{name}.nc"
    dataset.to_netcdf(path)
    return path
