"""Shared test fixtures.

Two responsibilities:
1. Stub OpenEO credentials before `utils.config` import.
2. Provide synthetic ERA5 NetCDF builders for derived-column tests, so
   the test suite runs without ECMWF/CDS access.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

os.environ.setdefault("OPENEO_USERNAME", "test-user")
os.environ.setdefault("OPENEO_PASSWORD", "test-pass")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TEST_AREA = {"west": 18.0, "east": 18.2, "south": 59.2, "north": 59.4}


@pytest.fixture
def test_area():
    return dict(TEST_AREA)


def _make_grid(area, n_lat=3, n_lon=3):
    lats = np.linspace(area["south"], area["north"], n_lat)
    lons = np.linspace(area["west"], area["east"], n_lon)
    return lats, lons


def make_era5_land_dataset(
    *, start: str, end: str, area=None,
    t2m_pattern="seasonal", tp_pattern="dry_with_event",
) -> xr.Dataset:
    """Synthetic ERA5-Land hourly dataset with t2m and tp.

    Hourly resolution so future tests can exercise pass-time sampling.
    Time range is inclusive of `start` 00:00 and exclusive of `end` 00:00.
    """
    if area is None:
        area = TEST_AREA

    times = pd.date_range(start=start, end=end, freq="h", inclusive="left")
    lats, lons = _make_grid(area)
    n_t, n_la, n_lo = len(times), len(lats), len(lons)

    if t2m_pattern == "seasonal":
        doy = (times.dayofyear - 1).to_numpy()
        hour = times.hour.to_numpy()
        seasonal = 288.15 + 7.0 * np.sin(2 * np.pi * doy / 365)
        diurnal = 4.0 * np.sin(2 * np.pi * (hour - 6) / 24)
        t2m_base = (seasonal + diurnal).astype(np.float32)
        t2m = np.broadcast_to(t2m_base[:, None, None], (n_t, n_la, n_lo)).copy()
    elif t2m_pattern == "cold":
        t2m = np.full((n_t, n_la, n_lo), 268.15, dtype=np.float32)  # −5 °C
    elif t2m_pattern == "warm":
        t2m = np.full((n_t, n_la, n_lo), 293.15, dtype=np.float32)  # 20 °C
    else:
        raise ValueError(f"Unknown t2m_pattern: {t2m_pattern}")

    tp = np.zeros((n_t, n_la, n_lo), dtype=np.float32)
    if tp_pattern == "dry_with_event":
        rain_day = pd.Timestamp(start) + pd.Timedelta(days=5)
        mask = (times.normalize() == rain_day.normalize())
        tp[mask, :, :] = 0.01 / 24  # 10 mm across that day, in metres
    elif tp_pattern == "dry":
        pass
    elif tp_pattern == "wet_continuous":
        tp[:] = 0.002 / 24  # 2 mm/day every day
    else:
        raise ValueError(f"Unknown tp_pattern: {tp_pattern}")

    return xr.Dataset(
        {
            "t2m": (("time", "latitude", "longitude"), t2m),
            "tp": (("time", "latitude", "longitude"), tp),
        },
        coords={"time": times, "latitude": lats, "longitude": lons},
    )


def write_netcdf(ds: xr.Dataset, path: Path) -> Path:
    """Write via scipy engine (NetCDF3) so tests don't depend on netCDF4
    or h5netcdf. Time pinned to seconds-since-epoch float64 to avoid an
    xarray + pandas-2.x decode quirk on the default hours-since-X encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(
        path,
        engine="scipy",
        encoding={"time": {"units": "seconds since 1970-01-01", "dtype": "float64"}},
    )
    return path


@pytest.fixture
def era5_land_factory(tmp_path):
    """Build an ERA5-Land NetCDF on demand; returns its path."""
    counter = {"n": 0}

    def _build(**kwargs) -> Path:
        counter["n"] += 1
        ds = make_era5_land_dataset(**kwargs)
        return write_netcdf(ds, tmp_path / f"era5_land_{counter['n']}.nc")

    return _build
