"""Shared test fixtures for the metafilter test suite.

Two responsibilities:

1. Set fake OpenEO credentials so `utils.config` import doesn't raise during
   collection. Tests that need real OpenEO are marked separately.
2. Provide synthetic ERA5 NetCDF builders so derived-column logic can be tested
   without ECMWF/CDS access.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Stub credentials before any production module imports.
os.environ.setdefault("OPENEO_USERNAME", "test-user")
os.environ.setdefault("OPENEO_PASSWORD", "test-pass")

# Ensure repo root is on sys.path so `from utils.config import AREA` works.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ── Fixture AOI ────────────────────────────────────────────────────────────

TEST_AREA = {"west": 18.0, "east": 18.2, "south": 59.2, "north": 59.4}


@pytest.fixture
def test_area():
    return dict(TEST_AREA)


# ── Synthetic ERA5-Land NetCDF builders ────────────────────────────────────

def _make_grid(area, n_lat=3, n_lon=3):
    lats = np.linspace(area["south"], area["north"], n_lat)
    lons = np.linspace(area["west"], area["east"], n_lon)
    return lats, lons


def make_era5_land_dataset(
    *,
    start: str,
    end: str,
    area=None,
    t2m_pattern="seasonal",
    tp_pattern="dry_with_event",
    swvl1_pattern="flat",
    skt_pattern=None,
    sd_pattern=None,
    ssrd_pattern=None,
    seed=0,
) -> xr.Dataset:
    """Build a synthetic ERA5-Land hourly dataset.

    Hourly resolution so pass-time sampling can be exercised. Time range is
    inclusive of `start` 00:00 and exclusive of `end` 00:00.
    """
    if area is None:
        area = TEST_AREA

    rng = np.random.default_rng(seed)
    times = pd.date_range(start=start, end=end, freq="h", inclusive="left")
    lats, lons = _make_grid(area)
    n_t, n_la, n_lo = len(times), len(lats), len(lons)

    # ── Patterns (Kelvin / m / m^3/m^3 / J/m^2) ─────────────────────────
    t2m = np.zeros((n_t, n_la, n_lo), dtype=np.float32)
    if t2m_pattern == "seasonal":
        # Sinusoidal seasonal + diurnal cycle, base 15 °C summer
        doy = (times.dayofyear - 1).to_numpy()
        hour = times.hour.to_numpy()
        seasonal = 288.15 + 7.0 * np.sin(2 * np.pi * doy / 365)
        diurnal = 4.0 * np.sin(2 * np.pi * (hour - 6) / 24)
        base = (seasonal + diurnal)[:, None, None]
        t2m[:] = base + rng.normal(0, 0.5, t2m.shape).astype(np.float32)
    elif t2m_pattern == "cold":
        t2m[:] = 268.15  # −5 °C, well below freeze threshold
    elif t2m_pattern == "warm":
        t2m[:] = 293.15  # 20 °C, comfortably above growth threshold
    else:
        raise ValueError(f"Unknown t2m_pattern: {t2m_pattern}")

    tp = np.zeros((n_t, n_la, n_lo), dtype=np.float32)
    if tp_pattern == "dry_with_event":
        # One rain event on day 5 (10 mm = 0.01 m), otherwise zero
        rain_day = pd.Timestamp(start) + pd.Timedelta(days=5)
        mask = (times.normalize() == rain_day.normalize())
        tp[mask, :, :] = 0.01 / 24  # 10 mm spread over 24h, in metres
    elif tp_pattern == "dry":
        pass  # zeros
    elif tp_pattern == "wet_continuous":
        tp[:] = 0.002 / 24  # 2 mm/day every day
    elif tp_pattern == "very_wet":
        tp[:] = 0.010 / 24  # 10 mm/day every day → ~70 mm/week
    elif tp_pattern == "wet_with_dry_tail":
        # 2 mm/day for the first part of the run, then zero for the trailing week.
        # Satisfies precip_prev30d_mm ∈ [30, 150] while keeping prev24h/48h/7d at 0.
        days_offset = (times - pd.Timestamp(start)).days.to_numpy()
        total_days = days_offset.max() + 1
        dry_tail_start = max(0, total_days - 8)
        wet_mask = days_offset < dry_tail_start
        tp[wet_mask, :, :] = 0.002 / 24  # 2 mm/day during the wet phase
    else:
        raise ValueError(f"Unknown tp_pattern: {tp_pattern}")

    coords = {
        "time": times,
        "latitude": lats,
        "longitude": lons,
    }
    data_vars = {
        "t2m": (("time", "latitude", "longitude"), t2m),
        "tp": (("time", "latitude", "longitude"), tp),
    }

    if swvl1_pattern == "flat":
        swvl1 = np.full((n_t, n_la, n_lo), 0.25, dtype=np.float32)
        data_vars["swvl1"] = (("time", "latitude", "longitude"), swvl1)
    elif swvl1_pattern == "rising":
        ramp = np.linspace(0.15, 0.40, n_t, dtype=np.float32)
        swvl1 = np.broadcast_to(ramp[:, None, None], (n_t, n_la, n_lo)).copy()
        data_vars["swvl1"] = (("time", "latitude", "longitude"), swvl1)
    elif swvl1_pattern is None:
        pass

    if skt_pattern == "frost":
        skt = np.full((n_t, n_la, n_lo), 268.15, dtype=np.float32)
        data_vars["skt"] = (("time", "latitude", "longitude"), skt)
    elif skt_pattern == "warm":
        skt = np.full((n_t, n_la, n_lo), 288.15, dtype=np.float32)
        data_vars["skt"] = (("time", "latitude", "longitude"), skt)
    elif skt_pattern == "diurnal_freeze":
        # Peak ≈ +3 °C at 14 UTC, trough ≈ -5 °C at 02 UTC. Cosine centred on 14 UTC
        # so morning samples (e.g. 06 UTC) and afternoon samples (e.g. 14 UTC) differ.
        hour = times.hour.to_numpy()
        cycle = 271.15 + 5.0 * np.cos(2 * np.pi * (hour - 14) / 24)
        skt = np.broadcast_to(cycle[:, None, None], (n_t, n_la, n_lo)).astype(np.float32).copy()
        data_vars["skt"] = (("time", "latitude", "longitude"), skt)

    if sd_pattern == "snow":
        sd = np.full((n_t, n_la, n_lo), 0.10, dtype=np.float32)  # 10 cm
        data_vars["sd"] = (("time", "latitude", "longitude"), sd)
    elif sd_pattern == "no_snow":
        sd = np.zeros((n_t, n_la, n_lo), dtype=np.float32)
        data_vars["sd"] = (("time", "latitude", "longitude"), sd)

    if ssrd_pattern == "summer":
        # Sinusoidal day-night cycle, peak at noon. Amplitude chosen so the
        # integrated daily total comes out near ~18 MJ/m² — comfortably above
        # the 12 MJ/m² S2-extended threshold so the fixture isn't on the edge.
        hour = times.hour.to_numpy()
        peak = np.maximum(0, np.sin(np.pi * (hour - 6) / 12))
        ssrd_hourly = (peak * 2.4e6).astype(np.float32)
        ssrd = np.broadcast_to(ssrd_hourly[:, None, None], (n_t, n_la, n_lo)).copy()
        data_vars["ssrd"] = (("time", "latitude", "longitude"), ssrd)
    elif ssrd_pattern == "winter":
        ssrd = np.full((n_t, n_la, n_lo), 1e5, dtype=np.float32)  # very low
        data_vars["ssrd"] = (("time", "latitude", "longitude"), ssrd)

    return xr.Dataset(data_vars, coords=coords)


def make_era5_cloud_dataset(
    *,
    start: str,
    end: str,
    area=None,
    tcc_at_10utc: float = 0.10,
    lcc_at_10utc: float = 0.05,
) -> xr.Dataset:
    """Build a synthetic ERA5 single-levels cloud dataset.

    Sets all hourly TCC values to `tcc_at_10utc` to keep tests deterministic
    regardless of which hour the overpass sampler picks.
    """
    if area is None:
        area = TEST_AREA

    times = pd.date_range(start=start, end=end, freq="h", inclusive="left")
    lats, lons = _make_grid(area)
    shape = (len(times), len(lats), len(lons))

    tcc = np.full(shape, tcc_at_10utc, dtype=np.float32)
    lcc = np.full(shape, lcc_at_10utc, dtype=np.float32)

    return xr.Dataset(
        {
            "tcc": (("time", "latitude", "longitude"), tcc),
            "lcc": (("time", "latitude", "longitude"), lcc),
        },
        coords={"time": times, "latitude": lats, "longitude": lons},
    )


def write_netcdf(ds: xr.Dataset, path: Path) -> Path:
    """Write a dataset to NetCDF using scipy engine (NetCDF3, no h5netcdf required).

    Time is encoded as 'seconds since 1970-01-01' (float64) to sidestep an
    xarray+pandas-2.x scipy-backend bug that overflows the default 'hours
    since X' encoding when re-decoded without cftime.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        "time": {"units": "seconds since 1970-01-01", "dtype": "float64"},
    }
    ds.to_netcdf(path, engine="scipy", encoding=encoding)
    return path


@pytest.fixture
def era5_land_factory(tmp_path):
    """Factory: build + write an ERA5-Land NetCDF, return its path."""
    counter = {"n": 0}

    def _build(**kwargs) -> Path:
        counter["n"] += 1
        ds = make_era5_land_dataset(**kwargs)
        path = tmp_path / f"era5_land_{counter['n']}.nc"
        return write_netcdf(ds, path)

    return _build


@pytest.fixture
def era5_cloud_factory(tmp_path):
    """Factory: build + write an ERA5 cloud-cover NetCDF, return its path."""
    counter = {"n": 0}

    def _build(**kwargs) -> Path:
        counter["n"] += 1
        ds = make_era5_cloud_dataset(**kwargs)
        path = tmp_path / f"era5_clouds_{counter['n']}.nc"
        return write_netcdf(ds, path)

    return _build
