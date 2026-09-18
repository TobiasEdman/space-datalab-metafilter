"""Daily totals from hourly sources under both accumulation conventions."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from metafilter.core import (
    ACCUMULATION_ATTR,
    ACCUMULATION_HOURLY,
    MetafilterConfigurationError,
    calculate_daily_metrics,
)
from tests.conftest import as_forecast_start_accumulation

AREA = {"west": 18.09, "south": 59.29, "east": 18.11, "north": 59.31}


def _dataset(times, **variables):
    data = {
        name: (("time", "latitude", "longitude"), np.asarray(values, dtype=float).reshape(-1, 1, 1))
        for name, values in variables.items()
    }
    return xr.Dataset(data, coords={"time": times, "latitude": [59.3], "longitude": [18.1]})


def test_cds_precipitation_accumulation_yields_daily_total():
    times = pd.date_range("2024-08-01", periods=73, freq="h")
    # 1 mm every hour, delivered as ERA5-Land running totals in metres.
    tp = np.where(times.hour == 0, 24, times.hour).astype(float) / 1000.0
    result = calculate_daily_metrics(_dataset_path(_dataset(times, tp=tp)), area=AREA)
    assert result.loc[result.date == "2024-08-02", "total_precip_mm"].item() == pytest.approx(24.0)


def test_hourly_tagged_source_credits_midnight_sample_to_previous_day():
    times = pd.date_range("2024-08-01", periods=73, freq="h")
    tp = np.full(len(times), 1.0 / 1000.0)
    tp[times.hour == 0] = 5.0 / 1000.0  # the hour ending at 00:00 belongs to the day before
    dataset = _dataset(times, tp=tp)
    dataset.attrs[ACCUMULATION_ATTR] = ACCUMULATION_HOURLY
    result = calculate_daily_metrics(_dataset_path(dataset), area=AREA)
    assert result.loc[result.date == "2024-08-02", "total_precip_mm"].item() == pytest.approx(23 + 5)


def test_unknown_accumulation_tag_is_rejected():
    times = pd.date_range("2024-08-01", periods=24, freq="h")
    dataset = _dataset(times, tp=np.zeros(24))
    dataset.attrs[ACCUMULATION_ATTR] = "monthly"
    with pytest.raises(MetafilterConfigurationError, match="monthly"):
        calculate_daily_metrics(_dataset_path(dataset), area=AREA)


def test_fixture_accumulation_matches_era5_land_convention():
    times = pd.date_range("2024-08-01", periods=48, freq="h")
    increments = np.ones((48, 1, 1))
    accumulated = as_forecast_start_accumulation(increments, times)
    assert accumulated[1, 0, 0] == 1  # 01:00 restarts the running total
    assert accumulated[23, 0, 0] == 23
    assert accumulated[24, 0, 0] == 24  # 00:00 carries the previous day's total


_TMP = []


def _dataset_path(dataset):
    import tempfile
    handle = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
    handle.close()
    dataset.to_netcdf(handle.name, engine="scipy")
    _TMP.append(handle.name)
    return handle.name
