"""Rolling windows come from the requested column names."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from metafilter.core import (
    DEFAULT_LOOKBACK_COLUMNS,
    MetafilterConfigurationError,
    calculate_daily_metrics,
    lookback_days,
    parse_lookback,
    process_era5_data,
)

AREA = {"west": 18.09, "south": 59.29, "east": 18.11, "north": 59.31}


def _rain(path, days, mm_per_day):
    """One month of hourly rain, delivered in CDS's accumulation shape."""
    times = pd.date_range("2024-08-01", periods=days * 24 + 1, freq="h")
    hourly = np.where(times.hour == 0, 24, times.hour) * (mm_per_day / 24.0) / 1000.0
    xr.Dataset(
        {"tp": (("time", "latitude", "longitude"), hourly.reshape(-1, 1, 1))},
        coords={"time": times, "latitude": [59.3], "longitude": [18.1]},
    ).to_netcdf(path, engine="scipy")
    return path


def test_parse_lookback_reads_span_and_aggregation():
    assert parse_lookback("precip_prev14d_mm") == ("precip", 14, "sum")
    assert parse_lookback("precip_prev48h_mm") == ("precip", 2, "sum")
    assert parse_lookback("swvl1_prev30d_mean") == ("swvl1", 30, "mean")
    assert parse_lookback("mean_temp_c") is None
    assert parse_lookback("swvl1_delta_prev2d") is None


def test_partial_hour_window_is_rejected():
    with pytest.raises(MetafilterConfigurationError, match="whole days"):
        parse_lookback("precip_prev36h_mm")


def test_longest_window_wins():
    assert lookback_days(["precip_prev7d_mm", "gdd_prev30d_c", "mean_temp_c"]) == 30
    assert lookback_days(["mean_temp_c"]) == 0


def test_a_window_never_shipped_before(tmp_path):
    """14 days needs no code change - only the name."""
    frame = calculate_daily_metrics(
        _rain(tmp_path / "aug.nc", 31, 2.0), area=AREA,
        lookback_columns=("precip_prev14d_mm",),
    )
    assert "precip_prev14d_mm" in frame.columns
    assert pd.isna(frame["precip_prev14d_mm"].iloc[13])          # warm-up
    assert frame["precip_prev14d_mm"].iloc[14] == pytest.approx(28.0, abs=0.1)


def test_default_columns_are_unchanged(tmp_path):
    frame = calculate_daily_metrics(_rain(tmp_path / "aug.nc", 31, 2.0), area=AREA)
    for column in DEFAULT_LOOKBACK_COLUMNS:
        if column.startswith("precip"):
            assert column in frame.columns
    assert "dry_streak_days" in frame.columns


def test_profile_window_reaches_the_metrics_frame(tmp_path):
    """A rule naming a 14-day window gets that column computed for it."""
    result = process_era5_data(
        _rain(tmp_path / "aug.nc", 31, 1.0),
        {"rules": {"wet_fortnight": {"metric_column": "precip_prev14d_mm",
                                     "operator": "lt", "threshold": 100.0}}},
        area=AREA,
    )
    assert "precip_prev14d_mm" in result["daily_metrics"].columns
    assert len(result["selected_dates"]) > 0


def test_unknown_lookback_column_is_rejected(tmp_path):
    with pytest.raises(MetafilterConfigurationError, match="not a recognised lookback"):
        calculate_daily_metrics(
            _rain(tmp_path / "aug.nc", 5, 1.0), area=AREA,
            lookback_columns=("humidity_prev7d_pct",),
        )
