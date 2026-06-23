"""Tests for the short-window derived columns added in this PR.

Verifies:
* `min_temp_c`, `max_temp_c`, `freeze_flag` follow the temperature pattern
* `precip_prev24h_mm`, `precip_prev48h_mm` lag correctly by 1/2 days
"""
import pandas as pd
import pytest

from scripts.process_era5 import calculate_daily_metrics


def test_short_window_columns_present(era5_land_factory, test_area):
    path = era5_land_factory(start="2024-08-01", end="2024-08-10", t2m_pattern="warm")
    df = calculate_daily_metrics(path, area=test_area)

    expected = {
        "date", "mean_temp_c", "min_temp_c", "max_temp_c", "freeze_flag",
        "total_precip_mm", "precip_prev24h_mm", "precip_prev48h_mm",
    }
    assert expected.issubset(df.columns)


def test_precip_prev24h_lags_by_one_day(era5_land_factory, test_area):
    """10 mm event on day 5 → prev24h is 0 on day 5, 10 on day 6."""
    path = era5_land_factory(
        start="2024-08-01", end="2024-08-10",
        t2m_pattern="warm", tp_pattern="dry_with_event",
    )
    df = calculate_daily_metrics(path, area=test_area)

    rain_idx = df.index[df["date"] == "2024-08-06"][0]
    assert df.loc[rain_idx, "total_precip_mm"] == pytest.approx(10.0, abs=0.1)
    assert df.loc[rain_idx, "precip_prev24h_mm"] == pytest.approx(0.0, abs=0.1)
    assert df.loc[rain_idx + 1, "precip_prev24h_mm"] == pytest.approx(10.0, abs=0.1)


def test_precip_prev48h_sums_two_days(era5_land_factory, test_area):
    """2 mm/day every day → on day 3, prev48h is the sum of days 1+2 = 4 mm."""
    path = era5_land_factory(
        start="2024-08-01", end="2024-08-10",
        t2m_pattern="warm", tp_pattern="wet_continuous",
    )
    df = calculate_daily_metrics(path, area=test_area)
    # day index 2 (i.e. 2024-08-03) → prev48h sums day 0 + day 1 = 4 mm
    assert df.loc[2, "precip_prev48h_mm"] == pytest.approx(4.0, abs=0.2)


def test_freeze_flag_on_cold_dataset(era5_land_factory, test_area):
    path = era5_land_factory(start="2024-01-15", end="2024-01-20", t2m_pattern="cold")
    df = calculate_daily_metrics(path, area=test_area)
    assert df["freeze_flag"].all()


def test_freeze_flag_off_on_warm_dataset(era5_land_factory, test_area):
    path = era5_land_factory(start="2024-07-15", end="2024-07-20", t2m_pattern="warm")
    df = calculate_daily_metrics(path, area=test_area)
    assert not df["freeze_flag"].any()
