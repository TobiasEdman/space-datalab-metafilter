"""Tests for the long-window derived columns added in this PR.

Covers `precip_prev7d_mm`, `precip_prev30d_mm`, `ssrd_mj_m2`,
`ssrd_prev30d_mj_m2`, `gdd_prev30d_c`, `dry_streak_days`, and the
multi-file (buffer-month) concatenation path.
"""
import pandas as pd
import pytest

from scripts.process_era5 import calculate_daily_metrics


def test_long_window_columns_present(era5_land_factory, test_area):
    path = era5_land_factory(start="2024-07-01", end="2024-08-15", t2m_pattern="warm")
    df = calculate_daily_metrics(path, area=test_area)
    expected = {
        "precip_prev7d_mm", "precip_prev30d_mm",
        "dry_streak_days", "gdd_prev30d_c",
    }
    assert expected.issubset(df.columns)


def test_long_window_columns_are_nan_in_warmup(era5_land_factory, test_area):
    """7d rolling needs 7 days of history → first 7 rows NaN. Same for 30d."""
    path = era5_land_factory(start="2024-07-01", end="2024-08-15",
                             t2m_pattern="warm", tp_pattern="dry")
    df = calculate_daily_metrics(path, area=test_area)
    assert df.loc[:6, "precip_prev7d_mm"].isna().all()
    assert not pd.isna(df.loc[7, "precip_prev7d_mm"])
    assert df.loc[:29, "precip_prev30d_mm"].isna().all()
    assert not pd.isna(df.loc[30, "precip_prev30d_mm"])


def test_precip_prev30d_sums_correctly(era5_land_factory, test_area):
    """2 mm/day every day → on day 30, prev30d = 30 * 2 = 60 mm."""
    path = era5_land_factory(start="2024-06-01", end="2024-07-16",
                             t2m_pattern="warm", tp_pattern="wet_continuous")
    df = calculate_daily_metrics(path, area=test_area)
    assert df.loc[30, "precip_prev30d_mm"] == pytest.approx(60.0, abs=0.5)


def test_dry_streak_counts_consecutive_dry_days(era5_land_factory, test_area):
    """Day 5 has a 10 mm event → streak resets to 0 on day 6, grows afterwards."""
    path = era5_land_factory(start="2024-08-01", end="2024-08-15",
                             t2m_pattern="warm", tp_pattern="dry_with_event")
    df = calculate_daily_metrics(path, area=test_area)
    day6 = df.index[df["date"] == "2024-08-07"][0]
    assert df.loc[day6, "dry_streak_days"] == 0
    assert df.loc[day6 + 1, "dry_streak_days"] == 1
    assert df.loc[day6 + 5, "dry_streak_days"] == 5


def test_gdd_prev30d_accumulates_above_base(era5_land_factory, test_area):
    """Constant 20 °C, base 5 °C → daily GDD = 15 → 30-day sum = 450."""
    path = era5_land_factory(start="2024-06-01", end="2024-07-15", t2m_pattern="warm")
    df = calculate_daily_metrics(path, area=test_area)
    assert df.loc[30, "gdd_prev30d_c"] == pytest.approx(450.0, abs=2.0)


def test_ssrd_columns_only_when_present(era5_land_factory, test_area):
    """Without ssrd in the dataset, no ssrd_* columns; with, both."""
    path_no_ssrd = era5_land_factory(start="2024-08-01", end="2024-08-10", t2m_pattern="warm")
    df_no = calculate_daily_metrics(path_no_ssrd, area=test_area)
    assert "ssrd_mj_m2" not in df_no.columns

    path_ssrd = era5_land_factory(start="2024-06-01", end="2024-07-15",
                                  t2m_pattern="warm", ssrd_pattern="summer")
    df_yes = calculate_daily_metrics(path_ssrd, area=test_area)
    assert "ssrd_mj_m2" in df_yes.columns
    assert "ssrd_prev30d_mj_m2" in df_yes.columns
    # ssrd_prev30d_mj_m2 needs a 30-day warm-up
    assert df_yes.loc[:29, "ssrd_prev30d_mj_m2"].isna().all()
    assert not pd.isna(df_yes.loc[30, "ssrd_prev30d_mj_m2"])


def test_buffer_month_concat_enables_long_lookback(era5_land_factory, test_area):
    """Pass a list of NetCDFs → time concatenated → day-1-of-month-2 has
    its precip_prev30d_mm resolved without NaN, because the buffer month
    supplied the trailing window."""
    buffer = era5_land_factory(start="2024-07-01", end="2024-08-01",
                               t2m_pattern="warm", tp_pattern="wet_continuous")
    primary = era5_land_factory(start="2024-08-01", end="2024-09-01",
                                t2m_pattern="warm", tp_pattern="dry")
    df = calculate_daily_metrics([buffer, primary], area=test_area)

    aug_first = df.index[df["date"] == "2024-08-01"][0]
    assert not pd.isna(df.loc[aug_first, "precip_prev30d_mm"])
    # July had 2 mm/day × 30 = 60 mm
    assert df.loc[aug_first, "precip_prev30d_mm"] == pytest.approx(60.0, abs=0.5)
