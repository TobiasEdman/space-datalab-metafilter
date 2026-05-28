"""Derived-column tests — short windows, long windows, pass-time sampling.

Uses synthetic ERA5 NetCDF fixtures (see conftest.py) so the windowing logic
can be checked against known input shapes without ECMWF access.
"""
import numpy as np
import pandas as pd
import pytest

from scripts.process_era5 import calculate_daily_metrics


# ── Short-window derived columns ───────────────────────────────────────────

def test_short_window_columns_present(era5_land_factory, test_area):
    path = era5_land_factory(start="2024-08-01", end="2024-08-10", t2m_pattern="warm")
    df = calculate_daily_metrics(path, area=test_area)

    expected = {
        "date", "mean_temp_c", "min_temp_c", "max_temp_c", "freeze_flag",
        "total_precip_mm", "precip_prev24h_mm", "precip_prev48h_mm",
    }
    assert expected.issubset(df.columns)


def test_precip_prev24h_lags_by_one_day(era5_land_factory, test_area):
    """Dry first 5 days, 10 mm event on day 5 → prev24h is 0 on day 5, 10 on day 6."""
    path = era5_land_factory(start="2024-08-01", end="2024-08-10",
                             t2m_pattern="warm", tp_pattern="dry_with_event")
    df = calculate_daily_metrics(path, area=test_area)

    rain_idx = df.index[df["date"] == "2024-08-06"][0]  # day 5 (0-indexed → 5)
    assert df.loc[rain_idx, "total_precip_mm"] == pytest.approx(10.0, abs=0.1)
    # The day OF the event has prev24h = 0 (lag of 1 day)
    assert df.loc[rain_idx, "precip_prev24h_mm"] == pytest.approx(0.0, abs=0.1)
    # Day AFTER has prev24h ≈ 10
    assert df.loc[rain_idx + 1, "precip_prev24h_mm"] == pytest.approx(10.0, abs=0.1)


def test_freeze_flag_on_cold_dataset(era5_land_factory, test_area):
    path = era5_land_factory(start="2024-01-15", end="2024-01-20", t2m_pattern="cold")
    df = calculate_daily_metrics(path, area=test_area)
    assert df["freeze_flag"].all()


def test_freeze_flag_off_on_warm_dataset(era5_land_factory, test_area):
    path = era5_land_factory(start="2024-07-15", end="2024-07-20", t2m_pattern="warm")
    df = calculate_daily_metrics(path, area=test_area)
    assert not df["freeze_flag"].any()


# ── Long-window derived columns ────────────────────────────────────────────

def test_long_window_columns_have_nan_in_warmup_period(era5_land_factory, test_area):
    """7d window: first 7 rows should be NaN (rolling needs min_periods=7).
    30d window: first 30 rows NaN. Beyond that, values are deterministic."""
    path = era5_land_factory(start="2024-07-01", end="2024-08-15",
                             t2m_pattern="warm", tp_pattern="dry")
    df = calculate_daily_metrics(path, area=test_area)

    # 7d lookback at day index 7 means: rolling(7).sum() over days [0..6], shifted
    # by 1 → on day index 7 it sums days [0..6] which has min_periods met.
    # On day index < 7 → NaN.
    assert df.loc[:6, "precip_prev7d_mm"].isna().all()
    assert not pd.isna(df.loc[7, "precip_prev7d_mm"])

    # 30d window: 30 NaN at start
    assert df.loc[:29, "precip_prev30d_mm"].isna().all()
    assert not pd.isna(df.loc[30, "precip_prev30d_mm"])


def test_precip_prev30d_accumulates_correctly(era5_land_factory, test_area):
    """2 mm/day every day for 45 days → on day 31, precip_prev30d_mm = 60.0."""
    path = era5_land_factory(start="2024-06-01", end="2024-07-16",
                             t2m_pattern="warm", tp_pattern="wet_continuous")
    df = calculate_daily_metrics(path, area=test_area)

    # day index 30 → looks back at days [0..29] → 30 × 2 mm = 60 mm
    assert df.loc[30, "precip_prev30d_mm"] == pytest.approx(60.0, abs=0.5)


def test_dry_streak_counts_consecutive_dry_days(era5_land_factory, test_area):
    """Single rain event on day 5 — streak resets to 0 on day 6, grows after."""
    path = era5_land_factory(start="2024-08-01", end="2024-08-15",
                             t2m_pattern="warm", tp_pattern="dry_with_event")
    df = calculate_daily_metrics(path, area=test_area)

    # Days 0..4 are dry, day 5 has 10 mm event.
    # On day 5 the "prev" precip is day 4 (dry) → streak grows from initial.
    # On day 6 the "prev" precip is day 5 (wet) → streak resets to 0.
    # On day 7+ streak grows again from 1, 2, …
    day6_idx = df.index[df["date"] == "2024-08-07"][0]
    assert df.loc[day6_idx, "dry_streak_days"] == 0
    assert df.loc[day6_idx + 1, "dry_streak_days"] == 1
    assert df.loc[day6_idx + 5, "dry_streak_days"] == 5


def test_gdd_prev30d_accumulates_above_base(era5_land_factory, test_area):
    """Constant 20 °C (base 5 °C) → daily GDD = 15 → 30-day rolling sum = 450."""
    path = era5_land_factory(start="2024-06-01", end="2024-07-15", t2m_pattern="warm")
    df = calculate_daily_metrics(path, area=test_area)
    # warm pattern is constant 293.15 K = 20 °C → GDD daily = 15 → sum-30 = 450
    assert df.loc[30, "gdd_prev30d_c"] == pytest.approx(450.0, abs=2.0)


def test_swvl1_prev30d_mean_uses_rising_pattern(era5_land_factory, test_area):
    """Rising swvl1 from 0.15 → 0.40 over the run → 30-day mean stays inside."""
    path = era5_land_factory(start="2024-06-01", end="2024-07-15",
                             t2m_pattern="warm", swvl1_pattern="rising")
    df = calculate_daily_metrics(path, area=test_area)
    # day 30 mean of values from day 0..29 — should be in [0.15, 0.40]
    mean_30 = df.loc[30, "swvl1_prev30d_mean"]
    assert 0.15 < mean_30 < 0.40


# ── Pass-time sampling ─────────────────────────────────────────────────────

def test_skt_at_pass_matches_constant_temperature(era5_land_factory, test_area):
    """When skt is constant, sampling at any overpass returns that constant."""
    path = era5_land_factory(start="2024-07-01", end="2024-07-10",
                             t2m_pattern="warm", skt_pattern="warm")
    df = calculate_daily_metrics(path, area=test_area, overpass_time_utc="10:30")
    # All values should be 15 °C (288.15 K - 273.15)
    assert df["skt_at_pass_c"].dropna().between(14.9, 15.1).all()


def test_skt_at_pass_picks_correct_hour(era5_land_factory, test_area):
    """Diurnal freeze cycle: at 05:00 UTC skt is below freeze, at 14:00 above."""
    path = era5_land_factory(start="2024-04-01", end="2024-04-05",
                             t2m_pattern="warm", skt_pattern="diurnal_freeze")
    # Sample at 05:30 (rounds up to 06) — still cold side of cycle
    df_morning = calculate_daily_metrics(path, area=test_area, overpass_time_utc="05:30")
    # Sample at 14:00 — peak warmth
    df_afternoon = calculate_daily_metrics(path, area=test_area, overpass_time_utc="14:00")
    assert df_morning["skt_at_pass_c"].mean() < df_afternoon["skt_at_pass_c"].mean()


# ── Cloud-cover sampling (separate NetCDF) ─────────────────────────────────

def test_cloud_cover_columns_appear_when_file_supplied(
    era5_land_factory, era5_cloud_factory, test_area
):
    land = era5_land_factory(start="2024-08-01", end="2024-08-10", t2m_pattern="warm")
    cloud = era5_cloud_factory(start="2024-08-01", end="2024-08-10",
                               tcc_at_10utc=0.15, lcc_at_10utc=0.08)
    df = calculate_daily_metrics(land, area=test_area, cloud_file_path=cloud)

    assert "tcc_mean_overpass" in df.columns
    assert "lcc_mean_overpass" in df.columns
    # Cloud values are constant in fixture so mean ≈ set value
    assert df["tcc_mean_overpass"].dropna().between(0.14, 0.16).all()
    assert df["lcc_mean_overpass"].dropna().between(0.07, 0.09).all()


def test_cloud_columns_absent_when_no_file(era5_land_factory, test_area):
    path = era5_land_factory(start="2024-08-01", end="2024-08-05", t2m_pattern="warm")
    df = calculate_daily_metrics(path, area=test_area)
    assert "tcc_mean_overpass" not in df.columns
    assert "lcc_mean_overpass" not in df.columns


# ── Multi-file (buffer-month) handling ─────────────────────────────────────

def test_buffer_month_concat_enables_long_lookback(era5_land_factory, test_area):
    """Two contiguous months concatenated → first day of month 2 has its
    precip_prev30d_mm fully resolved (no NaN), because the buffer month
    supplied the trailing data."""
    buffer = era5_land_factory(start="2024-07-01", end="2024-08-01",
                               t2m_pattern="warm", tp_pattern="wet_continuous")
    primary = era5_land_factory(start="2024-08-01", end="2024-09-01",
                                t2m_pattern="warm", tp_pattern="dry")

    df = calculate_daily_metrics([buffer, primary], area=test_area)

    # Day 31 (i.e. first day of August) should have a non-NaN precip_prev30d_mm
    # because we now have 31 days of buffer behind it.
    aug_first_idx = df.index[df["date"] == "2024-08-01"][0]
    assert not pd.isna(df.loc[aug_first_idx, "precip_prev30d_mm"])
    # July had 2 mm/day continuous → 30 days × 2 mm = 60 mm
    assert df.loc[aug_first_idx, "precip_prev30d_mm"] == pytest.approx(60.0, abs=0.5)
