"""Tests for the surface-state derived columns + pass-time sampling added in this PR.

Covers `skt_at_pass_c`, `skt_mean_c`, `skt_min_c`, `stl1_mean_c`,
`swvl1_mean`, `swvl1_delta_prev2d`, `swvl1_prev30d_mean`,
`snow_depth_mean_m`. All are emitted only when their source variables
are present in the dataset.
"""
import pandas as pd
import pytest

from scripts.process_era5 import calculate_daily_metrics


def test_skt_columns_only_when_present(era5_land_factory, test_area):
    path_no = era5_land_factory(start="2024-08-01", end="2024-08-05", t2m_pattern="warm")
    df = calculate_daily_metrics(path_no, area=test_area)
    assert "skt_at_pass_c" not in df.columns

    path = era5_land_factory(start="2024-08-01", end="2024-08-05",
                             t2m_pattern="warm", skt_pattern="warm")
    df = calculate_daily_metrics(path, area=test_area)
    assert {"skt_mean_c", "skt_min_c", "skt_at_pass_c"}.issubset(df.columns)


def test_skt_at_pass_picks_correct_hour(era5_land_factory, test_area):
    """Diurnal-freeze cycle: morning sample (05:30 → 06 UTC) is colder than
    afternoon sample (14:00 UTC)."""
    path = era5_land_factory(start="2024-04-01", end="2024-04-05",
                             t2m_pattern="warm", skt_pattern="diurnal_freeze")
    morning = calculate_daily_metrics(path, area=test_area, overpass_time_utc="05:30")
    afternoon = calculate_daily_metrics(path, area=test_area, overpass_time_utc="14:00")
    assert morning["skt_at_pass_c"].mean() < afternoon["skt_at_pass_c"].mean()


def test_swvl1_columns_only_when_present(era5_land_factory, test_area):
    path_no = era5_land_factory(start="2024-08-01", end="2024-08-05", t2m_pattern="warm")
    df = calculate_daily_metrics(path_no, area=test_area)
    assert "swvl1_mean" not in df.columns

    path = era5_land_factory(start="2024-06-01", end="2024-07-15",
                             t2m_pattern="warm", swvl1_pattern="flat")
    df = calculate_daily_metrics(path, area=test_area)
    assert {"swvl1_mean", "swvl1_delta_prev2d", "swvl1_prev30d_mean"}.issubset(df.columns)


def test_swvl1_prev30d_mean_falls_inside_rising_range(era5_land_factory, test_area):
    """Rising swvl1 from 0.15 → 0.40 over the run; 30-day mean stays inside that band."""
    path = era5_land_factory(start="2024-06-01", end="2024-07-15",
                             t2m_pattern="warm", swvl1_pattern="rising")
    df = calculate_daily_metrics(path, area=test_area)
    mean_30 = df.loc[30, "swvl1_prev30d_mean"]
    assert 0.15 < mean_30 < 0.40


def test_snow_depth_column_only_when_present(era5_land_factory, test_area):
    path_no = era5_land_factory(start="2024-08-01", end="2024-08-05", t2m_pattern="warm")
    df = calculate_daily_metrics(path_no, area=test_area)
    assert "snow_depth_mean_m" not in df.columns

    path = era5_land_factory(start="2024-02-01", end="2024-02-10",
                             t2m_pattern="cold", sd_pattern="snow")
    df = calculate_daily_metrics(path, area=test_area)
    assert df["snow_depth_mean_m"].between(0.09, 0.11).all()
