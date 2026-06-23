"""Tests for the cloud-cover columns added in this PR.

`tcc_mean_overpass` and `lcc_mean_overpass` are emitted when:
1. `tcc` / `lcc` are present in the main land dataset (alternate-backend path), or
2. A separate `cloud_file_path` is supplied (CDS-Beta path: cloud cover lives
   in `reanalysis-era5-single-levels`, not `reanalysis-era5-land`).
"""
import pytest

from scripts.process_era5 import calculate_daily_metrics


def test_cloud_columns_absent_when_neither_source_given(era5_land_factory, test_area):
    path = era5_land_factory(start="2024-08-01", end="2024-08-05", t2m_pattern="warm")
    df = calculate_daily_metrics(path, area=test_area)
    assert "tcc_mean_overpass" not in df.columns
    assert "lcc_mean_overpass" not in df.columns


def test_cloud_columns_present_when_cloud_file_path_supplied(
    era5_land_factory, era5_cloud_factory, test_area
):
    land = era5_land_factory(start="2024-08-01", end="2024-08-10", t2m_pattern="warm")
    cloud = era5_cloud_factory(start="2024-08-01", end="2024-08-10",
                               tcc_at_overpass=0.15, lcc_at_overpass=0.08)
    df = calculate_daily_metrics(land, area=test_area, cloud_file_path=cloud)

    assert "tcc_mean_overpass" in df.columns
    assert "lcc_mean_overpass" in df.columns
    assert df["tcc_mean_overpass"].dropna().between(0.14, 0.16).all()
    assert df["lcc_mean_overpass"].dropna().between(0.07, 0.09).all()
