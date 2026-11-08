from pathlib import Path

import pytest

from scripts.process_era5 import MetafilterSelectionError, calculate_daily_metrics
from tests.synthetic_era5 import (
    GRID_LATITUDES,
    TEST_AREA,
    era5_land_dataset,
    hourly_times,
    write_netcdf,
)

TEMPERATURES_C = [12.0, 16.0, 20.0]


def test_daily_mean_temperature_over_the_area(tmp_path: Path) -> None:
    dataset = era5_land_dataset(hourly_times("2024-08-01", days=3), TEMPERATURES_C)
    path = write_netcdf(tmp_path, "three_days", dataset)

    metrics = calculate_daily_metrics(path, area=TEST_AREA)

    assert metrics["date"].tolist() == ["2024-08-01", "2024-08-02", "2024-08-03"]
    assert metrics["mean_temp_c"].tolist() == pytest.approx(TEMPERATURES_C, abs=1e-3)
    assert "total_precip_mm" in metrics.columns


def test_south_to_north_latitudes(tmp_path: Path) -> None:
    dataset = era5_land_dataset(
        hourly_times("2024-08-01", days=3),
        TEMPERATURES_C,
        latitudes=sorted(GRID_LATITUDES),
    )
    path = write_netcdf(tmp_path, "ascending_latitude", dataset)

    metrics = calculate_daily_metrics(path, area=TEST_AREA)

    assert metrics["mean_temp_c"].tolist() == pytest.approx(TEMPERATURES_C, abs=1e-3)


def test_time_dimension_named_time(tmp_path: Path) -> None:
    dataset = era5_land_dataset(
        hourly_times("2024-08-01", days=3), TEMPERATURES_C, time_name="time"
    )
    path = write_netcdf(tmp_path, "time_dimension", dataset)

    metrics = calculate_daily_metrics(path, area=TEST_AREA)

    assert metrics["mean_temp_c"].tolist() == pytest.approx(TEMPERATURES_C, abs=1e-3)


def test_area_outside_the_file_is_rejected(tmp_path: Path) -> None:
    dataset = era5_land_dataset(hourly_times("2024-08-01", days=1), [15.0])
    path = write_netcdf(tmp_path, "one_day", dataset)
    elsewhere = {"west": 12.0, "east": 12.2, "south": 56.0, "north": 56.2}

    with pytest.raises(MetafilterSelectionError, match="does not overlap"):
        calculate_daily_metrics(path, area=elsewhere)
