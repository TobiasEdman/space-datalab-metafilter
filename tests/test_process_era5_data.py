from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from scripts.process_era5 import (
    MetafilterSelectionError,
    apply_metafilter,
    calculate_daily_metrics,
    load_metafilter_parameters,
    normalize_metafilter_rules,
    process_era5_data,
    save_daily_metrics,
)
from tests.synthetic_era5 import TEST_AREA, era5_land_dataset, hourly_times, write_netcdf

SHIPPED_FILTER = Path(__file__).parent.parent / "filters" / "metafilter.json"

RULES: dict[str, Any] = {
    "temperature": {"threshold": 15.0, "unit": "Celsius"},
    "precipitation": {"threshold": 1.0, "unit": "mm/day"},
}


def test_warm_dry_days_are_selected(tmp_path: Path) -> None:
    # Every day is dry, so the temperature rule (> 15 °C) decides.
    temperatures_c = [12.0, 16.0, 20.0, 14.0, 18.0, 10.0]
    dataset = era5_land_dataset(hourly_times("2024-08-01", days=6), temperatures_c)
    path = write_netcdf(tmp_path, "six_days", dataset)

    results = process_era5_data(path, RULES, area=TEST_AREA)

    assert results["selected_dates"] == ["2024-08-02", "2024-08-03", "2024-08-05"]
    assert len(results["all_dates"]) == 6
    assert results["full_temporal_extent"] == ["2024-08-01", "2024-08-06"]
    assert results["selected_temporal_extent"] == ["2024-08-02", "2024-08-05"]
    assert [s["rule_name"] for s in results["rule_summaries"]] == [
        "temperature",
        "precipitation",
    ]


def test_no_matching_day_raises_with_diagnostics(tmp_path: Path) -> None:
    dataset = era5_land_dataset(hourly_times("2024-08-01", days=3), [5.0, 6.0, 7.0])
    path = write_netcdf(tmp_path, "cold_days", dataset)

    with pytest.raises(MetafilterSelectionError) as raised:
        process_era5_data(path, RULES, area=TEST_AREA)

    message = str(raised.value)
    assert "No dates matched the configured metafilter rules." in message
    assert "Evaluated 3 candidate day(s)." in message
    assert "Daily mean temperature: 0/3 days satisfied > 15.0 Celsius" in message
    assert raised.value.daily_metrics is not None
    assert len(raised.value.daily_metrics) == 3
    assert len(raised.value.rule_summaries) == 2


def test_shipped_filter_uses_columns_the_pipeline_produces(tmp_path: Path) -> None:
    dataset = era5_land_dataset(hourly_times("2024-08-01", days=2), [15.0, 15.0])
    metrics = calculate_daily_metrics(write_netcdf(tmp_path, "two_days", dataset), TEST_AREA)

    params = load_metafilter_parameters(SHIPPED_FILTER)
    _, summaries = apply_metafilter(metrics, params)

    shipped_rules = {rule["rule_name"] for rule in normalize_metafilter_rules(params)}
    assert {summary["rule_name"] for summary in summaries} == shipped_rules


def test_saved_metrics_round_trip(tmp_path: Path) -> None:
    metrics = pd.DataFrame({"date": ["2024-08-01"], "mean_temp_c": [16.5]})
    output = tmp_path / "nested" / "era5_daily_metrics.csv"

    save_daily_metrics(metrics, output)

    pd.testing.assert_frame_equal(pd.read_csv(output), metrics)
