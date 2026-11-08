import math
from typing import Any

import pandas as pd
import pytest

from scripts.process_era5 import (
    MetafilterConfigurationError,
    apply_metafilter,
    format_rule_summary,
)

RULES: dict[str, Any] = {
    "temperature": {"threshold": 15.0},
    "precipitation": {"threshold": 1.0},
}


def daily_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2024-08-01", "2024-08-02", "2024-08-03", "2024-08-04"],
            "mean_temp_c": [10.0, 16.0, 20.0, math.nan],
            "total_precip_mm": [0.0, 5.0, 0.0, 0.0],
        }
    )


def test_a_day_is_selected_only_when_every_rule_passes() -> None:
    filtered, _ = apply_metafilter(daily_metrics(), RULES)

    assert filtered["rule__temperature"].tolist() == [False, True, True, False]
    assert filtered["rule__precipitation"].tolist() == [True, False, True, True]
    assert filtered["selected"].tolist() == [False, False, True, False]


def test_input_frame_is_not_modified() -> None:
    metrics = daily_metrics()
    apply_metafilter(metrics, RULES)

    assert list(metrics.columns) == ["date", "mean_temp_c", "total_precip_mm"]


def test_summary_counts_valid_and_passing_days() -> None:
    _, summaries = apply_metafilter(daily_metrics(), RULES)
    temperature = next(s for s in summaries if s["rule_name"] == "temperature")

    assert temperature["valid_days"] == 3
    assert temperature["passed_days"] == 2
    assert temperature["observed_min"] == 10.0
    assert temperature["observed_max"] == 20.0


def test_summary_of_a_metric_without_values() -> None:
    metrics = daily_metrics().assign(mean_temp_c=math.nan)
    _, summaries = apply_metafilter(metrics, RULES)
    temperature = next(s for s in summaries if s["rule_name"] == "temperature")

    assert temperature["valid_days"] == 0
    assert temperature["observed_min"] is None
    assert "no valid values were available for metric 'mean_temp_c'" in (
        format_rule_summary(temperature, total_days=4)
    )


def test_rule_on_a_missing_column_is_rejected() -> None:
    rules = {"moisture": {"metric_column": "swvl1", "operator": "lt", "threshold": 0.3}}

    with pytest.raises(MetafilterConfigurationError, match="requires metric column 'swvl1'"):
        apply_metafilter(daily_metrics(), rules)
