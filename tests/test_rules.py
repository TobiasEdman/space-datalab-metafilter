from typing import Any

import pytest

from scripts.process_era5 import (
    MetafilterConfigurationError,
    format_rule_brief,
    normalize_metafilter_rules,
)


def test_legacy_rule_names_supply_defaults() -> None:
    [rule] = normalize_metafilter_rules({"temperature": {"threshold": 12.0}})

    assert rule["metric_column"] == "mean_temp_c"
    assert rule["operator"] == "gt"
    assert rule["name"] == "Daily mean temperature"
    assert rule["unit"] is None


def test_explicit_fields_override_defaults() -> None:
    [rule] = normalize_metafilter_rules(
        {"temperature": {"threshold": 12.0, "operator": "ge", "name": "Warm day"}}
    )

    assert rule["operator"] == "ge"
    assert rule["name"] == "Warm day"


def test_rule_name_becomes_display_name() -> None:
    [rule] = normalize_metafilter_rules(
        {"soil_moisture": {"metric_column": "swvl1", "operator": "lt", "threshold": 0.3}}
    )

    assert rule["name"] == "Soil Moisture"


@pytest.mark.parametrize(
    ("rule", "message"),
    [
        ({"metric_column": "mean_temp_c", "operator": "gt"}, "missing 'threshold'"),
        ({"operator": "gt", "threshold": 1.0}, "missing 'metric_column'"),
        (
            {"metric_column": "mean_temp_c", "operator": "nonsense", "threshold": 1.0},
            "unsupported operator 'nonsense'",
        ),
    ],
)
def test_invalid_rule_is_rejected(rule: dict[str, Any], message: str) -> None:
    with pytest.raises(MetafilterConfigurationError, match=message):
        normalize_metafilter_rules({"custom": rule})


def test_rule_brief_names_column_and_condition() -> None:
    [rule] = normalize_metafilter_rules(
        {"temperature": {"threshold": 15.0, "unit": "Celsius"}}
    )

    assert format_rule_brief(rule) == "Daily mean temperature (mean_temp_c > 15.0 Celsius)"
