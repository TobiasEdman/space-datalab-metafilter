import math

import pandas as pd
import pytest

from scripts.process_era5 import apply_metafilter


def rule_result(operator: str, values: list[float]) -> list[bool]:
    rules = {"check": {"metric_column": "value", "operator": operator, "threshold": 1.0}}
    filtered, _ = apply_metafilter(pd.DataFrame({"value": values}), rules)
    result: list[bool] = filtered["rule__check"].tolist()
    return result


@pytest.mark.parametrize(
    ("operator", "expected"),
    [
        ("gt", [False, False, True]),
        ("ge", [False, True, True]),
        ("lt", [True, False, False]),
        ("le", [True, True, False]),
    ],
)
def test_operator_at_threshold(operator: str, expected: list[bool]) -> None:
    assert rule_result(operator, [0.5, 1.0, 1.5]) == expected


@pytest.mark.parametrize("operator", ["gt", "ge", "lt", "le"])
def test_missing_value_never_passes(operator: str) -> None:
    assert rule_result(operator, [math.nan]) == [False]
