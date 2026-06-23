"""Tests for the comparison operators in `COMPARISON_OPERATORS`.

Each operator is exercised against a pandas Series and asserted on
specific values. `between` and `abs_lt` are the new additions; the
classic `gt/ge/lt/le` operators are also covered for completeness.
"""
import pandas as pd
import pytest

from scripts.process_era5 import (
    COMPARISON_OPERATORS,
    MetafilterConfigurationError,
    normalize_metafilter_rules,
)


def _apply(op, series, threshold):
    return COMPARISON_OPERATORS[op]["apply"](pd.Series(series), threshold)


def test_gt_strict():
    assert _apply("gt", [1, 2, 3], 2).tolist() == [False, False, True]


def test_ge_inclusive():
    assert _apply("ge", [1, 2, 3], 2).tolist() == [False, True, True]


def test_lt_strict():
    assert _apply("lt", [1, 2, 3], 2).tolist() == [True, False, False]


def test_le_inclusive():
    assert _apply("le", [1, 2, 3], 2).tolist() == [True, True, False]


def test_between_inclusive_both_ends():
    assert _apply("between", [0.0, 0.15, 0.25, 0.40, 0.45], [0.15, 0.40]).tolist() == [
        False, True, True, True, False,
    ]


def test_between_accepts_tuple_threshold():
    assert _apply("between", [10, 50, 100], (30, 150)).tolist() == [False, True, True]


def test_abs_lt_handles_negative_delta():
    assert _apply("abs_lt", [-0.10, -0.04, 0.0, 0.03, 0.06], 0.05).tolist() == [
        False, True, True, True, False,
    ]


def test_between_threshold_must_be_two_elements():
    payload = {
        "rules": {
            "bad": {
                "metric_column": "precip_prev30d_mm",
                "operator": "between",
                "threshold": 50.0,
            }
        }
    }
    with pytest.raises(MetafilterConfigurationError, match="not \\[low, high\\]"):
        normalize_metafilter_rules(payload)


def test_between_rejects_low_above_high():
    payload = {
        "rules": {
            "bad": {
                "metric_column": "precip_prev30d_mm",
                "operator": "between",
                "threshold": [150.0, 30.0],
            }
        }
    }
    with pytest.raises(MetafilterConfigurationError, match="low > high"):
        normalize_metafilter_rules(payload)


def test_symbols_are_distinct():
    """Every operator gets a unique symbol so report formatting can't collide."""
    symbols = {op_cfg["symbol"] for op_cfg in COMPARISON_OPERATORS.values()}
    assert len(symbols) == len(COMPARISON_OPERATORS)
