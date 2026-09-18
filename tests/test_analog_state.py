"""AnalogModel keeps a consistent fit and refuses undated candidates."""
import pandas as pd
import pytest

from metafilter.analog import AnalogModel


def _frame(dates, values):
    return pd.DataFrame({"date": dates, "a": values})


def test_undated_rows_are_counted_as_rejected():
    model = AnalogModel(features=["a"], metric="euclidean").fit(
        {2020: _frame(["2020-01-01"], [0.0]), 2021: _frame([None, "2021-01-01"], [0.0, 1.0])}
    )
    assert model.rejected_rows == {2020: 0, 2021: 1}
    assert [m.date for m in model.query(reference_year=2020, reference_date="2020-01-01")] == ["2021-01-01"]


def test_failed_refit_leaves_previous_fit_untouched():
    model = AnalogModel(features=["a"], metric="euclidean").fit(
        {2020: _frame(["2020-01-01"], [0.0]), 2021: _frame(["2021-01-01"], [3.0])}
    )
    before = model.query(reference_year=2020, reference_date="2020-01-01")[0]

    with pytest.raises(ValueError, match="missing columns"):
        model.fit({2020: _frame(["2020-01-01"], [100.0]), 2022: pd.DataFrame()})

    after = model.query(reference_year=2020, reference_date="2020-01-01")[0]
    assert after == before
    assert model.rejected_rows == {2020: 0, 2021: 0}
