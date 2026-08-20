from __future__ import annotations

import runpy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from metafilter import AnalogModel, fetch_daily_meteorology


FEATURES = (
    "gdd_prev30d_c",
    "precip_prev30d_mm",
    "precip_prev7d_mm",
    "swvl1_prev30d_mean",
    "ssrd_prev30d_mj_m2",
)


def _frame(year: int, vectors: list[list[float]]) -> pd.DataFrame:
    data = {"date": pd.date_range(f"{year}-05-15", periods=len(vectors))}
    for index, name in enumerate(FEATURES):
        data[name] = [row[index] for row in vectors]
    return pd.DataFrame(data)


def test_analog_model_recovers_exact_match():
    reference = [300.0, 40.0, 5.0, 0.30, 500.0]
    frames = {
        2019: _frame(2019, [reference, [330, 70, 10, 0.4, 550]]),
        2020: _frame(2020, [[350, 80, 14, 0.5, 600], reference]),
    }
    match = AnalogModel(features=FEATURES).fit(frames).query(
        reference_year=2019,
        reference_date="2019-05-15",
        candidate_years=[2020],
        limit=1,
    )[0]
    assert match.date == "2020-05-16"
    assert match.distance == 0.0
    assert set(match.normalized_deltas) == set(FEATURES)


def test_analog_model_regularizes_collinear_features():
    frames = {
        2019: _frame(2019, [[1, 2, 3, 4, 5], [2, 4, 6, 8, 10]]),
        2020: _frame(2020, [[3, 6, 9, 12, 15], [4, 8, 12, 16, 20]]),
    }
    matches = AnalogModel(features=FEATURES).fit(frames).query(
        reference_year=2019,
        reference_date="2019-05-15",
        candidate_years=[2020],
    )
    assert matches
    assert all(np.isfinite(match.distance) for match in matches)


def test_analog_model_reports_and_rejects_missing_rows():
    values = [[1, 2, 3, 4, 5], [2, 3, np.nan, 5, 6]]
    model = AnalogModel(features=FEATURES).fit(
        {2019: _frame(2019, values), 2020: _frame(2020, values)}
    )
    assert model.rejected_rows == {2019: 1, 2020: 1}


def test_analog_model_rejects_infinite_values_and_weights():
    values = [[1, 2, 3, 4, 5], [2, 3, np.inf, 5, 6]]
    model = AnalogModel(features=FEATURES).fit(
        {2019: _frame(2019, values), 2020: _frame(2020, values)}
    )
    assert model.rejected_rows == {2019: 1, 2020: 1}
    for weight in (np.inf, np.nan, -1.0):
        with np.testing.assert_raises(ValueError):
            AnalogModel(features=FEATURES, weights={FEATURES[0]: weight})
    for regularization in (np.inf, np.nan):
        with np.testing.assert_raises(ValueError):
            AnalogModel(features=FEATURES, regularization=regularization)
    with np.testing.assert_raises(ValueError):
        AnalogModel(features=[FEATURES[0], FEATURES[0]])


def test_mahalanobis_weights_change_distance():
    frames = {
        2019: _frame(2019, [[0, 0, 0, 0, 0], [2, 1, 2, 1, 2]]),
        2020: _frame(2020, [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]),
    }
    unweighted = AnalogModel(features=FEATURES).fit(frames).query(
        reference_year=2019, reference_date="2019-05-15", candidate_years=[2020]
    )
    weighted = AnalogModel(
        features=FEATURES, weights={FEATURES[0]: 25.0}
    ).fit(frames).query(
        reference_year=2019, reference_date="2019-05-15", candidate_years=[2020]
    )
    assert [match.distance for match in weighted] != [
        match.distance for match in unweighted
    ]


def test_query_limit_boundaries():
    model = AnalogModel(features=FEATURES).fit(
        {
            2019: _frame(2019, [[0, 0, 0, 0, 0]]),
            2020: _frame(2020, [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2]]),
        }
    )
    kwargs = dict(reference_year=2019, reference_date="2019-05-15")
    assert model.query(**kwargs, limit=0) == []
    assert len(model.query(**kwargs, limit=1)) == 1
    with np.testing.assert_raises(ValueError):
        model.query(**kwargs, limit=-1)


def test_fetch_daily_meteorology_caches_months(tmp_path):
    bbox = {"west": 18.0, "south": 59.2, "east": 18.2, "north": 59.4}
    def fake_fetch(year, month, **kwargs):
        month_times = pd.date_range(
            f"{year}-{month:02d}-01",
            periods=24 * pd.Period(f"{year}-{month:02d}").days_in_month,
            freq="h",
        )
        n = len(month_times)
        hourly = {
            "time": month_times.strftime("%Y-%m-%dT%H:%M").tolist(),
            "temperature_2m": [10.0] * n,
            "precipitation": [0.1] * n,
            "shortwave_radiation": [100.0] * n,
            "soil_moisture_0_to_7cm": [0.3] * n,
        }
        return hourly, 59.25, 18.0

    with patch("metafilter.meteorology.fetch_open_meteo_archive", side_effect=fake_fetch) as fetch:
        result = fetch_daily_meteorology(
            bbox_wgs84=bbox,
            date_start="2019-04-15",
            date_end="2019-06-15",
            cache_dir=tmp_path,
        )
        assert fetch.call_count == 3
        assert len(result.cache_paths) == 3
        assert result.frame["date"].iloc[0] == "2019-04-15"
        assert result.frame["date"].iloc[-1] == "2019-06-15"
        assert result.frame.loc[result.frame["date"] == "2019-05-15", "gdd_prev30d_c"].notna().all()

    with patch("metafilter.meteorology.fetch_open_meteo_archive") as fetch:
        fetch_daily_meteorology(
            bbox_wgs84=bbox,
            date_start="2019-04-15",
            date_end="2019-06-15",
            cache_dir=tmp_path,
        )
        fetch.assert_not_called()


def test_fetch_daily_meteorology_replaces_corrupt_cache(tmp_path):
    bbox = {"west": 18.0, "south": 59.2, "east": 18.2, "north": 59.4}
    # Resolve the actual cache name without making its hash part of the API.
    from metafilter.meteorology import _cache_name

    corrupt = tmp_path / _cache_name(bbox, 2019, 5)
    corrupt.write_bytes(b"not a netcdf file")

    def fake_fetch(year, month, **kwargs):
        times = pd.date_range("2019-05-01", periods=24 * 31, freq="h")
        hourly = {
            "time": times.strftime("%Y-%m-%dT%H:%M").tolist(),
            "temperature_2m": [10.0] * len(times),
            "precipitation": [0.1] * len(times),
            "shortwave_radiation": [100.0] * len(times),
            "soil_moisture_0_to_7cm": [0.3] * len(times),
        }
        return hourly, 59.25, 18.0

    with patch(
        "metafilter.meteorology.fetch_open_meteo_archive", side_effect=fake_fetch
    ) as fetch:
        fetch_daily_meteorology(
            bbox_wgs84=bbox,
            date_start="2019-05-01",
            date_end="2019-05-02",
            cache_dir=tmp_path,
        )
    fetch.assert_called_once()
    assert corrupt.stat().st_size > len(b"not a netcdf file")


def test_fetch_daily_meteorology_serializes_same_cache_entry(tmp_path):
    bbox = {"west": 18.0, "south": 59.2, "east": 18.2, "north": 59.4}

    def fake_fetch(year, month, **kwargs):
        times = pd.date_range("2019-05-01", periods=24 * 31, freq="h")
        hourly = {
            "time": times.strftime("%Y-%m-%dT%H:%M").tolist(),
            "temperature_2m": [10.0] * len(times),
            "precipitation": [0.1] * len(times),
            "shortwave_radiation": [100.0] * len(times),
            "soil_moisture_0_to_7cm": [0.3] * len(times),
        }
        return hourly, 59.25, 18.0

    def call():
        return fetch_daily_meteorology(
            bbox_wgs84=bbox,
            date_start="2019-05-01",
            date_end="2019-05-02",
            cache_dir=tmp_path,
        )

    with patch(
        "metafilter.meteorology.fetch_open_meteo_archive", side_effect=fake_fetch
    ) as fetch:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: call(), range(2)))
    assert fetch.call_count == 1
    assert all(not result.frame.empty for result in results)


def test_legacy_wrappers_forward_script_execution(monkeypatch):
    calls = []

    def fake_run_module(name, *, run_name):
        calls.append((name, run_name))

    monkeypatch.setattr(runpy, "run_module", fake_run_module)
    root = Path(__file__).parents[1]
    runpy.run_path(str(root / "scripts/process_era5.py"), run_name="__main__")
    runpy.run_path(str(root / "scripts/download_open_meteo.py"), run_name="__main__")
    assert calls == [
        ("metafilter.core", "__main__"),
        ("metafilter.open_meteo", "__main__"),
    ]
