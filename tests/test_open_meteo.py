"""Tests for `scripts/download_open_meteo` — the Open-Meteo ERA5 backend.

No live network calls. `requests.get` is patched on the one test that
exercises the HTTP layer.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from scripts.download_open_meteo import (
    OPEN_METEO_DEFAULT_HOURLY,
    _snap_to_era5_grid,
    fetch_open_meteo_archive,
    open_meteo_json_to_dataset,
)


def _build_om_hourly(start="2024-08-01", n_days=3, with_clouds=True):
    times = pd.date_range(start=start, periods=24 * n_days, freq="h")
    n = len(times)
    payload = {
        "time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
        "temperature_2m":            [20.0] * n,
        "precipitation":             [0.0] * n,
        "shortwave_radiation":       [0.0] * n,
        "soil_temperature_0_to_7cm": [18.0] * n,
        "soil_moisture_0_to_7cm":    [0.25] * n,
        "snow_depth":                [0.0] * n,
        "dew_point_2m":              [10.0] * n,
    }
    if with_clouds:
        payload["cloud_cover"] = [15.0] * n      # 15% → 0.15 fraction
        payload["cloud_cover_low"] = [8.0] * n   # 8% → 0.08 fraction
    return payload


def test_temperature_converted_to_kelvin():
    ds = open_meteo_json_to_dataset(_build_om_hourly(n_days=1), lat=59.3, lon=18.1)
    # 20 °C → 293.15 K
    np.testing.assert_allclose(ds["t2m"].values, 293.15, atol=0.01)


def test_precipitation_converted_to_metres():
    hourly = _build_om_hourly(n_days=1)
    hourly["precipitation"] = [2.0] * len(hourly["time"])  # 2 mm/h
    ds = open_meteo_json_to_dataset(hourly, lat=59.3, lon=18.1)
    np.testing.assert_allclose(ds["tp"].values, 0.002, atol=1e-6)


def test_cloud_cover_normalised_to_fraction():
    ds = open_meteo_json_to_dataset(
        _build_om_hourly(n_days=1, with_clouds=True), lat=59.3, lon=18.1
    )
    np.testing.assert_allclose(ds["tcc"].values, 0.15, atol=0.01)
    np.testing.assert_allclose(ds["lcc"].values, 0.08, atol=0.01)


def test_shortwave_w_per_m2_to_j_per_m2_per_hour():
    hourly = _build_om_hourly(n_days=1)
    hourly["shortwave_radiation"] = [100.0] * len(hourly["time"])
    ds = open_meteo_json_to_dataset(hourly, lat=59.3, lon=18.1)
    # 100 W/m² × 3600 s = 360 000 J/m²
    np.testing.assert_allclose(ds["ssrd"].values, 360_000.0, rtol=0.01)


def test_missing_variable_is_silently_skipped():
    ds = open_meteo_json_to_dataset(
        _build_om_hourly(n_days=1, with_clouds=False), lat=59.3, lon=18.1
    )
    assert "tcc" not in ds.data_vars
    assert "t2m" in ds.data_vars


def test_snap_to_era5_grid_collapses_neighbours():
    a = {"west": 18.0, "east": 18.1, "south": 59.2, "north": 59.3}
    b = {"west": 18.05, "east": 18.15, "south": 59.22, "north": 59.32}
    assert _snap_to_era5_grid(a) == _snap_to_era5_grid(b)


def test_fetch_open_meteo_archive_hits_correct_endpoint():
    fake_response = MagicMock()
    fake_response.json.return_value = {"hourly": _build_om_hourly(n_days=2)}
    fake_response.raise_for_status = MagicMock()

    with patch("requests.get", return_value=fake_response) as mock_get:
        hourly, lat, lon = fetch_open_meteo_archive(
            year=2024, month=8,
            area={"west": 18.0, "east": 18.2, "south": 59.2, "north": 59.4},
        )

    mock_get.assert_called_once()
    call = mock_get.call_args
    assert "archive-api.open-meteo.com" in call.args[0]
    params = call.kwargs["params"]
    assert params["start_date"] == "2024-08-01"
    assert params["end_date"] == "2024-08-31"
    assert "temperature_2m" in params["hourly"]
    assert "hourly" in fake_response.json.return_value
    assert "temperature_2m" in hourly


def test_fetch_raises_on_unexpected_response_shape():
    fake_response = MagicMock()
    fake_response.json.return_value = {"error": "bad request"}
    fake_response.raise_for_status = MagicMock()

    with patch("requests.get", return_value=fake_response):
        with pytest.raises(RuntimeError, match="missing 'hourly' block"):
            fetch_open_meteo_archive(2024, 8)


def test_default_hourly_variable_list_is_complete():
    """Lock the default Open-Meteo variable list against accidental shrinkage."""
    expected = {
        "temperature_2m", "precipitation", "shortwave_radiation",
        "cloud_cover", "cloud_cover_low",
        "soil_temperature_0_to_7cm", "soil_moisture_0_to_7cm",
        "snow_depth", "dew_point_2m",
    }
    assert expected.issubset(set(OPEN_METEO_DEFAULT_HOURLY))
