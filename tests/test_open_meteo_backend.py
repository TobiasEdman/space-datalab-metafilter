"""Tests for the Open-Meteo backend — JSON-to-xarray adapter, variable mapping,
backend selector dispatch, and end-to-end consumption via calculate_daily_metrics.

No network calls in the test suite. `requests.get` is patched on the calls
that would hit Open-Meteo's archive endpoint; the response is built from
fixture JSON with known properties.
"""
from __future__ import annotations

import json
import pathlib
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from scripts.download_open_meteo import (
    OPEN_METEO_DEFAULT_HOURLY,
    _snap_to_era5_grid,
    download_open_meteo_land,
    fetch_open_meteo_archive,
    open_meteo_json_to_dataset,
)
from scripts.download_era5 import backend_for_filter, download_period
from scripts.process_era5 import (
    apply_metafilter,
    calculate_daily_metrics,
    load_metafilter_parameters,
)


FILTERS_DIR = pathlib.Path(__file__).resolve().parent.parent / "filters"


# ── Helper: build a synthetic Open-Meteo hourly response ───────────────────

def _build_om_hourly(start="2024-08-01", n_days=10, *, with_clouds=True):
    """Return a dict shaped like Open-Meteo's `hourly` block."""
    times = pd.date_range(start=start, periods=24 * n_days, freq="h")
    n = len(times)

    payload = {
        "time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
        "temperature_2m":             [20.0] * n,                     # 20 °C constant
        "precipitation":              [0.0] * n,                      # dry
        "shortwave_radiation":        [_summer_radiation(t.hour) for t in times],
        "soil_temperature_0_to_7cm":  [18.0] * n,
        "soil_moisture_0_to_7cm":     [0.25] * n,
        "snow_depth":                 [0.0] * n,
        "dew_point_2m":               [10.0] * n,
    }
    if with_clouds:
        payload["cloud_cover"] = [15.0] * n     # 15% → 0.15 fraction
        payload["cloud_cover_low"] = [8.0] * n  # 8% → 0.08 fraction
    return payload


def _summer_radiation(hour):
    """W/m² that loosely tracks a summer day: peak around noon."""
    if hour < 6 or hour > 18:
        return 0.0
    return float(800.0 * np.sin(np.pi * (hour - 6) / 12))


# ── JSON → xarray conversion ──────────────────────────────────────────────

def test_open_meteo_to_dataset_preserves_time_dim():
    hourly = _build_om_hourly(start="2024-08-01", n_days=3)
    ds = open_meteo_json_to_dataset(hourly, lat=59.3, lon=18.1)
    assert ds.sizes["time"] == 72
    assert ds.sizes["latitude"] == 1
    assert ds.sizes["longitude"] == 1


def test_open_meteo_temperature_converted_to_kelvin():
    """Open-Meteo returns °C; downstream code expects Kelvin in t2m."""
    hourly = _build_om_hourly(start="2024-08-01", n_days=1)
    ds = open_meteo_json_to_dataset(hourly, lat=59.3, lon=18.1)
    # 20 °C → 293.15 K
    np.testing.assert_allclose(ds["t2m"].values, 293.15, atol=0.01)


def test_open_meteo_precipitation_converted_to_metres():
    """Open-Meteo returns mm/h; downstream expects m/h."""
    hourly = _build_om_hourly(start="2024-08-01", n_days=1)
    # Override precip to 2 mm/h
    hourly["precipitation"] = [2.0] * len(hourly["time"])
    ds = open_meteo_json_to_dataset(hourly, lat=59.3, lon=18.1)
    # 2 mm/h → 0.002 m/h
    np.testing.assert_allclose(ds["tp"].values, 0.002, atol=1e-6)


def test_open_meteo_cloud_fraction_normalised_to_0_1():
    """Open-Meteo returns 0–100 percent; downstream expects 0–1 fraction."""
    hourly = _build_om_hourly(start="2024-08-01", n_days=1, with_clouds=True)
    ds = open_meteo_json_to_dataset(hourly, lat=59.3, lon=18.1)
    np.testing.assert_allclose(ds["tcc"].values, 0.15, atol=0.01)
    np.testing.assert_allclose(ds["lcc"].values, 0.08, atol=0.01)


def test_open_meteo_shortwave_converted_to_joules_per_m2():
    """Open-Meteo returns W/m² instantaneous; downstream expects J/m² per hour."""
    hourly = _build_om_hourly(start="2024-08-01", n_days=1)
    # Override to constant 100 W/m² so the conversion is checkable
    hourly["shortwave_radiation"] = [100.0] * len(hourly["time"])
    ds = open_meteo_json_to_dataset(hourly, lat=59.3, lon=18.1)
    # 100 W/m² × 3600 s = 360 000 J/m²
    np.testing.assert_allclose(ds["ssrd"].values, 360_000.0, rtol=0.01)


def test_open_meteo_missing_variable_is_skipped():
    """If Open-Meteo doesn't return a requested variable, the field is omitted."""
    hourly = _build_om_hourly(start="2024-08-01", n_days=1, with_clouds=False)
    ds = open_meteo_json_to_dataset(hourly, lat=59.3, lon=18.1)
    assert "tcc" not in ds.data_vars
    assert "lcc" not in ds.data_vars
    # But other vars must still be present
    assert "t2m" in ds.data_vars
    assert "tp" in ds.data_vars


# ── Backend selector ──────────────────────────────────────────────────────

def test_backend_for_filter_defaults_to_cds():
    legacy = FILTERS_DIR / "metafilter.json"
    assert backend_for_filter(legacy) == "cds"


def test_backend_for_filter_extracts_open_meteo(tmp_path):
    fpath = tmp_path / "om.json"
    fpath.write_text(json.dumps({
        "sensor": "sentinel-2",
        "backend": "open-meteo",
        "rules": {"t": {"metric_column": "mean_temp_c", "operator": "gt", "threshold": 10.0}}
    }))
    assert backend_for_filter(fpath) == "open-meteo"


def test_shipped_openmeteo_profile_advertises_backend():
    profile = FILTERS_DIR / "sentinel2_openmeteo.json"
    assert backend_for_filter(profile) == "open-meteo"


def test_load_metafilter_parameters_passes_backend_through():
    normalized = load_metafilter_parameters(FILTERS_DIR / "sentinel2_openmeteo.json")
    assert normalized["backend"] == "open-meteo"


def test_load_metafilter_parameters_defaults_backend_to_cds_for_legacy():
    normalized = load_metafilter_parameters(FILTERS_DIR / "metafilter.json")
    assert normalized["backend"] == "cds"


# ── Grid snapping ─────────────────────────────────────────────────────────

def test_snap_to_era5_grid_collapses_neighbouring_bboxes():
    """Two AOIs within the same ERA5 cell snap to the same centroid."""
    area1 = {"west": 18.0, "east": 18.1, "south": 59.2, "north": 59.3}
    area2 = {"west": 18.05, "east": 18.15, "south": 59.22, "north": 59.32}
    assert _snap_to_era5_grid(area1) == _snap_to_era5_grid(area2)


# ── Network fetch is mocked ───────────────────────────────────────────────

def test_fetch_open_meteo_archive_calls_archive_endpoint():
    """fetch_open_meteo_archive must hit the right URL with the right params."""
    fake_hourly = _build_om_hourly(start="2024-08-01", n_days=2)
    fake_response = MagicMock()
    fake_response.json.return_value = {"hourly": fake_hourly}
    fake_response.raise_for_status = MagicMock()

    with patch("requests.get", return_value=fake_response) as mock_get:
        hourly, lat, lon = fetch_open_meteo_archive(
            year=2024,
            month=8,
            area={"west": 18.0, "east": 18.2, "south": 59.2, "north": 59.4},
        )

    mock_get.assert_called_once()
    call = mock_get.call_args
    assert "archive-api.open-meteo.com" in call.args[0]
    params = call.kwargs["params"]
    assert params["start_date"] == "2024-08-01"
    assert params["end_date"] == "2024-08-31"
    assert "temperature_2m" in params["hourly"]
    assert hourly == fake_hourly


def test_fetch_open_meteo_archive_raises_when_hourly_missing():
    """An unexpected response shape must raise rather than silently produce zeros."""
    fake_response = MagicMock()
    fake_response.json.return_value = {"error": "bad request"}
    fake_response.raise_for_status = MagicMock()

    with patch("requests.get", return_value=fake_response):
        with pytest.raises(RuntimeError, match="did not contain 'hourly' block"):
            fetch_open_meteo_archive(2024, 8)


# ── End-to-end: Open-Meteo NetCDF feeds calculate_daily_metrics ────────────

def test_open_meteo_dataset_feeds_calculate_daily_metrics(tmp_path):
    """The NetCDF written by download_open_meteo_land must be consumable by
    the same calculate_daily_metrics() that handles CDS NetCDFs."""
    hourly = _build_om_hourly(start="2024-08-01", n_days=15)
    fake_response = MagicMock()
    fake_response.json.return_value = {"hourly": hourly}
    fake_response.raise_for_status = MagicMock()

    # Patch OUTPUT_DIR so the downloader writes inside tmp_path
    with patch("scripts.download_open_meteo.OUTPUT_DIR", str(tmp_path)):
        with patch("requests.get", return_value=fake_response):
            path = download_open_meteo_land(
                year=2024, month=8,
                area={"west": 18.0, "east": 18.2, "south": 59.2, "north": 59.4},
            )

    # Now feed it to calculate_daily_metrics — should produce all the same columns
    # as the CDS path would, because the variable names are identical.
    area = {"west": 18.0, "east": 18.2, "south": 59.2, "north": 59.4}
    # Open-Meteo NetCDF has a single (lat, lon) cell — area subset must still
    # match it, so we use the snapped centroid as a 0-width box.
    snapped_lat, snapped_lon = _snap_to_era5_grid(area)
    snapped_area = {
        "west": snapped_lon - 0.01, "east": snapped_lon + 0.01,
        "south": snapped_lat - 0.01, "north": snapped_lat + 0.01,
    }
    df = calculate_daily_metrics(path, area=snapped_area)

    # Required columns should all appear
    expected = {
        "mean_temp_c", "min_temp_c", "max_temp_c", "freeze_flag",
        "total_precip_mm", "precip_prev24h_mm", "precip_prev48h_mm",
        "precip_prev7d_mm", "precip_prev30d_mm",
        "ssrd_mj_m2", "swvl1_mean", "swvl1_delta_prev2d",
        "snow_depth_mean_m", "tcc_mean_overpass", "lcc_mean_overpass",
    }
    assert expected.issubset(df.columns), f"Missing: {expected - set(df.columns)}"

    # And the values should match the fixture: temperature_2m=20°C, so mean_temp_c≈20
    assert df["mean_temp_c"].dropna().between(19.5, 20.5).all()
    # tcc was 15% → 0.15 fraction
    assert df["tcc_mean_overpass"].dropna().between(0.14, 0.16).all()


def test_small_aoi_off_cell_falls_back_to_nearest(tmp_path):
    """A tile-sized AOI whose bounds miss the snapped ERA5 cell must still
    resolve to that cell instead of failing — the fetch snapped it there."""
    from metafilter.core import calculate_daily_metrics as library_metrics

    hourly = _build_om_hourly(start="2024-08-01", n_days=5)
    ds = open_meteo_json_to_dataset(hourly, lat=56.75, lon=14.5)
    path = tmp_path / "era5.nc"
    ds.to_netcdf(path)
    # ~5 km AOI south-east of the cell centre; the cell lies outside its bounds
    # but well within one 0.25° grid step of the centroid.
    area = {"west": 14.55, "east": 14.63, "south": 56.66, "north": 56.71}
    df = library_metrics(str(path), area=area)
    assert len(df) > 0
    assert "mean_temp_c" in df.columns


def test_area_beyond_one_grid_step_still_fails(tmp_path):
    from metafilter.core import MetafilterSelectionError
    from metafilter.core import calculate_daily_metrics as library_metrics

    hourly = _build_om_hourly(start="2024-08-01", n_days=5)
    ds = open_meteo_json_to_dataset(hourly, lat=59.25, lon=18.0)
    path = tmp_path / "era5.nc"
    ds.to_netcdf(path)
    area = {"west": 14.0, "east": 14.1, "south": 56.6, "north": 56.7}
    with pytest.raises(MetafilterSelectionError):
        library_metrics(str(path), area=area)


# ── download_period dispatches to the right backend ───────────────────────

def test_download_period_dispatches_to_open_meteo(tmp_path):
    """When the filter says backend=open-meteo, download_period must call
    download_open_meteo_land (one call per month, no separate cloud retrieve)."""
    fpath = tmp_path / "om.json"
    fpath.write_text(json.dumps({
        "sensor": "sentinel-2",
        "backend": "open-meteo",
        "rules": {
            "t": {"metric_column": "mean_temp_c", "operator": "gt", "threshold": 10.0},
        }
    }))

    with patch("scripts.download_open_meteo.download_open_meteo_land") as mock_dl:
        mock_dl.side_effect = lambda y, m, area=None, variables=None: f"/tmp/om_{y}_{m}.nc"
        result = download_period(2024, 8, fpath)

    # No buffer needed (no long-lookback rule), so just one land path
    assert len(result["land"]) == 1
    assert result["cloud"] == []
    mock_dl.assert_called_once()


def test_download_period_dispatches_to_open_meteo_with_buffer(tmp_path):
    """Open-Meteo path also prepends the previous month when long lookback is active."""
    fpath = tmp_path / "om_long.json"
    fpath.write_text(json.dumps({
        "sensor": "sentinel-2",
        "backend": "open-meteo",
        "rules": {
            "precip_30d": {
                "metric_column": "precip_prev30d_mm",
                "operator": "between",
                "threshold": [30.0, 150.0],
            },
        }
    }))

    with patch("scripts.download_open_meteo.download_open_meteo_land") as mock_dl:
        mock_dl.side_effect = lambda y, m, area=None, variables=None: f"/tmp/om_{y}_{m}.nc"
        result = download_period(2024, 8, fpath)

    assert len(result["land"]) == 2  # buffer month + primary
    assert result["cloud"] == []
    assert mock_dl.call_count == 2


def test_download_period_rejects_unknown_backend(tmp_path):
    fpath = tmp_path / "bad.json"
    fpath.write_text(json.dumps({
        "sensor": "sentinel-2",
        "backend": "made-up-source",
        "rules": {
            "t": {"metric_column": "mean_temp_c", "operator": "gt", "threshold": 10.0},
        }
    }))
    with pytest.raises(ValueError, match="Unknown backend"):
        download_period(2024, 8, fpath)


# ── Integration: end-to-end filter against an Open-Meteo NetCDF ───────────

def test_openmeteo_profile_end_to_end_pass(tmp_path):
    """Run the shipped sentinel2_openmeteo.json against a synthetic Open-Meteo
    NetCDF that satisfies all its rules. Confirms backend selector + variable
    mapping + filter engine compose correctly."""
    # Build 45 days of data with: warm, dry tail, modest cloud, normal precip 30d
    hourly = _build_om_hourly(start="2024-07-15", n_days=45)
    # Add some precipitation early in July so the 30d window has 30-150 mm
    # but the immediate days are dry
    early_days = 24 * 10  # first 10 days wet
    hourly["precipitation"][:early_days] = [3.0] * early_days  # ~720 mm if all kept

    # Trim to a more realistic precip footprint: 2 mm/day for first 15 days, dry after
    for i, t in enumerate(hourly["time"]):
        ts = pd.Timestamp(t)
        days_since_start = (ts - pd.Timestamp("2024-07-15")).days
        if days_since_start < 15:
            hourly["precipitation"][i] = 2.0 / 24  # 2 mm/day, in mm/h
        else:
            hourly["precipitation"][i] = 0.0

    fake_response = MagicMock()
    fake_response.json.return_value = {"hourly": hourly}
    fake_response.raise_for_status = MagicMock()

    with patch("scripts.download_open_meteo.OUTPUT_DIR", str(tmp_path)):
        with patch("requests.get", return_value=fake_response):
            path = download_open_meteo_land(
                year=2024, month=7,
                area={"west": 18.0, "east": 18.2, "south": 59.2, "north": 59.4},
            )

    snapped_lat, snapped_lon = _snap_to_era5_grid(
        {"west": 18.0, "east": 18.2, "south": 59.2, "north": 59.4}
    )
    snapped_area = {
        "west": snapped_lon - 0.01, "east": snapped_lon + 0.01,
        "south": snapped_lat - 0.01, "north": snapped_lat + 0.01,
    }

    profile = load_metafilter_parameters(FILTERS_DIR / "sentinel2_openmeteo.json")
    df = calculate_daily_metrics(
        path,
        area=snapped_area,
        sensor=profile["sensor"],
        overpass_time_utc=profile["overpass_time_utc"],
    )
    filtered, _ = apply_metafilter(df, profile)

    selected = filtered.loc[filtered["selected"], "date"].tolist()
    # Days late August (after dry spell + ramp-up of GDD) should pass
    assert any(d.startswith("2024-08") for d in selected), (
        f"Expected some August days to pass; got: {selected}"
    )


def test_openmeteo_profile_excludes_cloudy_days(tmp_path):
    """High-cloud fixture should produce zero selected days under the
    sentinel2_openmeteo.json profile."""
    hourly = _build_om_hourly(start="2024-07-15", n_days=45)
    # Crank cloud cover up to 80%
    hourly["cloud_cover"] = [80.0] * len(hourly["time"])
    hourly["cloud_cover_low"] = [70.0] * len(hourly["time"])

    fake_response = MagicMock()
    fake_response.json.return_value = {"hourly": hourly}
    fake_response.raise_for_status = MagicMock()

    with patch("scripts.download_open_meteo.OUTPUT_DIR", str(tmp_path)):
        with patch("requests.get", return_value=fake_response):
            path = download_open_meteo_land(
                year=2024, month=7,
                area={"west": 18.0, "east": 18.2, "south": 59.2, "north": 59.4},
            )

    snapped_lat, snapped_lon = _snap_to_era5_grid(
        {"west": 18.0, "east": 18.2, "south": 59.2, "north": 59.4}
    )
    snapped_area = {
        "west": snapped_lon - 0.01, "east": snapped_lon + 0.01,
        "south": snapped_lat - 0.01, "north": snapped_lat + 0.01,
    }

    profile = load_metafilter_parameters(FILTERS_DIR / "sentinel2_openmeteo.json")
    df = calculate_daily_metrics(
        path,
        area=snapped_area,
        sensor=profile["sensor"],
        overpass_time_utc=profile["overpass_time_utc"],
    )
    filtered, _ = apply_metafilter(df, profile)

    selected = filtered.loc[filtered["selected"], "date"].tolist()
    assert selected == [], f"Expected no days to pass with 80% cloud, got {selected}"
