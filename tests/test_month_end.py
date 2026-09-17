"""Daily coverage regressions: real aggregation and I/O, offline providers."""
import calendar
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from metafilter import fetch_daily_meteorology
from metafilter.core import ACCUMULATION_ATTR, apply_metafilter, calculate_daily_metrics
from metafilter.meteorology import _cache_name
from metafilter.open_meteo import open_meteo_json_to_dataset
from scripts.download_era5 import download_era5_land

AREA = {"west": 18.09, "south": 59.29, "east": 18.11, "north": 59.31}
DRY_RULE = {"precipitation": {"threshold": 1.0}}


def _write(path, times, variables, **attrs):
    dataset = xr.Dataset(
        {name: (("time", "latitude", "longitude"), np.asarray(values).reshape(-1, 1, 1))
         for name, values in variables.items()},
        coords={"time": times, "latitude": [59.3], "longitude": [18.1]},
        attrs=attrs,
    )
    dataset.to_netcdf(path, engine="scipy")
    return path


@pytest.mark.parametrize("convention", ["hourly", "forecast_start"])
@pytest.mark.parametrize("gap", ["none", "last_hour", "interior_hour", "null", "duplicate"])
def test_daily_totals_require_complete_coverage(tmp_path, convention, gap):
    times = pd.date_range("2024-08-31", "2024-09-01", freq="h")
    rain = np.zeros(len(times))
    rain[-1] = 0.010  # 10 mm in the hour ending at September 1 00:00
    radiation = (np.where(times.hour == 0, 24, times.hour)
                 if convention == "forecast_start" else np.ones(len(times))) * 1e6
    values = {"tp": rain, "ssrd": radiation.astype(float)}
    if gap in ("last_hour", "interior_hour", "duplicate"):
        remove = -1 if gap == "last_hour" else 12
        times = times.delete(remove)
        values = {name: np.delete(data, remove) for name, data in values.items()}
        if gap == "duplicate":
            times = times.insert(12, times[11])
            values = {name: np.insert(data, 12, data[11]) for name, data in values.items()}
    elif gap == "null":
        for data in values.values():
            data[12] = np.nan
    path = _write(tmp_path / "land.nc", times, values, **{ACCUMULATION_ATTR: convention})
    frame = calculate_daily_metrics(path, area=AREA)
    assert frame.date.tolist() == ["2024-08-31"]  # no synthetic September day
    if gap == "none":
        assert frame.total_precip_mm.item() == pytest.approx(10)
        assert frame.ssrd_mj_m2.item() == pytest.approx(24)
    else:
        assert frame.total_precip_mm.isna().all()
        assert frame.ssrd_mj_m2.isna().all()
    selected, _ = apply_metafilter(frame, DRY_RULE)
    assert not selected.selected.any()


def test_missing_precipitation_propagates_into_lookbacks(tmp_path):
    times = pd.date_range("2024-08-01", "2024-08-06", freq="h")
    rain = np.zeros(len(times))
    rain[times == pd.Timestamp("2024-08-02T12:00")] = np.nan
    rain[times == pd.Timestamp("2024-08-04T12:00")] = 0.010
    path = _write(tmp_path / "gapped.nc", times, {"tp": rain}, **{ACCUMULATION_ATTR: "hourly"})
    frame = calculate_daily_metrics(path, area=AREA).set_index("date")
    assert pd.isna(frame.loc["2024-08-02", "total_precip_mm"])
    assert pd.isna(frame.loc["2024-08-03", "precip_prev24h_mm"])
    assert frame.loc[["2024-08-03", "2024-08-04"], "precip_prev48h_mm"].isna().all()
    assert frame.loc[["2024-08-03", "2024-08-04"], "dry_streak_days"].isna().all()
    assert frame.loc["2024-08-05", "dry_streak_days"] == 0
    assert frame.loc["2024-08-05", "precip_prev48h_mm"] == pytest.approx(10)
    selected, _ = apply_metafilter(frame.loc[["2024-08-02"]], DRY_RULE)
    assert not selected.selected.any()


@pytest.mark.parametrize("gap", ["none", "hour", "day", "boundary_only", "permanent_mask"])
def test_spatial_gaps_cannot_remove_the_wet_cell(tmp_path, gap):
    times = pd.date_range("2024-08-30", "2024-09-01", freq="h")
    values = np.zeros((len(times), 1, 2))
    values[:, 0, 1] = 1.0  # one dry cell and one wet cell
    if gap == "hour":
        values[times == pd.Timestamp("2024-08-31T12:00"), 0, 1] = np.nan
    elif gap == "day":
        values[times > pd.Timestamp("2024-08-31"), 0, 1] = np.nan
    elif gap == "permanent_mask":
        values[:, 0, 1] = np.nan
    elif gap == "boundary_only":
        values[:, 0, 1] = np.nan
        values[-1, 0, 1] = 10  # cumulative reading, missing its predecessor
    dataset = xr.Dataset(
        {"tp": (("time", "latitude", "longitude"), values / 1000),
         "ssrd": (("time", "latitude", "longitude"), values * 1e6)},
        coords={"time": times, "latitude": [59.3], "longitude": [18.095, 18.105]},
        attrs={ACCUMULATION_ATTR: "forecast_start" if gap == "boundary_only" else "hourly"},
    )
    path = tmp_path / "spatial.nc"
    dataset.to_netcdf(path, engine="scipy")
    frame = calculate_daily_metrics(path, area=AREA)
    last = frame.iloc[[-1]]
    if gap in ("hour", "day", "boundary_only"):
        assert last.total_precip_mm.isna().all()
        assert last.ssrd_mj_m2.isna().all()
    else:
        expected = 12 if gap == "none" else 0
        assert last.total_precip_mm.item() == pytest.approx(expected)
        assert last.ssrd_mj_m2.item() == pytest.approx(expected)
    selected, _ = apply_metafilter(last, DRY_RULE)
    assert selected.selected.item() == (gap == "permanent_mask")


@pytest.mark.parametrize("month", ["2024-08", "2024-02", "2024-12"])
def test_public_api_fetches_boundary_and_replaces_old_cache(tmp_path, month):
    start = pd.Timestamp(month + "-01")
    boundary = start + pd.offsets.MonthBegin(1)
    last_day = (boundary - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    calls = []

    def hourly(times):
        return {
            "time": times.strftime("%Y-%m-%dT%H:%M").tolist(),
            "precipitation": np.where(times == boundary, 10., 0.).tolist(),
            "temperature_2m": [20.] * len(times),
            "shortwave_radiation": [1e6 / 3600] * len(times),
            "soil_moisture_0_to_7cm": [0.25] * len(times),
            "cloud_cover": [10.] * len(times),
        }

    # This cache has the right model but lacks the required boundary sample.
    old_times = pd.date_range(start, boundary, freq="h", inclusive="left")
    old = open_meteo_json_to_dataset(hourly(old_times), 59.3, 18.1)
    cache_path = tmp_path / _cache_name(AREA, start.year, start.month)
    old.to_netcdf(cache_path, engine="scipy")

    def get(url, *, params, timeout):
        calls.append(params)
        assert params["start_date"] == start.strftime("%Y-%m-%d")
        assert params["end_date"] == boundary.strftime("%Y-%m-%d")
        times = pd.date_range(start, boundary + pd.Timedelta(days=1), freq="h", inclusive="left")
        return Mock(json=lambda: {"hourly": hourly(times)})

    with patch("requests.get", side_effect=get):
        result = fetch_daily_meteorology(
            bbox_wgs84=AREA, date_start=last_day, date_end=last_day, cache_dir=tmp_path,
        )
    assert len(calls) == 1
    assert result.frame.date.tolist() == [last_day]
    assert result.frame.total_precip_mm.item() == pytest.approx(10)
    assert result.frame.ssrd_mj_m2.item() == pytest.approx(24)
    assert result.frame.tcc_mean_overpass.item() == pytest.approx(0.1)
    selected, _ = apply_metafilter(result.frame, DRY_RULE)
    assert not selected.selected.any()
    with xr.open_dataset(cache_path, engine="scipy") as cached:
        assert cached.sizes["time"] == len(old_times) + 1
        assert pd.Timestamp(cached.time.values[-1]) == boundary
    with patch("requests.get") as get:
        again = fetch_daily_meteorology(
            bbox_wgs84=AREA, date_start=last_day, date_end=last_day, cache_dir=tmp_path,
        )
    get.assert_not_called()
    pd.testing.assert_frame_equal(result.frame, again.frame)


@pytest.mark.parametrize("year,month", [(2024, 8), (2024, 2), (2024, 12)])
def test_cds_download_requests_only_one_extra_hour(tmp_path, monkeypatch, year, month):
    boundary = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthBegin(1)
    calls = []

    def retrieve(dataset, request, output):
        calls.append(request)
        times = pd.DatetimeIndex([
            f"{request['year']}-{request['month']}-{day}T{hour}"
            for day in request["day"] for hour in request["time"]
        ])
        _write(output, times, {
            "t2m": np.full(len(times), 293.15),
            "tp": np.where(times == boundary, 0.010, 0.),
        })

    monkeypatch.setitem(sys.modules, "cdsapi", SimpleNamespace(Client=lambda: SimpleNamespace(retrieve=retrieve)))
    with patch("scripts.download_era5.OUTPUT_DIR", str(tmp_path)):
        path = download_era5_land(year, month, area=AREA)
    assert len(calls) == 2
    assert calls[1]["year"] == str(boundary.year)
    assert calls[1]["month"] == f"{boundary.month:02d}"
    assert calls[1]["day"] == ["01"]
    assert calls[1]["time"] == ["00:00"]
    assert calls[1]["variable"] == calls[0]["variable"]
    frame = calculate_daily_metrics(path, area=AREA)
    assert len(frame) == calendar.monthrange(year, month)[1]
    assert frame.total_precip_mm.iloc[-1] == pytest.approx(10)
    assert frame.mean_temp_c.iloc[-1] == pytest.approx(20)
    selected, _ = apply_metafilter(frame.iloc[[-1]], DRY_RULE)
    assert not selected.selected.any()


def test_cds_join_preserves_independently_packed_values(tmp_path, monkeypatch):
    def retrieve(dataset, request, output):
        boundary = request["month"] == "09"
        times = (pd.date_range("2024-09-01", periods=1, freq="h") if boundary
                 else pd.date_range("2024-08-01", periods=31 * 24, freq="h"))
        # The new boundary value does not fit the monthly file's int16 scale.
        data = xr.Dataset(
            {"tp": (("valid_time", "latitude", "longitude"),
                    np.full((len(times), 1, 1), 0.010 if boundary else 0.0))},
            coords={"valid_time": times, "latitude": [59.3], "longitude": [18.1]},
        )
        data.to_netcdf(output, engine="h5netcdf", encoding={
            "tp": {"dtype": "int16", "scale_factor": 1e-5 if boundary else 1e-7,
                   "_FillValue": -32768, "zlib": True},
        })

    monkeypatch.setitem(sys.modules, "cdsapi", SimpleNamespace(Client=lambda: SimpleNamespace(retrieve=retrieve)))
    with patch("scripts.download_era5.OUTPUT_DIR", str(tmp_path)):
        path = download_era5_land(2024, 8, area=AREA, variables=["total_precipitation"])
    with xr.open_dataset(path, engine="scipy") as result:
        assert result.tp.isel(time=-1).item() == pytest.approx(0.010)
    frame = calculate_daily_metrics(path, area=AREA)
    assert frame.total_precip_mm.iloc[-1] == pytest.approx(10)


@pytest.mark.parametrize("failure", ["missing_sample", "request_error"])
def test_cds_boundary_failure_preserves_existing_output(tmp_path, monkeypatch, failure):
    destination = tmp_path / "era5" / "era5_land_2024_08.nc"
    destination.parent.mkdir()
    destination.write_bytes(b"existing output")

    def retrieve(dataset, request, output):
        if request["month"] == "09" and failure == "request_error":
            raise RuntimeError("provider unavailable")
        # Missing boundary response deliberately ends an hour too soon.
        times = pd.date_range("2024-08-01", periods=31 * 24, freq="h")
        _write(output, times, {"tp": np.zeros(len(times))})

    monkeypatch.setitem(sys.modules, "cdsapi", SimpleNamespace(Client=lambda: SimpleNamespace(retrieve=retrieve)))
    with patch("scripts.download_era5.OUTPUT_DIR", str(tmp_path)):
        with pytest.raises((ValueError, RuntimeError), match="boundary|provider unavailable"):
            download_era5_land(2024, 8, area=AREA)
    assert destination.read_bytes() == b"existing output"
    assert list(destination.parent.iterdir()) == [destination]
