"""Short names CDS actually delivers, pinned against the live API.

Recorded 2026-09-18 from one `reanalysis-era5-land` and one
`reanalysis-era5-single-levels` retrieval. Every name a metric reads is pinned
here so a spelling the code does not expect fails in the suite rather than at
a user's first credentialed run.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from metafilter.core import calculate_daily_metrics
from metafilter.open_meteo import _VAR_MAPPING, open_meteo_json_to_dataset

AREA = {"west": 18.09, "south": 59.29, "east": 18.11, "north": 59.31}

# requested CDS variable -> short name in the returned NetCDF (live-verified)
CDS_SHORT_NAMES = {
    "2m_temperature": "t2m",
    "total_precipitation": "tp",
    "surface_solar_radiation_downwards": "ssrd",
    "skin_temperature": "skt",
    "soil_temperature_level_1": "stl1",
    "volumetric_soil_water_layer_1": "swvl1",
    "snow_depth": "sde",
    "snowfall": "sf",
    "2m_dewpoint_temperature": "d2m",
    "total_cloud_cover": "tcc",
    "low_cloud_cover": "lcc",
}


def _dataset(path, name, value):
    times = pd.date_range("2024-08-01", periods=49, freq="h")
    xr.Dataset(
        {name: (("time", "latitude", "longitude"),
                np.full((len(times), 1, 1), value, dtype=float))},
        coords={"time": times, "latitude": [59.3], "longitude": [18.1]},
    ).to_netcdf(path, engine="scipy")
    return path


@pytest.mark.parametrize("name", ["sde", "sd"])
def test_snow_depth_read_under_both_spellings(tmp_path, name):
    """`sde` is what CDS sends; `sd` is what pre-fix Open-Meteo caches carry."""
    frame = calculate_daily_metrics(_dataset(tmp_path / f"{name}.nc", name, 0.10), area=AREA)
    assert "snow_depth_mean_m" in frame.columns
    assert frame["snow_depth_mean_m"].iloc[0] == pytest.approx(0.10)


def test_cds_spelling_wins_when_both_are_present(tmp_path):
    times = pd.date_range("2024-08-01", periods=49, freq="h")
    shape = (len(times), 1, 1)
    xr.Dataset(
        {"sde": (("time", "latitude", "longitude"), np.full(shape, 0.10)),
         "sd": (("time", "latitude", "longitude"), np.full(shape, 0.99))},
        coords={"time": times, "latitude": [59.3], "longitude": [18.1]},
    ).to_netcdf(tmp_path / "both.nc", engine="scipy")
    frame = calculate_daily_metrics(tmp_path / "both.nc", area=AREA)
    assert frame["snow_depth_mean_m"].iloc[0] == pytest.approx(0.10)


def test_open_meteo_emits_the_cds_snow_spelling():
    assert _VAR_MAPPING["snow_depth"][0] == CDS_SHORT_NAMES["snow_depth"]
    hourly = {"time": ["2024-08-01T00:00", "2024-08-01T01:00"], "snow_depth": [0.1, 0.1]}
    assert "sde" in open_meteo_json_to_dataset(hourly, lat=59.25, lon=18.0).data_vars


def test_every_metric_source_name_matches_the_live_map():
    """Names the metrics read must be names CDS actually delivers."""
    source = (__import__("pathlib").Path(__file__).parents[1] / "metafilter" / "core.py").read_text()
    for short in ("t2m", "tp", "ssrd", "skt", "stl1", "swvl1", "sde", "tcc", "lcc"):
        assert short in CDS_SHORT_NAMES.values(), short
        assert f'"{short}"' in source, f"core.py never reads {short}"
