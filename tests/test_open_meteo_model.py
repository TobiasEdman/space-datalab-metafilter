"""The Open-Meteo backend pins one reanalysis model and caches are bound to it."""
import calendar

import numpy as np
import pandas as pd
import xarray as xr

from metafilter.core import ACCUMULATION_ATTR, ACCUMULATION_HOURLY
from metafilter.meteorology import _cache_name, _valid_cache
from metafilter.open_meteo import OPEN_METEO_MODEL, open_meteo_json_to_dataset

BBOX = {"west": 18.0, "south": 59.2, "east": 18.2, "north": 59.4}


def _month_dataset(year=2024, month=8, **attrs):
    hours = calendar.monthrange(year, month)[1] * 24
    times = pd.date_range(f"{year}-{month:02d}-01", periods=hours, freq="h")
    shape = (hours, 1, 1)
    data = {
        name: (("time", "latitude", "longitude"), np.full(shape, value, dtype=np.float32))
        for name, value in {"t2m": 293.15, "tp": 0.0, "ssrd": 1e5, "swvl1": 0.25}.items()
    }
    return xr.Dataset(data, coords={"time": times, "latitude": [59.25], "longitude": [18.0]}, attrs=attrs)


def test_cache_name_is_keyed_by_model():
    assert OPEN_METEO_MODEL in _cache_name(BBOX, 2024, 8)


def test_cache_without_model_identity_is_not_reused(tmp_path):
    path = tmp_path / _cache_name(BBOX, 2024, 8)
    _month_dataset().to_netcdf(path, engine="scipy")
    assert _valid_cache(path, 2024, 8) is False


def test_cache_with_model_identity_is_reused(tmp_path):
    path = tmp_path / _cache_name(BBOX, 2024, 8)
    _month_dataset(
        metafilter_source_model=OPEN_METEO_MODEL, **{ACCUMULATION_ATTR: ACCUMULATION_HOURLY}
    ).to_netcdf(path, engine="scipy")
    assert _valid_cache(path, 2024, 8) is True


def test_adapter_dataset_carries_model_identity():
    hourly = {"time": ["2024-08-01T00:00", "2024-08-01T01:00"], "temperature_2m": [20.0, 21.0]}
    dataset = open_meteo_json_to_dataset(hourly, lat=59.25, lon=18.0)
    assert dataset.attrs["metafilter_source_model"] == OPEN_METEO_MODEL
    assert dataset.attrs[ACCUMULATION_ATTR] == ACCUMULATION_HOURLY
