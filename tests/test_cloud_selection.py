"""Cloud-cover files follow the same AOI rules as the land file."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from metafilter.core import MetafilterSelectionError, _open_and_concat, calculate_daily_metrics

AREA = {"west": 18.09, "south": 59.29, "east": 18.11, "north": 59.31}


def _write(path, lat, lon, **variables):
    times = pd.date_range("2024-08-01", periods=48, freq="h")
    data = {
        name: (("time", "latitude", "longitude"), np.full((48, 1, 1), value, dtype=float))
        for name, value in variables.items()
    }
    xr.Dataset(data, coords={"time": times, "latitude": [lat], "longitude": [lon]}).to_netcdf(
        path, engine="scipy"
    )
    return path


def test_cloud_cell_beyond_one_grid_step_is_a_real_mismatch(tmp_path):
    land = _write(tmp_path / "land.nc", 59.3, 18.1, t2m=293.15, tp=0.0)
    cloud = _write(tmp_path / "cloud.nc", 58.5, 17.0, tcc=0.1)  # ~0.8 deg away
    with pytest.raises(MetafilterSelectionError, match="cloud dataset"):
        calculate_daily_metrics(land, area=AREA, cloud_file_path=cloud)


def test_empty_path_list_is_rejected_explicitly():
    with pytest.raises(ValueError, match="at least one NetCDF path"):
        _open_and_concat([])
