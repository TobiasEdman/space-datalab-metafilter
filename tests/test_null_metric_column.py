"""A rule without a usable metric_column stays a configuration error.

`.get(key, default)` returns None when the key is present and null, so a JSON
null reached the lookback regex and surfaced as TypeError — escaping every
caller's MetafilterError handling, including the CLI's.
"""
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from metafilter.cli import process_era5_main
from metafilter.core import (
    MetafilterConfigurationError,
    MetafilterError,
    parse_lookback,
    process_era5_data,
)
from scripts.download_era5 import lookback_days_needed

AREA = {"west": 18.09, "south": 59.29, "east": 18.11, "north": 59.31}
NULL_PROFILE = {"rules": {"temperature": {"metric_column": None, "operator": "gt", "threshold": 0}}}


@pytest.fixture
def era5_file(tmp_path):
    times = pd.date_range("2024-08-01", periods=49, freq="h")
    xr.Dataset(
        {"t2m": (("time", "latitude", "longitude"), np.full((49, 1, 1), 293.15)),
         "tp": (("time", "latitude", "longitude"), np.zeros((49, 1, 1)))},
        coords={"time": times, "latitude": [59.3], "longitude": [18.1]},
    ).to_netcdf(tmp_path / "aug.nc", engine="scipy")
    return tmp_path / "aug.nc"


@pytest.mark.parametrize("column", [None, 123, ["precip_prev7d_mm"]])
def test_parse_lookback_ignores_non_strings(column):
    assert parse_lookback(column) is None


def test_api_raises_the_configuration_error(era5_file):
    with pytest.raises(MetafilterConfigurationError, match="missing 'metric_column'"):
        process_era5_data(era5_file, NULL_PROFILE, area=AREA)


def test_the_error_stays_within_metafilter_error(era5_file):
    """Library callers catch MetafilterError; a TypeError escapes that."""
    with pytest.raises(MetafilterError):
        process_era5_data(era5_file, NULL_PROFILE, area=AREA)


def test_cli_reports_it_concisely(tmp_path, era5_file, capsys):
    profile = tmp_path / "null.json"
    profile.write_text(json.dumps(NULL_PROFILE))

    with pytest.raises(SystemExit) as exc_info:
        process_era5_main([str(era5_file), "--filter", str(profile),
                           "--bbox", "18.09", "59.29", "18.11", "59.31"])

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "missing 'metric_column'" in captured.err
    assert "Traceback" not in captured.err


def test_buffer_sizing_tolerates_a_null_column(tmp_path):
    profile = tmp_path / "null.json"
    profile.write_text(json.dumps(NULL_PROFILE))
    assert lookback_days_needed(profile) == 0
