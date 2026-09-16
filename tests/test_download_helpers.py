"""Tests for download_era5 helpers — buffer-month detection + cloud-need detection.

The actual cdsapi-retrieve calls aren't exercised (they'd hit ECMWF), but the
profile-inspection helpers are pure-Python and must be correct because they
gate whether the network calls happen at all.
"""
import json

import pytest

from scripts.download_era5 import (
    cloud_vars_needed,
    long_lookback_needed,
    _previous_month,
)


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def test_legacy_filter_does_not_need_cloud(tmp_path):
    payload = {
        "temperature": {"metric_column": "mean_temp_c", "operator": "gt", "threshold": 15.0},
        "precipitation": {"metric_column": "total_precip_mm", "operator": "lt", "threshold": 1.0},
    }
    path = _write(tmp_path, "legacy.json", payload)
    assert cloud_vars_needed(path) is False
    assert long_lookback_needed(path) is False


def test_s2_extended_needs_cloud_and_buffer(tmp_path):
    """The shipped sentinel2_extended.json uses tcc + precip_prev30d, so both flags trip."""
    import pathlib
    fpath = pathlib.Path(__file__).resolve().parent.parent / "filters" / "sentinel2_extended.json"
    assert cloud_vars_needed(fpath) is True
    assert long_lookback_needed(fpath) is True


def test_s1_default_does_not_need_cloud_or_buffer(tmp_path):
    """S1 default uses only short-window + pass-time columns."""
    import pathlib
    fpath = pathlib.Path(__file__).resolve().parent.parent / "filters" / "sentinel1_default.json"
    assert cloud_vars_needed(fpath) is False
    assert long_lookback_needed(fpath) is False


def test_long_lookback_triggered_by_any_long_prefix(tmp_path):
    for col in ["precip_prev7d_mm", "precip_prev30d_mm", "ssrd_prev30d_mj_m2",
                "gdd_prev30d_c", "swvl1_prev30d_mean", "dry_streak_days"]:
        payload = {
            "rules": {
                "x": {"metric_column": col, "operator": "gt", "threshold": 0.0}
            }
        }
        path = _write(tmp_path, f"prof_{col}.json", payload)
        assert long_lookback_needed(path), f"{col} did not trigger long_lookback_needed"


def test_previous_month_normal():
    assert _previous_month(2024, 8) == (2024, 7)


def test_previous_month_january_wraps_year():
    assert _previous_month(2024, 1) == (2023, 12)


# ── ERA5-Land variables follow the profile's rules ─────────────────────────

def test_era5_land_variables_for_s1_profile():
    from scripts.download_era5 import era5_land_variables_for_filter
    from tests.conftest import REPO_ROOT

    variables = era5_land_variables_for_filter(REPO_ROOT / "filters" / "sentinel1_default.json")
    assert variables == [
        "total_precipitation",
        "skin_temperature",
        "volumetric_soil_water_layer_1",
        "snow_depth",
    ]


def test_era5_land_variables_for_extended_s2_profile_skip_cloud_columns():
    from scripts.download_era5 import era5_land_variables_for_filter
    from tests.conftest import REPO_ROOT

    variables = era5_land_variables_for_filter(REPO_ROOT / "filters" / "sentinel2_extended.json")
    assert variables == [
        "2m_temperature",
        "total_precipitation",
        "surface_solar_radiation_downwards",
        "volumetric_soil_water_layer_1",
    ]


def test_era5_land_variables_for_legacy_profile_use_rule_defaults():
    from scripts.download_era5 import era5_land_variables_for_filter
    from tests.conftest import REPO_ROOT

    variables = era5_land_variables_for_filter(REPO_ROOT / "filters" / "metafilter.json")
    assert variables == ["2m_temperature", "total_precipitation"]


def test_unknown_metric_column_is_rejected(tmp_path):
    import json
    import pytest
    from scripts.download_era5 import era5_land_variables_for_filter

    profile = tmp_path / "odd.json"
    profile.write_text(json.dumps({"rules": {"x": {"metric_column": "humidity_mean", "operator": "gt", "threshold": 1}}}))
    with pytest.raises(ValueError, match="humidity_mean"):
        era5_land_variables_for_filter(profile)
