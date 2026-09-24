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


def test_s1_default_needs_no_cloud_and_one_day_of_history(tmp_path):
    """S1 default needs no cloud file, but its 24h window reaches back a day.

    Without that day the first of every month has no precip_prev24h_mm and the
    dry-canopy rule can never select it.
    """
    import pathlib
    from scripts.download_era5 import lookback_days_needed
    fpath = pathlib.Path(__file__).resolve().parent.parent / "filters" / "sentinel1_default.json"
    assert cloud_vars_needed(fpath) is False
    assert lookback_days_needed(fpath) == 1
    assert long_lookback_needed(fpath) is True


def test_lookback_days_read_from_the_column_name(tmp_path):
    """The window is whatever the name says, including spans never shipped."""
    from scripts.download_era5 import lookback_days_needed
    expected = {
        "precip_prev24h_mm": 1,
        "precip_prev48h_mm": 2,
        "precip_prev7d_mm": 7,
        "precip_prev14d_mm": 14,
        "precip_prev30d_mm": 30,
        "precip_prev45d_mm": 45,
        "ssrd_prev30d_mj_m2": 30,
        "gdd_prev30d_c": 30,
        "swvl1_prev30d_mean": 30,
        "dry_streak_days": 30,
    }
    for col, days in expected.items():
        path = _write(tmp_path, f"prof_{col}.json",
                      {"rules": {"x": {"metric_column": col, "operator": "gt", "threshold": 0.0}}})
        assert lookback_days_needed(path) == days, col
        assert long_lookback_needed(path) is True


def test_column_without_a_window_needs_no_history(tmp_path):
    from scripts.download_era5 import lookback_days_needed
    path = _write(tmp_path, "prof_plain.json",
                  {"rules": {"x": {"metric_column": "mean_temp_c", "operator": "gt", "threshold": 0.0}}})
    assert lookback_days_needed(path) == 0
    assert long_lookback_needed(path) is False


def test_buffer_covers_the_window_across_a_short_february():
    """A 30-day window over March needs January too: February is 28 days.

    One buffer month was assumed before, so precip_prev30d_mm was missing for
    the first two days of every March.
    """
    from scripts.download_era5 import buffer_months
    assert buffer_months(2024, 3, 30) == [(2024, 1), (2024, 2)]
    assert buffer_months(2024, 8, 30) == [(2024, 7)]
    assert buffer_months(2024, 3, 7) == [(2024, 2)]
    assert buffer_months(2024, 1, 30) == [(2023, 12)]
    assert buffer_months(2024, 8, 0) == []


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
