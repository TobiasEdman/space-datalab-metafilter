"""Tests for the pure-Python helpers in `scripts/download_era5.py`.

The actual `cdsapi.Client().retrieve(...)` calls are not exercised here —
they need a live ECMWF account. But the profile-inspection helpers and
dispatch logic in `download_period()` are deterministic and unit-testable.
"""
import json

import pytest

from scripts.download_era5 import (
    _previous_month,
    cloud_vars_needed,
    download_period,
    long_lookback_needed,
)


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def test_previous_month_normal():
    assert _previous_month(2024, 8) == (2024, 7)


def test_previous_month_january_wraps_year():
    assert _previous_month(2024, 1) == (2023, 12)


def test_legacy_filter_needs_neither_cloud_nor_buffer(tmp_path):
    payload = {
        "temperature": {"metric_column": "mean_temp_c", "operator": "gt", "threshold": 15.0},
        "precipitation": {"metric_column": "total_precip_mm", "operator": "lt", "threshold": 1.0},
    }
    p = _write(tmp_path, "legacy.json", payload)
    assert cloud_vars_needed(p) is False
    assert long_lookback_needed(p) is False


def test_cloud_vars_needed_detects_tcc_lcc(tmp_path):
    payload = {
        "rules": {
            "c": {"metric_column": "tcc_mean_overpass", "operator": "lt", "threshold": 0.3}
        }
    }
    p = _write(tmp_path, "cloud.json", payload)
    assert cloud_vars_needed(p) is True


@pytest.mark.parametrize(
    "metric",
    ["precip_prev7d_mm", "precip_prev30d_mm", "ssrd_prev30d_mj_m2",
     "gdd_prev30d_c", "swvl1_prev30d_mean", "dry_streak_days"],
)
def test_long_lookback_needed_detected_for_each_prefix(tmp_path, metric):
    payload = {"rules": {"x": {"metric_column": metric, "operator": "gt", "threshold": 0.0}}}
    p = _write(tmp_path, f"prof_{metric}.json", payload)
    assert long_lookback_needed(p), f"{metric} did not trigger long_lookback_needed"


def test_download_period_dispatches_buffer_and_cloud(tmp_path, monkeypatch):
    """download_period should call land() twice (buffer + primary) and
    cloud() twice when the profile needs both."""
    payload = {
        "rules": {
            "p30": {"metric_column": "precip_prev30d_mm", "operator": "between", "threshold": [30, 150]},
            "tcc": {"metric_column": "tcc_mean_overpass", "operator": "lt", "threshold": 0.3},
        }
    }
    p = _write(tmp_path, "ext.json", payload)

    land_calls, cloud_calls = [], []
    monkeypatch.setattr(
        "scripts.download_era5.download_era5_land",
        lambda y, m, area=None: land_calls.append((y, m)) or f"/tmp/land_{y}_{m}.nc",
    )
    monkeypatch.setattr(
        "scripts.download_era5.download_era5_cloud",
        lambda y, m, area=None: cloud_calls.append((y, m)) or f"/tmp/cloud_{y}_{m}.nc",
    )

    result = download_period(2024, 8, p)
    assert land_calls == [(2024, 7), (2024, 8)]   # buffer then primary
    assert cloud_calls == [(2024, 7), (2024, 8)]
    assert len(result["land"]) == 2
    assert len(result["cloud"]) == 2


def test_download_period_skips_buffer_when_only_short_window(tmp_path, monkeypatch):
    payload = {
        "rules": {
            "t": {"metric_column": "mean_temp_c", "operator": "gt", "threshold": 10}
        }
    }
    p = _write(tmp_path, "short.json", payload)

    land_calls, cloud_calls = [], []
    monkeypatch.setattr(
        "scripts.download_era5.download_era5_land",
        lambda y, m, area=None: land_calls.append((y, m)) or f"/tmp/land_{y}_{m}.nc",
    )
    monkeypatch.setattr(
        "scripts.download_era5.download_era5_cloud",
        lambda y, m, area=None: cloud_calls.append((y, m)) or f"/tmp/cloud_{y}_{m}.nc",
    )

    result = download_period(2024, 8, p)
    assert land_calls == [(2024, 8)]
    assert cloud_calls == []
    assert result["cloud"] == []
