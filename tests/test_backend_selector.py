"""Tests for the backend selector wiring in download_era5.

`backend_for_filter()` reads the optional `backend` field. `download_period()`
dispatches to `download_era5_land` or `download_open_meteo_land` based on it.
"""
import json

import pytest

from scripts.download_era5 import (
    backend_for_filter,
    download_period,
)


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def test_backend_for_filter_defaults_to_cds(tmp_path):
    payload = {"rules": {"t": {"metric_column": "mean_temp_c", "operator": "gt", "threshold": 10}}}
    p = _write(tmp_path, "no_backend.json", payload)
    assert backend_for_filter(p) == "cds"


def test_backend_for_filter_extracts_open_meteo(tmp_path):
    payload = {"backend": "open-meteo", "rules": {}}
    p = _write(tmp_path, "om.json", payload)
    assert backend_for_filter(p) == "open-meteo"


def test_backend_for_filter_works_on_legacy_flat(tmp_path):
    """Legacy flat dict has no `backend` field; default CDS."""
    payload = {
        "temperature": {"metric_column": "mean_temp_c", "operator": "gt", "threshold": 15.0}
    }
    p = _write(tmp_path, "legacy.json", payload)
    assert backend_for_filter(p) == "cds"


def test_download_period_dispatches_to_open_meteo(tmp_path, monkeypatch):
    payload = {
        "backend": "open-meteo",
        "rules": {"t": {"metric_column": "mean_temp_c", "operator": "gt", "threshold": 10}},
    }
    p = _write(tmp_path, "om.json", payload)

    land_calls, cloud_calls = [], []
    monkeypatch.setattr(
        "scripts.download_open_meteo.download_open_meteo_land",
        lambda y, m, area=None: land_calls.append((y, m)) or f"/tmp/om_{y}_{m}.nc",
    )
    monkeypatch.setattr(
        "scripts.download_era5.download_era5_land",
        lambda y, m, area=None: cloud_calls.append(("LAND", y, m))
                                 or f"/tmp/cds_{y}_{m}.nc",
    )

    result = download_period(2024, 8, p)
    assert land_calls == [(2024, 8)]
    assert cloud_calls == []  # CDS land() must NOT be called
    assert result["cloud"] == []


def test_download_period_open_meteo_still_prepends_buffer(tmp_path, monkeypatch):
    """Long-lookback rule with open-meteo backend: still fetches buffer month."""
    payload = {
        "backend": "open-meteo",
        "rules": {
            "p30": {"metric_column": "precip_prev30d_mm", "operator": "between", "threshold": [30, 150]}
        },
    }
    p = _write(tmp_path, "om_long.json", payload)

    calls = []
    monkeypatch.setattr(
        "scripts.download_open_meteo.download_open_meteo_land",
        lambda y, m, area=None: calls.append((y, m)) or f"/tmp/om_{y}_{m}.nc",
    )

    result = download_period(2024, 8, p)
    assert calls == [(2024, 7), (2024, 8)]
    assert result["cloud"] == []


def test_download_period_rejects_unknown_backend(tmp_path):
    payload = {"backend": "made-up", "rules": {}}
    p = _write(tmp_path, "bad.json", payload)
    with pytest.raises(ValueError, match="Unknown backend"):
        download_period(2024, 8, p)
