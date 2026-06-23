"""Tests for `load_metafilter_parameters` / `_normalize_metafilter_payload`.

Both the legacy flat-dict format and the new {sensor, rules} shape must
produce the same internal payload — backward compatibility is the central
guarantee of this PR.
"""
import json

import pytest

from scripts.process_era5 import (
    DEFAULT_OVERPASS_TIME_UTC,
    _normalize_metafilter_payload,
    load_metafilter_parameters,
    normalize_metafilter_rules,
)


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def test_legacy_flat_format_loads_as_sentinel2_default(tmp_path):
    """The shipped filters/metafilter.json (flat) must continue to work."""
    payload = {
        "temperature": {
            "metric_column": "mean_temp_c",
            "operator": "gt",
            "threshold": 15.0,
        },
        "precipitation": {
            "metric_column": "total_precip_mm",
            "operator": "lt",
            "threshold": 1.0,
        },
    }
    path = _write(tmp_path, "legacy.json", payload)
    normalized = load_metafilter_parameters(path)

    assert normalized["sensor"] == "sentinel-2"
    assert normalized["overpass_time_utc"] == "10:30"
    assert set(normalized["rules"].keys()) == {"temperature", "precipitation"}


def test_new_format_round_trips_sensor_and_overpass(tmp_path):
    payload = {
        "sensor": "sentinel-1",
        "overpass_time_utc": "17:00",
        "rules": {
            "warm_enough": {
                "metric_column": "mean_temp_c",
                "operator": "gt",
                "threshold": 2.0,
            }
        },
    }
    path = _write(tmp_path, "s1.json", payload)
    normalized = load_metafilter_parameters(path)

    assert normalized["sensor"] == "sentinel-1"
    assert normalized["overpass_time_utc"] == "17:00"
    assert "warm_enough" in normalized["rules"]


def test_new_format_fills_default_overpass_for_sensor():
    payload = {"sensor": "sentinel-1", "rules": {}}
    normalized = _normalize_metafilter_payload(payload)
    assert normalized["overpass_time_utc"] == DEFAULT_OVERPASS_TIME_UTC["sentinel-1"]

    payload = {"sensor": "sentinel-2", "rules": {}}
    normalized = _normalize_metafilter_payload(payload)
    assert normalized["overpass_time_utc"] == DEFAULT_OVERPASS_TIME_UTC["sentinel-2"]


def test_normalize_metafilter_rules_accepts_both_shapes():
    """Validator must work on the legacy flat dict AND the new {rules:} wrapper."""
    flat = {
        "temperature": {"threshold": 15.0},
        "precipitation": {"threshold": 1.0},
    }
    wrapped = {
        "sensor": "sentinel-2",
        "overpass_time_utc": "10:30",
        "rules": flat,
    }
    assert normalize_metafilter_rules(flat) == normalize_metafilter_rules(wrapped)


def test_shipped_metafilter_json_loads_cleanly():
    """Smoke test against the actual shipped filter file."""
    import pathlib
    fpath = pathlib.Path(__file__).resolve().parent.parent / "filters" / "metafilter.json"
    normalized = load_metafilter_parameters(fpath)
    rules = normalize_metafilter_rules(normalized)
    assert rules, "shipped metafilter.json normalized to an empty rule list"
