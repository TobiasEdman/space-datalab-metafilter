"""Schema loader tests — legacy flat format must still work; new format normalizes."""
import json

import pytest

from scripts.process_era5 import (
    MetafilterConfigurationError,
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
            "name": "Daily mean temperature",
            "metric_column": "mean_temp_c",
            "operator": "gt",
            "threshold": 15.0,
            "unit": "Celsius",
        },
        "precipitation": {
            "name": "Daily total precipitation",
            "metric_column": "total_precip_mm",
            "operator": "lt",
            "threshold": 1.0,
            "unit": "mm/day",
        },
    }
    path = _write(tmp_path, "legacy.json", payload)
    normalized = load_metafilter_parameters(path)

    assert normalized["sensor"] == "sentinel-2"
    assert normalized["overpass_time_utc"] == "10:30"
    assert set(normalized["rules"].keys()) == {"temperature", "precipitation"}


def test_new_format_passes_through_sensor_and_overpass(tmp_path):
    payload = {
        "sensor": "sentinel-1",
        "overpass_time_utc": "17:00",
        "rules": {
            "frost_free": {
                "metric_column": "skt_at_pass_c",
                "operator": "gt",
                "threshold": 2.0,
            }
        },
    }
    path = _write(tmp_path, "s1.json", payload)
    normalized = load_metafilter_parameters(path)

    assert normalized["sensor"] == "sentinel-1"
    assert normalized["overpass_time_utc"] == "17:00"
    assert "frost_free" in normalized["rules"]


def test_new_format_fills_default_overpass_for_sensor():
    payload = {"sensor": "sentinel-1", "rules": {}}
    normalized = _normalize_metafilter_payload(payload)
    assert normalized["overpass_time_utc"] == "05:30"

    payload = {"sensor": "sentinel-2", "rules": {}}
    normalized = _normalize_metafilter_payload(payload)
    assert normalized["overpass_time_utc"] == "10:30"


def test_legacy_inferred_defaults_for_named_rules():
    """Legacy 'temperature' / 'precipitation' bare-bones rules pick up
    metric_column + operator from LEGACY_RULE_DEFAULTS."""
    payload = {
        "temperature": {"threshold": 15.0},
        "precipitation": {"threshold": 1.0},
    }
    rules = normalize_metafilter_rules(payload)

    by_name = {r["rule_name"]: r for r in rules}
    assert by_name["temperature"]["metric_column"] == "mean_temp_c"
    assert by_name["temperature"]["operator"] == "gt"
    assert by_name["precipitation"]["metric_column"] == "total_precip_mm"
    assert by_name["precipitation"]["operator"] == "lt"


def test_between_threshold_must_be_two_elements():
    payload = {
        "rules": {
            "bad": {
                "metric_column": "precip_prev30d_mm",
                "operator": "between",
                "threshold": 50.0,
            }
        }
    }
    with pytest.raises(MetafilterConfigurationError, match="not \\[low, high\\]"):
        normalize_metafilter_rules(payload)


def test_between_threshold_rejects_low_above_high():
    payload = {
        "rules": {
            "bad": {
                "metric_column": "precip_prev30d_mm",
                "operator": "between",
                "threshold": [150.0, 30.0],
            }
        }
    }
    with pytest.raises(MetafilterConfigurationError, match="low > high"):
        normalize_metafilter_rules(payload)


def test_unknown_operator_rejected():
    payload = {
        "rules": {
            "bad": {
                "metric_column": "mean_temp_c",
                "operator": "approx_eq",
                "threshold": 15.0,
            }
        }
    }
    with pytest.raises(MetafilterConfigurationError, match="unsupported operator"):
        normalize_metafilter_rules(payload)


def test_shipped_filter_files_load_cleanly():
    """Smoke test — every JSON in filters/ must normalize without errors."""
    import pathlib
    filters_dir = pathlib.Path(__file__).resolve().parent.parent / "filters"
    for fpath in filters_dir.glob("*.json"):
        normalized = load_metafilter_parameters(fpath)
        # Must produce a valid rule list
        rules = normalize_metafilter_rules(normalized)
        assert rules, f"{fpath.name} produced empty rule list"
