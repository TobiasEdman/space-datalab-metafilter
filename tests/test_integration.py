"""End-to-end tests: NetCDF → calculate_daily_metrics → apply_metafilter → dates.

These guard the *combination* of changes — schema normalization + new operators
+ derived columns + multi-file concatenation — against regressions.
"""
import json
import pathlib

import pytest

from scripts.process_era5 import (
    MetafilterSelectionError,
    apply_metafilter,
    calculate_daily_metrics,
    load_metafilter_parameters,
    process_era5_data,
)


FILTERS_DIR = pathlib.Path(__file__).resolve().parent.parent / "filters"


# ── Regression: legacy filter behaves identically to before the PR ─────────

def test_legacy_metafilter_produces_two_required_columns(era5_land_factory, test_area):
    """The shipped filters/metafilter.json requires mean_temp_c + total_precip_mm.
    Both must be present after calculate_daily_metrics."""
    path = era5_land_factory(start="2024-08-01", end="2024-08-10", t2m_pattern="warm")
    df = calculate_daily_metrics(path, area=test_area)
    assert "mean_temp_c" in df.columns
    assert "total_precip_mm" in df.columns


def test_legacy_filter_filters_dry_warm_days(era5_land_factory, test_area):
    """Legacy filter: temp > 15 °C AND precip < 1 mm. Warm + dry-with-event fixture:
    - All days warm (20 °C)
    - One day with 10 mm rain — must be filtered out
    - Other days dry — must pass."""
    path = era5_land_factory(start="2024-08-01", end="2024-08-10",
                             t2m_pattern="warm", tp_pattern="dry_with_event")
    legacy = load_metafilter_parameters(FILTERS_DIR / "metafilter.json")
    df = calculate_daily_metrics(path, area=test_area)
    filtered, _ = apply_metafilter(df, legacy)

    selected = filtered.loc[filtered["selected"], "date"].tolist()
    # The 10-mm event day should not be selected
    assert "2024-08-06" not in selected
    # Other warm dry days should be
    assert "2024-08-02" in selected


# ── New: S1 default profile filters frost / snow / wet canopy days ─────────

def test_s1_filter_passes_warm_dry_snow_free(
    era5_land_factory, test_area
):
    """S1 default profile: should pass days with warm skt, dry canopy, no snow."""
    path = era5_land_factory(start="2024-07-01", end="2024-07-15",
                             t2m_pattern="warm", skt_pattern="warm",
                             sd_pattern="no_snow", tp_pattern="dry")
    s1 = load_metafilter_parameters(FILTERS_DIR / "sentinel1_default.json")
    df = calculate_daily_metrics(path, area=test_area,
                                 sensor=s1["sensor"],
                                 overpass_time_utc=s1["overpass_time_utc"])
    filtered, _ = apply_metafilter(df, s1)
    selected = filtered.loc[filtered["selected"], "date"].tolist()
    # Multiple days should pass — warm + dry + snow-free
    assert len(selected) >= 5


def test_s1_filter_rejects_frozen_days(era5_land_factory, test_area):
    path = era5_land_factory(start="2024-01-15", end="2024-01-25",
                             t2m_pattern="cold", skt_pattern="frost",
                             sd_pattern="no_snow", tp_pattern="dry")
    s1 = load_metafilter_parameters(FILTERS_DIR / "sentinel1_default.json")
    df = calculate_daily_metrics(path, area=test_area,
                                 sensor=s1["sensor"],
                                 overpass_time_utc=s1["overpass_time_utc"])
    filtered, _ = apply_metafilter(df, s1)
    selected = filtered.loc[filtered["selected"], "date"].tolist()
    assert selected == []


def test_s1_filter_rejects_snowy_days(era5_land_factory, test_area):
    path = era5_land_factory(start="2024-02-15", end="2024-02-25",
                             t2m_pattern="warm", skt_pattern="warm",
                             sd_pattern="snow", tp_pattern="dry")
    s1 = load_metafilter_parameters(FILTERS_DIR / "sentinel1_default.json")
    df = calculate_daily_metrics(path, area=test_area,
                                 sensor=s1["sensor"],
                                 overpass_time_utc=s1["overpass_time_utc"])
    filtered, _ = apply_metafilter(df, s1)
    selected = filtered.loc[filtered["selected"], "date"].tolist()
    assert selected == []


# ── New: S2 extended profile requires cloud cover and long lookback ────────

def test_s2_extended_runs_with_buffer_month_and_cloud(
    era5_land_factory, era5_cloud_factory, test_area
):
    """Use buffer-month concat so 30d lookback resolves on the primary month."""
    # July fills the 30d precip band (≥30 mm) but tails off to dry by month-end
    # so the immediate prev24h/48h/7d rules also pass on the August side.
    buffer_land = era5_land_factory(
        start="2024-07-01", end="2024-08-01",
        t2m_pattern="warm", tp_pattern="wet_with_dry_tail",
        swvl1_pattern="flat", ssrd_pattern="summer",
    )
    primary_land = era5_land_factory(
        start="2024-08-01", end="2024-09-01",
        t2m_pattern="warm", tp_pattern="dry",
        swvl1_pattern="flat", ssrd_pattern="summer",
    )
    buffer_cloud = era5_cloud_factory(
        start="2024-07-01", end="2024-08-01",
        tcc_at_10utc=0.10, lcc_at_10utc=0.05,
    )
    primary_cloud = era5_cloud_factory(
        start="2024-08-01", end="2024-09-01",
        tcc_at_10utc=0.10, lcc_at_10utc=0.05,
    )

    s2_ext = load_metafilter_parameters(FILTERS_DIR / "sentinel2_extended.json")
    df = calculate_daily_metrics(
        [buffer_land, primary_land],
        area=test_area,
        sensor=s2_ext["sensor"],
        overpass_time_utc=s2_ext["overpass_time_utc"],
        cloud_file_path=[buffer_cloud, primary_cloud],
    )
    filtered, _ = apply_metafilter(df, s2_ext)
    selected = filtered.loc[filtered["selected"], "date"].tolist()
    # August days should pass: warm + dry + low cloud + buffer gave us GDD/precip context
    assert any(d.startswith("2024-08") for d in selected)


def test_s2_extended_excludes_cloudy_days(
    era5_land_factory, era5_cloud_factory, test_area
):
    """If cloud cover fixture is set high, no days should pass."""
    buffer_land = era5_land_factory(
        start="2024-07-01", end="2024-08-01",
        t2m_pattern="warm", tp_pattern="wet_with_dry_tail", ssrd_pattern="summer",
    )
    primary_land = era5_land_factory(
        start="2024-08-01", end="2024-09-01",
        t2m_pattern="warm", tp_pattern="dry", ssrd_pattern="summer",
    )
    cloud = era5_cloud_factory(
        start="2024-07-01", end="2024-09-01",
        tcc_at_10utc=0.80, lcc_at_10utc=0.60,  # very cloudy
    )

    s2_ext = load_metafilter_parameters(FILTERS_DIR / "sentinel2_extended.json")
    df = calculate_daily_metrics(
        [buffer_land, primary_land],
        area=test_area,
        sensor=s2_ext["sensor"],
        overpass_time_utc=s2_ext["overpass_time_utc"],
        cloud_file_path=cloud,
    )
    filtered, _ = apply_metafilter(df, s2_ext)
    selected = filtered.loc[filtered["selected"], "date"].tolist()
    assert selected == []


# ── process_era5_data convenience wrapper ──────────────────────────────────

def test_process_era5_data_handles_legacy_payload(era5_land_factory, test_area):
    """The original process_era5_data() signature must still work with the
    flat legacy filter file."""
    path = era5_land_factory(start="2024-08-01", end="2024-08-10",
                             t2m_pattern="warm", tp_pattern="dry_with_event")
    legacy = load_metafilter_parameters(FILTERS_DIR / "metafilter.json")
    result = process_era5_data(path, legacy, area=test_area)

    assert "all_dates" in result
    assert "selected_dates" in result
    assert isinstance(result["daily_metrics"], type(result["daily_metrics"]))
    assert len(result["selected_dates"]) >= 1


def test_process_era5_data_raises_when_zero_match(era5_land_factory, test_area):
    """All-cold dataset should produce zero selected dates under legacy temp > 15."""
    path = era5_land_factory(start="2024-01-01", end="2024-01-10", t2m_pattern="cold")
    legacy = load_metafilter_parameters(FILTERS_DIR / "metafilter.json")
    with pytest.raises(MetafilterSelectionError):
        process_era5_data(path, legacy, area=test_area)
