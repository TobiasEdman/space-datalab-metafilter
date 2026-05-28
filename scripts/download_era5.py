"""ERA5 download wrappers used by main.py and ad-hoc CLI calls.

Two CDS datasets are involved:

* `reanalysis-era5-land` — high-res (~9 km) surface state. Default source for
  temperature, precipitation, soil moisture, snow, radiation, skin temperature.
* `reanalysis-era5-single-levels` — ~30 km, atmospheric variables including
  total/low cloud cover (not in ERA5-Land). Fetched only when active rules
  reference `tcc_*` / `lcc_*` columns.

`download_period()` is the convenience wrapper main.py / new code should call.
It looks at the active filter profile and decides:

* whether to fetch a buffer (previous) month (when any rule uses ≥7d lookback)
* whether to also fetch ERA5 single-levels cloud cover

so that downstream NetCDF inputs match the lookback windows the filter expects.
"""
import json
from pathlib import Path

from utils.config import AREA, OUTPUT_DIR


ERA5_LAND_VARIABLES = [
    "2m_temperature",
    "total_precipitation",
    "surface_solar_radiation_downwards",
    "skin_temperature",
    "soil_temperature_level_1",
    "volumetric_soil_water_layer_1",
    "snow_depth",
    "snowfall",
    "2m_dewpoint_temperature",
]

ERA5_LAND_LEGACY_VARIABLES = ["2m_temperature", "total_precipitation"]

ERA5_SINGLE_LEVELS_VARIABLES = [
    "total_cloud_cover",
    "low_cloud_cover",
]

_LONG_LOOKBACK_PREFIXES = (
    "precip_prev7d_",
    "precip_prev30d_",
    "ssrd_prev30d_",
    "gdd_prev30d_",
    "swvl1_prev30d_",
    "dry_streak_",
)


def _rules_from_filter(filter_path):
    """Return the dict of rule_name → rule_config regardless of file format."""
    with open(filter_path, "r") as f:
        cfg = json.load(f)
    return cfg.get("rules", cfg)


def cloud_vars_needed(filter_path):
    """True if any active rule references tcc_*/lcc_* metric columns."""
    rules = _rules_from_filter(filter_path)
    return any(
        rule.get("metric_column", "").startswith(("tcc_", "lcc_"))
        for rule in rules.values()
    )


def long_lookback_needed(filter_path):
    """True if any active rule needs ≥7 days of trailing data (requires buffer month)."""
    rules = _rules_from_filter(filter_path)
    return any(
        any(rule.get("metric_column", "").startswith(p) for p in _LONG_LOOKBACK_PREFIXES)
        for rule in rules.values()
    )


def _previous_month(year, month):
    return (year, month - 1) if month > 1 else (year - 1, 12)


def download_era5_land(year=2024, month=8, variables=None, area=None):
    """Retrieve one month of ERA5-Land hourly data over `area`.

    Defaults preserve the original main-branch behaviour: 2024-08, two variables,
    `data/era5/era5_land_<year>_<MM>.nc`. Extended call sites pass `variables`
    explicitly to fetch the wider set required by the new filter profiles.
    """
    import cdsapi

    area = area or AREA
    variables = variables or ERA5_LAND_LEGACY_VARIABLES
    out_path = f"{OUTPUT_DIR}/era5/era5_land_{year}_{month:02d}.nc"

    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-land",
        {
            "variable": variables,
            "year": str(year),
            "month": f"{month:02d}",
            "day": [f"{day:02d}" for day in range(1, 32)],
            "time": [f"{hour:02d}:00" for hour in range(24)],
            # CDS expects [north, west, south, east].
            "area": [area["north"], area["west"], area["south"], area["east"]],
            "data_format": "netcdf",
        },
        out_path,
    )
    return out_path


def download_era5_cloud(year, month, area=None):
    """Retrieve one month of ERA5 single-levels cloud cover. Returns NetCDF path."""
    import cdsapi

    area = area or AREA
    out_path = f"{OUTPUT_DIR}/era5/era5_clouds_{year}_{month:02d}.nc"

    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": ERA5_SINGLE_LEVELS_VARIABLES,
            "year": str(year),
            "month": f"{month:02d}",
            "day": [f"{day:02d}" for day in range(1, 32)],
            "time": [f"{hour:02d}:00" for hour in range(24)],
            "area": [area["north"], area["west"], area["south"], area["east"]],
            "data_format": "netcdf",
        },
        out_path,
    )
    return out_path


def download_period(year, month, filter_path, area=None, variables=None):
    """Fetch primary month + optional buffer month + optional cloud data.

    Inspects the active filter profile and:
      - prepends the previous month to the fetch list when any rule needs
        ≥7-day trailing data (otherwise the first ~30 days of the primary
        month produce NaN in the rolling lookback columns).
      - also fetches ERA5 single-levels cloud cover when any rule references
        tcc_/lcc_ columns.

    Returns a dict {"land": [paths…], "cloud": [paths…]}. Either list can
    have one or two entries depending on buffer + cloud needs.

    The returned land/cloud path lists are designed to be passed directly to
    `calculate_daily_metrics(file_path=...)` and `cloud_file_path=...` — both
    accept a list and concatenate along the time axis.
    """
    months_to_fetch = [(year, month)]
    if long_lookback_needed(filter_path):
        months_to_fetch.insert(0, _previous_month(year, month))

    land_paths = [
        download_era5_land(y, m, variables=variables, area=area)
        for y, m in months_to_fetch
    ]

    cloud_paths = []
    if cloud_vars_needed(filter_path):
        cloud_paths = [download_era5_cloud(y, m, area=area) for y, m in months_to_fetch]

    return {"land": land_paths, "cloud": cloud_paths}


if __name__ == "__main__":
    download_era5_land()
