"""CDS retrieval entry points.

Two retrievals are supported:

* `download_era5_land()` — `reanalysis-era5-land`, the high-res surface
  product. Default source for temperature, precipitation, soil moisture,
  snow, skin temperature, radiation.
* `download_era5_cloud()` — `reanalysis-era5-single-levels`, fetched only
  when the active filter profile references total/low cloud cover.
  ERA5-Land does not include `tcc`/`lcc`.

`download_period()` is the convenience wrapper that inspects a filter
profile and decides which retrievals + how many months are needed.
"""
from __future__ import annotations

import json
from pathlib import Path

from utils.config import AREA, OUTPUT_DIR


_LONG_LOOKBACK_PREFIXES = (
    "precip_prev7d_",
    "precip_prev30d_",
    "ssrd_prev30d_",
    "gdd_prev30d_",
    "swvl1_prev30d_",
    "dry_streak_",
)


ERA5_SINGLE_LEVELS_CLOUD_VARIABLES = [
    "total_cloud_cover",
    "low_cloud_cover",
]


def download_era5_land(year=2024, month=8, area=None):
    import cdsapi

    area = area or AREA
    out_path = f"{OUTPUT_DIR}/era5/era5_land_{year}_{month:02d}.nc"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-land",
        {
            "variable": ["2m_temperature", "total_precipitation"],
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
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": ERA5_SINGLE_LEVELS_CLOUD_VARIABLES,
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


def _load_filter_cfg(filter_path: str | Path) -> dict:
    """Return the raw filter JSON as a dict (legacy flat or new schema)."""
    with open(filter_path, "r") as f:
        return json.load(f)


def _rules_from_filter(filter_path):
    """Return the dict of rule_name → rule_config regardless of file format."""
    cfg = _load_filter_cfg(filter_path)
    return cfg.get("rules", cfg)


def backend_for_filter(filter_path: str | Path, default: str = "cds") -> str:
    """Return the requested backend ("cds" or "open-meteo"), defaulting to CDS.

    The backend selector lives on the filter profile so the JSON is the
    single source of truth for both *what* to filter and *where* the
    source data comes from. Legacy flat filters have no `backend` field
    → CDS.
    """
    cfg = _load_filter_cfg(filter_path)
    return cfg.get("backend", default)


def cloud_vars_needed(filter_path):
    """True if any active rule references tcc_*/lcc_* metric columns."""
    rules = _rules_from_filter(filter_path)
    return any(
        rule.get("metric_column", "").startswith(("tcc_", "lcc_"))
        for rule in rules.values()
    )


def long_lookback_needed(filter_path):
    """True if any active rule needs ≥7 days of trailing data."""
    rules = _rules_from_filter(filter_path)
    return any(
        any(rule.get("metric_column", "").startswith(p) for p in _LONG_LOOKBACK_PREFIXES)
        for rule in rules.values()
    )


def _previous_month(year, month):
    return (year, month - 1) if month > 1 else (year - 1, 12)


def download_period(year, month, filter_path, area=None):
    """Inspect the active filter profile and dispatch the right retrievals.

    Backend selection comes from the `backend` field in the filter JSON
    (defaults to "cds"):

      "cds"        → reanalysis-era5-land + optional reanalysis-era5-single-levels
      "open-meteo" → scripts.download_open_meteo.download_open_meteo_land
                     (cloud cover bundled with land — no separate cloud retrieve)

    Returns `{"land": [paths…], "cloud": [paths…]}`. The land list has one
    or two entries: just the primary month, or the previous month
    prepended when any active rule needs long-window lookback. The cloud
    list mirrors the same months whenever any active rule references
    `tcc_*`/`lcc_*` columns (CDS only); otherwise it's empty.

    Both lists feed directly into `calculate_daily_metrics(file_path=...)`
    and `cloud_file_path=...` — they each accept a list and concatenate
    along time internally.
    """
    months = [(year, month)]
    if long_lookback_needed(filter_path):
        months.insert(0, _previous_month(year, month))

    backend = backend_for_filter(filter_path)
    if backend == "open-meteo":
        # Cloud cover ships in the same Open-Meteo response, so no separate
        # cloud retrieve. Lazy import so the CDS path never pays for it.
        from scripts.download_open_meteo import download_open_meteo_land
        land_paths = [download_open_meteo_land(y, m, area=area) for y, m in months]
        return {"land": land_paths, "cloud": []}

    if backend != "cds":
        raise ValueError(
            f"Unknown backend {backend!r} in {filter_path!r}; "
            f"supported: 'cds', 'open-meteo'."
        )

    land_paths = [download_era5_land(y, m, area=area) for y, m in months]

    cloud_paths = []
    if cloud_vars_needed(filter_path):
        cloud_paths = [download_era5_cloud(y, m, area=area) for y, m in months]

    return {"land": land_paths, "cloud": cloud_paths}


if __name__ == "__main__":
    download_era5_land()
