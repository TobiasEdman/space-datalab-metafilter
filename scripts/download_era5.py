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

CDS-Beta zip wrapping
---------------------
CDS-Beta defaults to packaging ERA5 retrieves as a zip archive containing
the requested NetCDF, even when ``data_format: 'netcdf'`` is set. The fix
is to explicitly request ``download_format: 'unarchived'`` in the body,
which makes CDS skip the zip wrapping. Without that, downstream
``xarray.open_dataset()`` raises the misleading error "did not find a
match in any of xarray's currently installed IO backends" on the zip
that was written under the user-supplied .nc filename.
"""
import calendar
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from metafilter.core import LEGACY_RULE_DEFAULTS, _open_and_concat, lookback_days
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

# Metric-column prefix -> the CDS ERA5-Land variable calculate_daily_metrics
# derives it from. Cloud columns (tcc_*, lcc_*) come from ERA5 single levels
# and are covered by cloud_vars_needed().
_METRIC_SOURCE_VARIABLES = (
    (("mean_temp_", "min_temp_", "max_temp_", "freeze_", "gdd_"), "2m_temperature"),
    (("total_precip_", "precip_", "dry_streak_"), "total_precipitation"),
    (("ssrd_",), "surface_solar_radiation_downwards"),
    (("skt_",), "skin_temperature"),
    (("stl1_",), "soil_temperature_level_1"),
    (("swvl1_",), "volumetric_soil_water_layer_1"),
    (("snow_depth_",), "snow_depth"),
)


def _load_filter_cfg(filter_path):
    """Return the raw filter file as a dict (may be legacy flat or new schema)."""
    with open(filter_path, "r") as f:
        return json.load(f)


def _rules_from_filter(filter_path):
    """Return the dict of rule_name → rule_config regardless of file format."""
    cfg = _load_filter_cfg(filter_path)
    return cfg.get("rules", cfg)


def backend_for_filter(filter_path, default="cds"):
    """Return the requested backend ('cds' | 'open-meteo'), defaulting to CDS.

    The backend selector lives on the filter profile so the JSON is the single
    source of truth for both *what* to filter and *where* the source data
    comes from. Legacy flat filters have no `backend` field → CDS.
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


def era5_land_variables_for_filter(filter_path):
    """ERA5-Land variables the profile's active rules need, in retrieval order.

    Fetching only the legacy temperature/precipitation pair leaves the
    extended profiles without their inputs: the S1 profile then fails on a
    missing `skt_at_pass_c`, the extended S2 profile on `ssrd_mj_m2`.
    """
    needed = set()
    for rule_name, rule in _rules_from_filter(filter_path).items():
        column = rule.get("metric_column") or LEGACY_RULE_DEFAULTS.get(rule_name, {}).get(
            "metric_column", ""
        )
        if column.startswith(("tcc_", "lcc_")):
            continue
        for prefixes, variable in _METRIC_SOURCE_VARIABLES:
            if column.startswith(prefixes):
                needed.add(variable)
                break
        else:
            raise ValueError(
                f"rule {rule_name!r}: no ERA5-Land variable is known for metric column "
                f"{column!r}"
            )
    return [variable for variable in ERA5_LAND_VARIABLES if variable in needed]


# dry_streak_days carries no window in its name: the streak is unbounded and
# simply runs back as far as the data goes. Without history it restarts at
# zero on the first of every month and understates every streak that crosses
# the boundary, so it asks for the month of context it has always been given.
_UNBOUNDED_LOOKBACK_DAYS = {"dry_streak_days": 30}


def lookback_days_needed(filter_path):
    """Longest span of trailing data, in days, that the profile's rules need.

    The span is read from the column name, so a profile can introduce a window
    the downloader has never seen and still get enough history.
    """
    columns = [
        rule.get("metric_column")
        for rule in _rules_from_filter(filter_path).values()
        if isinstance(rule, dict)
    ]
    named = lookback_days(columns)
    unbounded = max(
        (_UNBOUNDED_LOOKBACK_DAYS.get(c, 0) for c in columns if isinstance(c, str)),
        default=0,
    )
    return max(named, unbounded)


def long_lookback_needed(filter_path):
    """True if the profile needs trailing data from before the requested month."""
    return lookback_days_needed(filter_path) > 0


def buffer_months(year, month, days_needed):
    """Preceding months to fetch so `days_needed` of history precede day one.

    One month was assumed before, which silently fell short whenever the
    preceding month was shorter than the window: a 30-day window over March
    saw only February's 28 days, leaving the column missing for the first two
    days of every March.
    """
    months = []
    covered = 0
    cursor = (year, month)
    while covered < days_needed:
        cursor = _previous_month(*cursor)
        months.insert(0, cursor)
        covered += calendar.monthrange(*cursor)[1]
    return months


def _previous_month(year, month):
    return (year, month - 1) if month > 1 else (year - 1, 12)


def download_era5_land(year=2024, month=8, variables=None, area=None):
    """Retrieve a month plus the following 00:00 boundary over `area`.

    Defaults preserve the original main-branch behaviour: 2024-08, two variables,
    `data/era5/era5_land_<year>_<MM>.nc`. Extended call sites pass `variables`
    explicitly to fetch the wider set required by the new filter profiles.
    """
    import cdsapi

    area = area or AREA
    variables = variables or ERA5_LAND_LEGACY_VARIABLES
    out_path = f"{OUTPUT_DIR}/era5/era5_land_{year}_{month:02d}.nc"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    c = cdsapi.Client()
    request = {
        "variable": variables,
        "year": str(year),
        "month": f"{month:02d}",
        "day": [f"{day:02d}" for day in range(1, calendar.monthrange(year, month)[1] + 1)],
        "time": [f"{hour:02d}:00" for hour in range(24)],
        # CDS expects [north, west, south, east].
        "area": [area["north"], area["west"], area["south"], area["east"]],
        "data_format": "netcdf",
        # Without this, CDS-Beta wraps the NetCDF in a zip archive and
        # writes it under the user-supplied .nc filename — confusing
        # every downstream xarray.open_dataset() call.
        "download_format": "unarchived",
    }
    boundary = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthBegin(1)
    boundary_request = {
        **request,
        "year": str(boundary.year),
        "month": f"{boundary.month:02d}",
        "day": ["01"],
        "time": ["00:00"],
    }
    # CDS date fields form a Cartesian product: a separate one-hour request
    # avoids downloading an entire additional month for the boundary.
    with TemporaryDirectory(dir=Path(out_path).parent) as scratch:
        month_path = str(Path(scratch) / "month.nc")
        boundary_path = str(Path(scratch) / "boundary.nc")
        c.retrieve("reanalysis-era5-land", request, month_path)
        c.retrieve("reanalysis-era5-land", boundary_request, boundary_path)
        dataset = _open_and_concat([month_path, boundary_path])
        dataset = dataset.sel(time=slice(f"{year}-{month:02d}-01", boundary))
        if boundary not in pd.DatetimeIndex(dataset["time"].values):
            raise ValueError("CDS response is missing the following midnight boundary")
        complete_path = Path(scratch) / "complete.nc"
        # The files may use different integer packing scales. Persist the
        # decoded values, not the first file's inherited scale/dtype, which
        # can overflow when packing a larger boundary accumulation.
        dataset.drop_encoding().to_netcdf(
            complete_path, engine="scipy",
            encoding={"time": {"units": "seconds since 1970-01-01", "dtype": "float64"}},
        )
        os.replace(complete_path, out_path)
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
            "variable": ERA5_SINGLE_LEVELS_VARIABLES,
            "year": str(year),
            "month": f"{month:02d}",
            "day": [f"{day:02d}" for day in range(1, 32)],
            "time": [f"{hour:02d}:00" for hour in range(24)],
            "area": [area["north"], area["west"], area["south"], area["east"]],
            "data_format": "netcdf",
            # Without this, CDS-Beta wraps the NetCDF in a zip archive and
            # writes it under the user-supplied .nc filename — confusing
            # every downstream xarray.open_dataset() call.
            "download_format": "unarchived",
        },
        out_path,
    )
    return out_path


def download_period(year, month, filter_path, area=None, variables=None):
    """Fetch primary month + optional buffer month + optional cloud data.

    Inspects the active filter profile and dispatches to the requested backend:

      - `backend: "cds"` (default) — issue `reanalysis-era5-land` retrieves,
        plus `reanalysis-era5-single-levels` when any rule needs cloud cover.
      - `backend: "open-meteo"` — single HTTP call per month against the
        Historical Archive; cloud cover comes in the same response so the
        returned `cloud` list is empty (signal lives in the land NetCDF).

    In both cases, prepends the previous month when any active rule needs
    ≥7-day trailing data so rolling-window columns resolve on the first day
    of the requested month.

    Returns a dict `{"land": [paths…], "cloud": [paths…]}`. Either list can
    have one or two entries depending on buffer + cloud needs.

    The returned land/cloud path lists are designed to be passed directly to
    `calculate_daily_metrics(file_path=...)` and `cloud_file_path=...` — both
    accept a list and concatenate along the time axis.
    """
    months_to_fetch = buffer_months(year, month, lookback_days_needed(filter_path))
    months_to_fetch.append((year, month))

    backend = backend_for_filter(filter_path)
    if backend == "open-meteo":
        # Open-Meteo response includes cloud cover; no separate retrieval.
        from scripts.download_open_meteo import download_open_meteo_land
        land_paths = [
            download_open_meteo_land(y, m, area=area, variables=variables)
            for y, m in months_to_fetch
        ]
        return {"land": land_paths, "cloud": []}

    if backend != "cds":
        raise ValueError(
            f"Unknown backend {backend!r} in filter {filter_path!r}; "
            f"supported: 'cds', 'open-meteo'."
        )

    if variables is None:
        variables = era5_land_variables_for_filter(filter_path)
    land_paths = [
        download_era5_land(y, m, variables=variables, area=area)
        for y, m in months_to_fetch
    ]

    cloud_paths = []
    if cloud_vars_needed(filter_path):
        cloud_paths = [download_era5_cloud(y, m, area=area) for y, m in months_to_fetch]

    return {"land": land_paths, "cloud": cloud_paths}


def backend_for_filter(filter_path, default="cds"):
    """Return the requested backend ('cds' | 'open-meteo'), defaulting to CDS.

    The backend selector lives on the filter profile so the JSON is the single
    source of truth for both *what* to filter and *where* the source data
    comes from. Legacy flat filters have no `backend` field → CDS.
    """
    cfg = _load_filter_cfg(filter_path)
    return cfg.get("backend", default)


def _load_filter_cfg(filter_path):
    """Return the raw filter file as a dict (may be legacy flat or new schema)."""
    with open(filter_path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    download_era5_land()
