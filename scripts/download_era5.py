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

Known CDS-side quirk handled here
---------------------------------
The CDS API sometimes returns a *zip archive* containing the NetCDF file
rather than the NetCDF itself, even when ``data_format: 'netcdf'`` is
requested. The zip is written to the user-supplied path with the user-
supplied filename — including ``.nc`` extension — so any downstream
``xarray.open_dataset(...)`` raises the misleading error "did not find a
match in any of xarray's currently installed IO backends". After every
successful retrieve we therefore check the magic bytes and, if they're a
zip header, extract the inner NetCDF in place via
``_maybe_extract_zipped_netcdf``.
"""
import json
import shutil
import zipfile
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

# ZIP local-file-header magic. Used to detect CDS responses that arrive as
# a zip archive wrapping the NetCDF instead of the NetCDF directly.
_ZIP_MAGIC = b"PK\x03\x04"


def _maybe_extract_zipped_netcdf(path):
    """If `path` is actually a zip archive (CDS quirk), extract the NetCDF
    inside and replace the zip with the extracted file. Otherwise no-op.

    Returns the path (unchanged) for chaining.

    Raises RuntimeError if the file is a zip but contains zero or more than
    one .nc member — that shape isn't documented anywhere as a CDS response
    and indicates we should fail loudly rather than guess.
    """
    path = Path(path)
    with open(path, "rb") as f:
        magic = f.read(4)

    if magic != _ZIP_MAGIC:
        return path  # already a valid NetCDF/GRIB/etc., nothing to do

    with zipfile.ZipFile(path) as zf:
        members = zf.namelist()
        nc_members = [m for m in members if m.lower().endswith(".nc")]
        if not nc_members:
            raise RuntimeError(
                f"{path} is a zip archive but contains no .nc files "
                f"(members: {members}). Cannot auto-extract."
            )
        if len(nc_members) > 1:
            raise RuntimeError(
                f"{path} is a zip archive with multiple .nc files "
                f"(members: {nc_members}). Ambiguous which to extract."
            )

        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with zf.open(nc_members[0]) as src, open(tmp_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

    # Atomic replace, so a half-extracted file never lingers at `path`.
    tmp_path.replace(path)
    return path


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

    Auto-extracts the result if CDS returned a zip-wrapped NetCDF.
    """
    import cdsapi

    area = area or AREA
    variables = variables or ERA5_LAND_LEGACY_VARIABLES
    out_path = f"{OUTPUT_DIR}/era5/era5_land_{year}_{month:02d}.nc"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

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
    _maybe_extract_zipped_netcdf(out_path)
    return out_path


def download_era5_cloud(year, month, area=None):
    """Retrieve one month of ERA5 single-levels cloud cover. Returns NetCDF path.

    Auto-extracts the result if CDS returned a zip-wrapped NetCDF.
    """
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
        },
        out_path,
    )
    _maybe_extract_zipped_netcdf(out_path)
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
    months_to_fetch = [(year, month)]
    if long_lookback_needed(filter_path):
        months_to_fetch.insert(0, _previous_month(year, month))

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
