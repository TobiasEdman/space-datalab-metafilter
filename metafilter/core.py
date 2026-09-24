import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .config import AREA


COMPARISON_OPERATORS = {
    "gt": {
        "apply": lambda series, threshold: series > threshold,
        "symbol": ">",
    },
    "ge": {
        "apply": lambda series, threshold: series >= threshold,
        "symbol": ">=",
    },
    "lt": {
        "apply": lambda series, threshold: series < threshold,
        "symbol": "<",
    },
    "le": {
        "apply": lambda series, threshold: series <= threshold,
        "symbol": "<=",
    },
    "between": {
        "apply": lambda series, threshold: (series >= threshold[0]) & (series <= threshold[1]),
        "symbol": "in",
    },
    "abs_lt": {
        "apply": lambda series, threshold: series.abs() < threshold,
        "symbol": "|x| <",
    },
}

LEGACY_RULE_DEFAULTS = {
    "temperature": {
        "metric_column": "mean_temp_c",
        "operator": "gt",
        "name": "Daily mean temperature",
    },
    "precipitation": {
        "metric_column": "total_precip_mm",
        "operator": "lt",
        "name": "Daily total precipitation",
    },
}

DEFAULT_OVERPASS_TIME_UTC = {
    "sentinel-2": "10:30",
    "sentinel-1": "05:30",
}


class MetafilterError(ValueError):
    pass


class MetafilterConfigurationError(MetafilterError):
    pass


class MetafilterSelectionError(MetafilterError):
    def __init__(self, message, daily_metrics=None, rule_summaries=None):
        super().__init__(message)
        self.daily_metrics = daily_metrics
        self.rule_summaries = rule_summaries or []


def load_metafilter_parameters(json_file):
    with open(json_file, "r") as file:
        payload = json.load(file)
    return _normalize_metafilter_payload(payload)


def _normalize_metafilter_payload(payload):
    """Normalize both legacy (flat dict of rules) and new ({sensor, rules}) formats.

    Returns the new format every time, so downstream code only handles one shape.
    Legacy format is detected by absence of a top-level "rules" key. The optional
    `backend` field (cds | open-meteo) is preserved when present and defaulted
    to "cds" otherwise — the value is consumed by scripts/download_era5.py to
    pick the data source, not by the rule engine itself.
    """
    if not isinstance(payload, dict):
        raise MetafilterConfigurationError(
            "Metafilter profile must be a JSON object."
        )

    if "rules" in payload:
        if not isinstance(payload["rules"], dict):
            raise MetafilterConfigurationError(
                "Metafilter profile 'rules' must be a JSON object."
            )
        sensor = payload.get("sensor", "sentinel-2")
        overpass = payload.get(
            "overpass_time_utc",
            DEFAULT_OVERPASS_TIME_UTC.get(sensor, "10:30"),
        )
        return {
            "sensor": sensor,
            "overpass_time_utc": overpass,
            "backend": payload.get("backend", "cds"),
            "rules": payload["rules"],
        }
    # Legacy: flat dict — every key is a rule. Treat as sentinel-2.
    return {
        "sensor": "sentinel-2",
        "overpass_time_utc": DEFAULT_OVERPASS_TIME_UTC["sentinel-2"],
        "backend": "cds",
        "rules": payload,
    }


_ERA5_GRID_DEG = 0.25

# Rolling-lookback columns are named for their window, and the name is the
# contract: filter profiles select by it, and the downloader reads it to size
# the buffer it must fetch. So the window is parsed out of the name rather
# than hard-coded, and a profile can ask for any span without a code change.
#
#   precip_prev14d_mm  ->  14-day rolling sum of total_precip_mm
#
# `_LOOKBACK_STEMS` maps the stem to (source column, aggregation, suffix).
_LOOKBACK_STEMS = {
    "precip": ("total_precip_mm", "sum", "mm"),
    "ssrd": ("ssrd_mj_m2", "sum", "mj_m2"),
    "gdd": ("gdd_daily_c", "sum", "c"),
    "swvl1": ("swvl1_mean", "mean", "mean"),
}

_LOOKBACK_PATTERN = re.compile(
    r"^(?P<stem>" + "|".join(_LOOKBACK_STEMS) + r")_prev(?P<span>\d+)(?P<unit>[dh])_(?P<suffix>.+)$"
)

# What calculate_daily_metrics produces when the caller names nothing. These
# are the columns the shipped profiles and AnalogModel.DEFAULT_FEATURES use,
# so the default frame is unchanged from before windows became parseable.
DEFAULT_LOOKBACK_COLUMNS = (
    "precip_prev24h_mm",
    "precip_prev48h_mm",
    "precip_prev7d_mm",
    "precip_prev30d_mm",
    "ssrd_prev30d_mj_m2",
    "gdd_prev30d_c",
    "swvl1_prev30d_mean",
)


def parse_lookback(column):
    """Return (stem, days, aggregation) for a lookback column, else None.

    Hours are accepted because the shipped profiles use `prev24h`/`prev48h`
    for the short spans; they must be whole days, since the underlying series
    is daily.
    """
    match = _LOOKBACK_PATTERN.match(column)
    if match is None:
        return None
    stem = match.group("stem")
    source, aggregation, suffix = _LOOKBACK_STEMS[stem]
    if match.group("suffix") != suffix:
        return None
    span = int(match.group("span"))
    if match.group("unit") == "h":
        if span % 24:
            raise MetafilterConfigurationError(
                f"lookback column {column!r} asks for {span} hours; only whole days are available."
            )
        span //= 24
    if span < 1:
        raise MetafilterConfigurationError(f"lookback column {column!r} asks for a window under one day.")
    return stem, span, aggregation


def lookback_days(columns):
    """Longest window, in days, among the lookback columns in `columns`."""
    spans = [parsed[1] for parsed in map(parse_lookback, columns) if parsed]
    return max(spans, default=0)

# Accumulation convention of the hourly source, carried as a dataset attribute.
# CDS ERA5-Land accumulates tp/ssrd from the 00 UTC forecast start; Open-Meteo
# serves per-hour sums. Untagged data is CDS output — the only source that
# arrives without our tag — so that is the default, never a guess from values.
ACCUMULATION_ATTR = "metafilter_accumulation"
ACCUMULATION_FORECAST_START = "forecast_start"
ACCUMULATION_HOURLY = "hourly"


def _nearest_cell_within_grid_step(dataset, area, grid_deg=_ERA5_GRID_DEG):
    """Select the grid cell nearest the AOI centroid, within one grid step.

    Fetches snap small AOIs to the ~0.25° ERA5 grid so neighbouring AOIs
    share cache, which means a single-cell dataset can lie just outside
    the exact AOI bounds. Honour that snapping contract instead of
    failing; anything farther than one grid step away is a genuine
    mismatch and returns None.
    """
    center_lat = (area["south"] + area["north"]) / 2
    center_lon = (area["west"] + area["east"]) / 2
    try:
        return dataset.sel(
            latitude=[center_lat],
            longitude=[center_lon],
            method="nearest",
            tolerance=grid_deg,
        )
    except KeyError:
        return None


def subset_dataset_to_area(dataset, area):
    latitude = dataset["latitude"]
    longitude = dataset["longitude"]

    if latitude[0] > latitude[-1]:
        latitude_slice = slice(area["north"], area["south"])
    else:
        latitude_slice = slice(area["south"], area["north"])

    if longitude[0] > longitude[-1]:
        longitude_slice = slice(area["east"], area["west"])
    else:
        longitude_slice = slice(area["west"], area["east"])

    return dataset.sel(latitude=latitude_slice, longitude=longitude_slice)


def _netcdf_engine(path):
    """Choose an xarray backend from the file signature when possible."""
    with Path(path).open("rb") as stream:
        signature = stream.read(8)
    if signature[:4] in {b"CDF\x01", b"CDF\x02"}:
        return "scipy"
    if signature == b"\x89HDF\r\n\x1a\n":
        return "h5netcdf"
    return None


def _open_and_concat(file_paths):
    """Open one or more NetCDF files and concatenate along the time dimension.

    Used to stitch a buffer month onto the primary month so long lookback
    windows (7d, 30d) can reach back beyond the start of the primary month.
    """
    if isinstance(file_paths, (str, Path)):
        file_paths = [file_paths]
    if not file_paths:
        raise ValueError("at least one NetCDF path is required")

    datasets = []
    for path in file_paths:
        path = Path(path)
        engine = _netcdf_engine(path)
        kwargs = {"engine": engine} if engine else {}
        with xr.open_dataset(path, **kwargs) as opened:
            ds = opened
            if "valid_time" in ds.coords or "valid_time" in ds.dims:
                ds = ds.rename({"valid_time": "time"})
            # Per file, never after concatenation: a cache written before the
            # `sde` correction carries `sd`, and concatenating it with a newer
            # file yields both variables, each missing outside its own months.
            # Selecting one globally would drop the other file's observations.
            # A file that already has `sde` keeps it and its `sd` is left
            # alone, since `sd` elsewhere in ERA5 is snow water equivalent.
            if "sde" not in ds.data_vars and "sd" in ds.data_vars:
                ds = ds.rename({"sd": "sde"})
            datasets.append(ds.load())

    combined = datasets[0] if len(datasets) == 1 else xr.concat(datasets, dim="time")
    return combined.sortby("time").drop_duplicates(dim="time")


def _hourly_increments(var):
    """Per-hour increments of a forecast-start accumulation.

    ERA5-Land stamps D hh:00 (hh = 1..23) with the total since D 00:00 and
    stamps D+1 00:00 with day D's 24-hour total. The increment is therefore
    the difference from the previous hour, except at 01:00 where it is the
    value itself. Like every hourly increment here, the result is stamped at
    the hour it ends; the first sample has no predecessor and becomes NaN.
    """
    hours = var["time"].dt.hour
    increments = var - var.shift(time=1)
    return increments.where(hours != 1, var)


def _daily_sum(hourly, spatial, *, active_cells):
    """Sum hour-ending increments per calendar day.

    A value stamped 00:00 is the last hour of the previous day, so the series
    is moved back one hour before resampling. All 24 hourly values must be
    present; incomplete days remain NaN rather than becoming partial totals.
    """
    shifted = hourly.assign_coords(time=hourly["time"] - pd.Timedelta(hours=1))
    daily = shifted.resample(time="1D").sum(min_count=24)
    if spatial:
        # Ignore permanently masked cells, but never silently drop a cell
        # with a transient gap from the AOI mean (it may be the wet cell).
        # Activity comes from raw observations: differencing can erase an
        # isolated cumulative reading when its predecessor is missing.
        incomplete = (shifted.resample(time="1D").count() < 24) & active_cells
        daily = daily.mean(dim=spatial, skipna=True).where(~incomplete.any(dim=spatial))
    return daily


def _select_area(source, area, what):
    """Subset to the AOI, falling back to the nearest cell within one grid step.

    Raises MetafilterSelectionError when the dataset genuinely does not cover
    the AOI; `what` names the dataset in the message.
    """
    dataset = subset_dataset_to_area(source, area)
    if dataset.sizes.get("latitude", 0) == 0 or dataset.sizes.get("longitude", 0) == 0:
        dataset = _nearest_cell_within_grid_step(source, area)
    if (
        dataset is None
        or dataset.sizes.get("latitude", 0) == 0
        or dataset.sizes.get("longitude", 0) == 0
    ):
        raise MetafilterSelectionError(f"Configured AREA does not overlap the {what} dataset.")
    return dataset


def _sample_at_overpass(var, overpass_time_utc, spatial_dims):
    """Sample an hourly variable at the closest hourly slot to overpass_time_utc.

    overpass_time_utc: 'HH:MM' string. Rounded to nearest hour.
    If no matching hourly slot exists (e.g. coarser data), fall back to daily mean.
    """
    hour = int(overpass_time_utc.split(":")[0])
    minute = int(overpass_time_utc.split(":")[1])
    if minute >= 30:
        hour = (hour + 1) % 24

    times = pd.to_datetime(var["time"].values)
    mask = times.hour == hour
    if not mask.any():
        return var.resample(time="1D").mean().mean(dim=spatial_dims, skipna=True).values

    sampled = var.isel(time=mask)
    daily = sampled.resample(time="1D").mean()
    if spatial_dims:
        daily = daily.mean(dim=spatial_dims, skipna=True)
    return daily.values


def calculate_daily_metrics(
    file_path,
    area=AREA,
    *,
    sensor="sentinel-2",
    overpass_time_utc=None,
    cloud_file_path=None,
    lookback_columns=DEFAULT_LOOKBACK_COLUMNS,
):
    """Compute daily aggregates for surface state and (optionally) cloud cover.

    Args:
        file_path: path to an ERA5-Land NetCDF, OR a list of paths (primary + buffer).
        area: AOI dict with west/east/south/north.
        sensor: 'sentinel-2' or 'sentinel-1' — currently only changes default overpass time.
        overpass_time_utc: 'HH:MM'. Defaults from sensor when omitted.
        cloud_file_path: path (or list) to ERA5 single-levels NetCDF with tcc/lcc.
            Required only if active rules reference cloud columns.

    Returns:
        pd.DataFrame indexed by 'date' with all derivable columns. Columns whose
        source variables are absent from the NetCDF are silently skipped.
    """
    if overpass_time_utc is None:
        overpass_time_utc = DEFAULT_OVERPASS_TIME_UTC.get(sensor, "10:30")

    dataset = _select_area(_open_and_concat(file_path), area, "ERA5")

    spatial = tuple(d for d in ("latitude", "longitude") if d in dataset.dims)

    accumulation = dataset.attrs.get(ACCUMULATION_ATTR, ACCUMULATION_FORECAST_START)
    if accumulation not in (ACCUMULATION_FORECAST_START, ACCUMULATION_HOURLY):
        raise MetafilterConfigurationError(
            f"Unknown {ACCUMULATION_ATTR} value {accumulation!r}; expected "
            f"{ACCUMULATION_FORECAST_START!r} or {ACCUMULATION_HOURLY!r}."
        )

    # A lone trailing midnight completes yesterday's accumulations. It is
    # not a new day of temperature/cloud observations or a selectable date.
    accumulation_dataset = dataset
    last = pd.Timestamp(dataset["time"].values[-1])
    if last == last.normalize():
        dataset = dataset.isel(time=slice(None, -1))
    if dataset.sizes["time"] == 0:
        raise MetafilterSelectionError("ERA5 contains only a boundary sample, no daily observations")
    daily_time_index = pd.to_datetime(
        dataset["time"].resample(time="1D").mean()["time"].values
    )
    df = pd.DataFrame({"date": daily_time_index.strftime("%Y-%m-%d")})

    def hourly_increments(name):
        var = accumulation_dataset[name]
        if accumulation == ACCUMULATION_FORECAST_START:
            var = _hourly_increments(var)
        return var

    def align_daily(daily):
        series = pd.Series(daily.values, index=pd.to_datetime(daily["time"].values))
        return series.reindex(daily_time_index).values

    # ── Temperature ────────────────────────────────────────────────────
    if "t2m" in dataset:
        t_c = dataset["t2m"] - 273.15
        df["mean_temp_c"] = (
            t_c.resample(time="1D").mean().mean(dim=spatial, skipna=True).values
        )
        df["min_temp_c"] = (
            t_c.resample(time="1D").min().mean(dim=spatial, skipna=True).values
        )
        df["max_temp_c"] = (
            t_c.resample(time="1D").max().mean(dim=spatial, skipna=True).values
        )
        df["freeze_flag"] = (df["min_temp_c"] < 0).astype(int)

    # ── Precipitation + lookbacks ─────────────────────────────────────
    if "tp" in dataset:
        precip = _daily_sum(
            hourly_increments("tp") * 1000.0, spatial,
            active_cells=accumulation_dataset["tp"].notnull().any(dim="time"),
        )
        df["total_precip_mm"] = align_daily(precip)

    # ── Solar radiation ───────────────────────────────────────────────
    if "ssrd" in dataset:
        ssrd_daily_mj = align_daily(_daily_sum(
            hourly_increments("ssrd"), spatial,
            active_cells=accumulation_dataset["ssrd"].notnull().any(dim="time"),
        ) / 1e6)
        df["ssrd_mj_m2"] = ssrd_daily_mj

    # ── GDD accumulation (requires mean_temp_c) ───────────────────────
    if "mean_temp_c" in df.columns:
        # Kept out of the returned frame; it exists only to feed gdd_prev*_c.
        gdd_source = (df["mean_temp_c"].clip(lower=5) - 5).fillna(0)

    # ── Surface state (S1-relevant, but cheap to always compute) ──────
    if "skt" in dataset:
        skt_c = dataset["skt"] - 273.15
        df["skt_mean_c"] = (
            skt_c.resample(time="1D").mean().mean(dim=spatial, skipna=True).values
        )
        df["skt_min_c"] = (
            skt_c.resample(time="1D").min().mean(dim=spatial, skipna=True).values
        )
        df["skt_at_pass_c"] = _sample_at_overpass(skt_c, overpass_time_utc, spatial)

    if "stl1" in dataset:
        stl1_c = dataset["stl1"] - 273.15
        df["stl1_mean_c"] = (
            stl1_c.resample(time="1D").mean().mean(dim=spatial, skipna=True).values
        )

    if "swvl1" in dataset:
        swvl1_daily = (
            dataset["swvl1"].resample(time="1D").mean().mean(dim=spatial, skipna=True).values
        )
        df["swvl1_mean"] = swvl1_daily
        df["swvl1_delta_prev2d"] = (
            pd.Series(swvl1_daily) - pd.Series(swvl1_daily).shift(2)
        ).fillna(0).values

    # CDS `reanalysis-era5-land` returns `snow_depth` as `sde` (snow depth in
    # metres) - verified against the live API 2026-09-18. `sd` is ERA5's snow
    # *water equivalent* elsewhere, and the spelling Open-Meteo caches written
    # before this fix carry, so it is accepted second and never preferred.
    # `_open_and_concat` has already mapped the legacy `sd` spelling per file.
    if "sde" in dataset:
        df["snow_depth_mean_m"] = (
            dataset["sde"].resample(time="1D").mean()
            .mean(dim=spatial, skipna=True).values
        )

    # ── Rolling lookbacks, one per requested column ───────────────────
    # Every window is a shifted rolling aggregate of a daily series, so the
    # only thing that varies is the span. `min_periods` equals the span: a
    # partially covered window stays missing rather than reporting a total
    # over fewer days than it claims.
    sources = {"gdd": locals().get("gdd_source")}
    for column in lookback_columns:
        parsed = parse_lookback(column)
        if parsed is None:
            raise MetafilterConfigurationError(
                f"{column!r} is not a recognised lookback column; expected "
                f"<{'|'.join(_LOOKBACK_STEMS)}>_prev<N>d_<suffix>."
            )
        stem, span, aggregation = parsed
        source_name = _LOOKBACK_STEMS[stem][0]
        series = sources.get(stem)
        if series is None:
            if source_name not in df.columns:
                continue  # the input lacks the variable this window needs
            series = df[source_name]
        rolling = pd.Series(series.values if hasattr(series, "values") else series).rolling(
            span, min_periods=span
        )
        aggregated = rolling.sum() if aggregation == "sum" else rolling.mean()
        df[column] = aggregated.shift(1).values

    # Dry streak: consecutive dry days ending yesterday
    if "total_precip_mm" in df.columns:
        streak = 0.0
        streaks = []
        for rain in df["total_precip_mm"]:
            streaks.append(streak)
            if pd.isna(rain):
                streak = float("nan")
            elif rain >= 0.5:
                streak = 0.0
            else:
                streak += 1
        df["dry_streak_days"] = streaks

    # ── Cloud cover when present in the same dataset ──────────────────
    # The CDS path delivers tcc/lcc in a separate `reanalysis-era5-single-levels`
    # NetCDF (handled below). The Open-Meteo path bundles them in the same JSON
    # response → the variables live on `dataset` directly. We detect both shapes.
    if "tcc" in dataset:
        df["tcc_mean_overpass"] = _sample_at_overpass(
            dataset["tcc"], overpass_time_utc, spatial
        )
    if "lcc" in dataset:
        df["lcc_mean_overpass"] = _sample_at_overpass(
            dataset["lcc"], overpass_time_utc, spatial
        )

    # ── Cloud cover (separate ERA5 single-levels file) ────────────────
    # An empty list is what download_period returns for a backend that bundles
    # cloud cover with the land variables; treat it like no file at all.
    if cloud_file_path:
        cds = _select_area(_open_and_concat(cloud_file_path), area, "cloud")
        cspatial = tuple(d for d in ("latitude", "longitude") if d in cds.dims)
        if "tcc" in cds:
            df["tcc_mean_overpass"] = _sample_at_overpass(
                cds["tcc"], overpass_time_utc, cspatial
            )
        if "lcc" in cds:
            df["lcc_mean_overpass"] = _sample_at_overpass(
                cds["lcc"], overpass_time_utc, cspatial
            )

    return df


def normalize_metafilter_rules(metafilter_params):
    """Validate + flatten the rule list. Accepts either the new {sensor, rules}
    form or the legacy flat form (already normalized by load_metafilter_parameters,
    but kept for direct callers passing raw dicts)."""
    if "rules" in metafilter_params:
        rules_dict = metafilter_params["rules"]
    else:
        rules_dict = metafilter_params

    normalized_rules = []
    for rule_name, rule_config in rules_dict.items():
        defaults = LEGACY_RULE_DEFAULTS.get(rule_name, {})
        merged_rule = {**defaults, **rule_config}

        if "threshold" not in merged_rule:
            raise MetafilterConfigurationError(
                f"Metafilter rule '{rule_name}' is missing 'threshold'."
            )

        metric_column = merged_rule.get("metric_column")
        if not metric_column:
            raise MetafilterConfigurationError(
                f"Metafilter rule '{rule_name}' is missing 'metric_column'."
            )

        operator = merged_rule.get("operator")
        if operator not in COMPARISON_OPERATORS:
            supported = ", ".join(sorted(COMPARISON_OPERATORS))
            raise MetafilterConfigurationError(
                f"Metafilter rule '{rule_name}' has unsupported operator '{operator}'. "
                f"Supported operators: {supported}."
            )

        if operator == "between":
            threshold = merged_rule["threshold"]
            if not (isinstance(threshold, (list, tuple)) and len(threshold) == 2):
                raise MetafilterConfigurationError(
                    f"Metafilter rule '{rule_name}' uses 'between' but threshold is not "
                    f"[low, high]; got {threshold!r}."
                )
            if threshold[0] > threshold[1]:
                raise MetafilterConfigurationError(
                    f"Metafilter rule '{rule_name}' has 'between' threshold low > high."
                )

        normalized_rules.append(
            {
                "rule_name": rule_name,
                "name": merged_rule.get("name", rule_name.replace("_", " ").title()),
                "description": merged_rule.get("description"),
                "metric_column": metric_column,
                "operator": operator,
                "threshold": merged_rule["threshold"],
                "unit": merged_rule.get("unit"),
            }
        )

    return normalized_rules


def build_rule_summary(rule, metric_series, pass_mask):
    valid_values = metric_series.dropna()
    passed_days = int(pass_mask.fillna(False).sum())
    summary = {
        "rule_name": rule["rule_name"],
        "name": rule["name"],
        "description": rule.get("description"),
        "metric_column": rule["metric_column"],
        "operator": rule["operator"],
        "threshold": rule["threshold"],
        "unit": rule.get("unit"),
        "valid_days": int(valid_values.shape[0]),
        "passed_days": passed_days,
        "observed_min": None,
        "observed_max": None,
    }

    if not valid_values.empty:
        summary["observed_min"] = float(valid_values.min())
        summary["observed_max"] = float(valid_values.max())

    return summary


def apply_metafilter(daily_metrics, metafilter_params):
    filtered_metrics = daily_metrics.copy()
    normalized_rules = normalize_metafilter_rules(metafilter_params)

    selected_mask = pd.Series(True, index=filtered_metrics.index, dtype=bool)
    rule_summaries = []
    available_metrics = ", ".join(sorted(filtered_metrics.columns))

    for rule in normalized_rules:
        metric_column = rule["metric_column"]
        if metric_column not in filtered_metrics.columns:
            raise MetafilterConfigurationError(
                f"Metafilter rule '{rule['rule_name']}' requires metric column "
                f"'{metric_column}', but the current ERA5 processing only provides: "
                f"{available_metrics}."
            )

        metric_series = pd.to_numeric(filtered_metrics[metric_column], errors="coerce")
        operator = COMPARISON_OPERATORS[rule["operator"]]
        pass_mask = operator["apply"](metric_series, rule["threshold"])
        filtered_metrics[f"rule__{rule['rule_name']}"] = pass_mask.fillna(False)
        selected_mask &= pass_mask.fillna(False)
        rule_summaries.append(build_rule_summary(rule, metric_series, pass_mask))

    filtered_metrics["selected"] = selected_mask
    return filtered_metrics, rule_summaries


def format_rule_condition(rule_summary):
    symbol = COMPARISON_OPERATORS[rule_summary["operator"]]["symbol"]
    unit = f" {rule_summary['unit']}" if rule_summary.get("unit") else ""
    return f"{symbol} {rule_summary['threshold']}{unit}"


def format_rule_brief(rule):
    return f"{rule['name']} ({rule['metric_column']} {format_rule_condition(rule)})"


def format_rule_summary(rule_summary, total_days):
    base_text = (
        f"- {rule_summary['name']}: {rule_summary['passed_days']}/{total_days} days "
        f"satisfied {format_rule_condition(rule_summary)}"
    )

    if rule_summary["observed_min"] is not None and rule_summary["observed_max"] is not None:
        observed = (
            f"; observed range was {rule_summary['observed_min']:.2f} "
            f"to {rule_summary['observed_max']:.2f}"
        )
        if rule_summary.get("unit"):
            observed = f"{observed} {rule_summary['unit']}"
        base_text = f"{base_text}{observed}"
    else:
        base_text = (
            f"{base_text}; no valid values were available for metric "
            f"'{rule_summary['metric_column']}'"
        )

    if rule_summary.get("description"):
        base_text = f"{base_text}. {rule_summary['description']}"

    return base_text


def format_selection_error_message(rule_summaries, total_days):
    summary_lines = [
        "No dates matched the configured metafilter rules.",
        f"Evaluated {total_days} candidate day(s).",
        "Rule diagnostics:",
    ]
    summary_lines.extend(
        format_rule_summary(rule_summary, total_days)
        for rule_summary in rule_summaries
    )
    return "\n".join(summary_lines)


def process_era5_data(
    file_path,
    metafilter_params,
    area=AREA,
    *,
    cloud_file_path=None,
):
    """End-to-end: NetCDF → daily metrics → filter application → date lists.

    cloud_file_path is forwarded to calculate_daily_metrics so cloud-cover rules
    work transparently when an ERA5 single-levels file is supplied.
    """
    normalized = (
        metafilter_params
        if "rules" in metafilter_params
        else _normalize_metafilter_payload(metafilter_params)
    )
    sensor = normalized.get("sensor", "sentinel-2")
    overpass = normalized.get(
        "overpass_time_utc", DEFAULT_OVERPASS_TIME_UTC.get(sensor, "10:30")
    )
    # A profile may name any window it likes; compute those on top of the
    # defaults so a rule never meets a missing column, and the frame keeps
    # the columns other callers already rely on.
    requested = [
        rule.get("metric_column", "")
        for rule in normalized.get("rules", {}).values()
        if isinstance(rule, dict)
    ]
    lookbacks = list(DEFAULT_LOOKBACK_COLUMNS)
    lookbacks += [
        column for column in requested
        if parse_lookback(column) and column not in lookbacks
    ]

    daily_metrics = calculate_daily_metrics(
        file_path,
        area=area,
        sensor=sensor,
        overpass_time_utc=overpass,
        cloud_file_path=cloud_file_path,
        lookback_columns=lookbacks,
    )
    filtered_metrics, rule_summaries = apply_metafilter(daily_metrics, normalized)

    all_dates = filtered_metrics["date"].tolist()
    selected_dates = filtered_metrics.loc[filtered_metrics["selected"], "date"].tolist()
    if not selected_dates:
        raise MetafilterSelectionError(
            format_selection_error_message(rule_summaries, total_days=len(filtered_metrics)),
            daily_metrics=filtered_metrics,
            rule_summaries=rule_summaries,
        )

    return {
        "all_dates": all_dates,
        "selected_dates": selected_dates,
        "full_temporal_extent": [all_dates[0], all_dates[-1]],
        "selected_temporal_extent": [selected_dates[0], selected_dates[-1]],
        "daily_metrics": filtered_metrics,
        "rule_summaries": rule_summaries,
    }


def save_daily_metrics(daily_metrics, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily_metrics.to_csv(output_path, index=False)


if __name__ == "__main__":
    metafilter_file = "filters/metafilter.json"
    metafilter_params = load_metafilter_parameters(metafilter_file)
    try:
        results = process_era5_data("data/era5/era5_land_july_2024.nc", metafilter_params)
    except MetafilterError as exc:
        print(exc)
        raise SystemExit(1)
    print(results["selected_dates"])
