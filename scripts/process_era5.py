import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from utils.config import AREA


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
        # Inclusive on both ends. Threshold must be a 2-element [low, high] list.
        "apply": lambda series, threshold: (series >= threshold[0]) & (series <= threshold[1]),
        "symbol": "in",
    },
    "abs_lt": {
        # Absolute value comparison — useful for "stability" / "small change" gates.
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

# Default satellite overpass time (UTC) per sensor. Used by future
# pass-time-sampled metric columns (cloud cover, skin temperature, etc.).
# S2 → ~10:30 descending pass over Sweden. S1 → ~05:30 descending /
# ~17:00 ascending — descending is the default.
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
    """Load a filter profile and normalize to the new {sensor, overpass_time_utc, rules}
    shape. Accepts the legacy flat-dict form for backward compatibility."""
    with open(json_file, "r") as file:
        payload = json.load(file)
    return _normalize_metafilter_payload(payload)


def _normalize_metafilter_payload(payload):
    """Normalize both legacy (flat dict of rules) and new ({sensor, rules}) formats.

    Returns the new shape every time, so downstream code only handles one form:
        {"sensor": str, "overpass_time_utc": "HH:MM", "rules": {name: rule, ...}}

    Legacy format is detected by the absence of a top-level "rules" key — the
    whole dict is then treated as the rule set, with sensor defaulting to
    sentinel-2 and overpass_time_utc to the matching default.
    """
    if "rules" in payload:
        sensor = payload.get("sensor", "sentinel-2")
        overpass = payload.get(
            "overpass_time_utc",
            DEFAULT_OVERPASS_TIME_UTC.get(sensor, "10:30"),
        )
        return {
            "sensor": sensor,
            "overpass_time_utc": overpass,
            "rules": payload["rules"],
        }
    # Legacy: flat dict — every key is a rule. Treat as sentinel-2.
    return {
        "sensor": "sentinel-2",
        "overpass_time_utc": DEFAULT_OVERPASS_TIME_UTC["sentinel-2"],
        "rules": payload,
    }


def _sample_at_overpass(var, overpass_time_utc, spatial_dims):
    """Sample an hourly variable at the closest hourly slot to overpass_time_utc.

    `overpass_time_utc` is "HH:MM" — rounded to the nearest whole hour.
    Falls back to daily mean if the dataset has no matching hour slot
    (e.g. when the input is already daily-aggregated).
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


def _open_and_concat(file_paths):
    """Open one or more NetCDF files and stitch along the time dimension.

    Used to attach a buffer (previous) month to the primary month so the
    long-lookback rolling columns can resolve on the first day of the
    requested period.
    """
    if isinstance(file_paths, (str, Path)):
        file_paths = [file_paths]

    datasets = []
    for path in file_paths:
        ds = xr.open_dataset(path)
        if "valid_time" in ds.coords or "valid_time" in ds.dims:
            ds = ds.rename({"valid_time": "time"})
        datasets.append(ds)

    if len(datasets) == 1:
        return datasets[0]
    combined = xr.concat(datasets, dim="time")
    return combined.sortby("time").drop_duplicates(dim="time")


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


def calculate_daily_metrics(
    file_path,
    area=AREA,
    *,
    sensor="sentinel-2",
    overpass_time_utc=None,
):
    """Open one or more ERA5-Land NetCDFs and return a per-day DataFrame.

    `file_path` accepts a single path or a list of paths; multiple files
    are concatenated along time (deduplicated) so a buffer month can
    feed the long-window lookback columns.

    `sensor` and `overpass_time_utc` together select the hour at which
    pass-time-sampled columns (`skt_at_pass_c`, plus future cloud-cover
    columns) are evaluated. `overpass_time_utc` defaults to the matching
    `DEFAULT_OVERPASS_TIME_UTC[sensor]` when omitted.
    """
    if overpass_time_utc is None:
        overpass_time_utc = DEFAULT_OVERPASS_TIME_UTC.get(sensor, "10:30")

    dataset = _open_and_concat(file_path)

    dataset = subset_dataset_to_area(dataset, area)
    if dataset.sizes.get("latitude", 0) == 0 or dataset.sizes.get("longitude", 0) == 0:
        raise MetafilterSelectionError("Configured AREA does not overlap the ERA5 dataset.")

    temperature_c = dataset["t2m"] - 273.15
    precipitation_mm = dataset["tp"] * 1000.0

    spatial_dims = tuple(
        dimension
        for dimension in ("latitude", "longitude")
        if dimension in temperature_c.dims
    )

    daily_mean_temp = temperature_c.resample(time="1D").mean()
    daily_min_temp = temperature_c.resample(time="1D").min()
    daily_max_temp = temperature_c.resample(time="1D").max()
    daily_total_precip = precipitation_mm.resample(time="1D").sum()

    if spatial_dims:
        daily_mean_temp = daily_mean_temp.mean(dim=spatial_dims, skipna=True)
        daily_min_temp = daily_min_temp.mean(dim=spatial_dims, skipna=True)
        daily_max_temp = daily_max_temp.mean(dim=spatial_dims, skipna=True)
        daily_total_precip = daily_total_precip.mean(dim=spatial_dims, skipna=True)

    # Short-window precipitation lookbacks: shift(1) so day N looks at the
    # window ending the day before N — no leakage of day N's own rain into
    # its own "antecedent wetness" rule.
    total_precip = pd.Series(daily_total_precip.values)
    precip_prev24h = total_precip.shift(1).fillna(0).values
    precip_prev48h = (
        total_precip.rolling(2, min_periods=1).sum().shift(1).fillna(0).values
    )

    # Long-window lookbacks: strict windows (min_periods=N) so day N+ has a
    # value only when the full N-day history is available. NaN otherwise.
    precip_prev7d = (
        total_precip.rolling(7, min_periods=7).sum().shift(1).values
    )
    precip_prev30d = (
        total_precip.rolling(30, min_periods=30).sum().shift(1).values
    )

    # Dry-streak: consecutive dry (<0.5 mm) days ending yesterday.
    precip_prev = total_precip.shift(1).fillna(0)
    is_wet = (precip_prev >= 0.5).astype(int)
    dry_streak_days = is_wet.groupby(is_wet.cumsum()).cumcount().values

    min_temp_values = daily_min_temp.values
    freeze_flag = (pd.Series(min_temp_values) < 0).astype(int).values

    # GDD (base 5 °C) accumulated over the previous 30 days.
    mean_temp_series = pd.Series(daily_mean_temp.values)
    gdd_daily = (mean_temp_series.clip(lower=5) - 5).fillna(0)
    gdd_prev30d = (
        gdd_daily.rolling(30, min_periods=30).sum().shift(1).values
    )

    columns = {
        "date": pd.to_datetime(daily_mean_temp["time"].values).strftime("%Y-%m-%d"),
        "mean_temp_c": daily_mean_temp.values,
        "min_temp_c": min_temp_values,
        "max_temp_c": daily_max_temp.values,
        "freeze_flag": freeze_flag,
        "total_precip_mm": daily_total_precip.values,
        "precip_prev24h_mm": precip_prev24h,
        "precip_prev48h_mm": precip_prev48h,
        "precip_prev7d_mm": precip_prev7d,
        "precip_prev30d_mm": precip_prev30d,
        "dry_streak_days": dry_streak_days,
        "gdd_prev30d_c": gdd_prev30d,
    }

    # Optional: solar radiation. Only emitted when ssrd is in the dataset.
    if "ssrd" in dataset:
        ssrd_daily_mj = (
            dataset["ssrd"]
            .resample(time="1D").sum()
            .mean(dim=spatial_dims, skipna=True).values
            / 1e6
        )
        columns["ssrd_mj_m2"] = ssrd_daily_mj
        columns["ssrd_prev30d_mj_m2"] = (
            pd.Series(ssrd_daily_mj).rolling(30, min_periods=30).sum().shift(1).values
        )

    # Optional: skin temperature. Daily extremes + pass-time sample.
    if "skt" in dataset:
        skt_c = dataset["skt"] - 273.15
        columns["skt_mean_c"] = (
            skt_c.resample(time="1D").mean().mean(dim=spatial_dims, skipna=True).values
        )
        columns["skt_min_c"] = (
            skt_c.resample(time="1D").min().mean(dim=spatial_dims, skipna=True).values
        )
        columns["skt_at_pass_c"] = _sample_at_overpass(
            skt_c, overpass_time_utc, spatial_dims
        )

    # Optional: soil temperature layer 1.
    if "stl1" in dataset:
        stl1_c = dataset["stl1"] - 273.15
        columns["stl1_mean_c"] = (
            stl1_c.resample(time="1D").mean().mean(dim=spatial_dims, skipna=True).values
        )

    # Optional: surface soil moisture (0–7 cm).
    if "swvl1" in dataset:
        swvl1_daily = (
            dataset["swvl1"].resample(time="1D").mean()
            .mean(dim=spatial_dims, skipna=True).values
        )
        swvl1_series = pd.Series(swvl1_daily)
        columns["swvl1_mean"] = swvl1_daily
        columns["swvl1_delta_prev2d"] = (
            (swvl1_series - swvl1_series.shift(2)).fillna(0).values
        )
        columns["swvl1_prev30d_mean"] = (
            swvl1_series.rolling(30, min_periods=30).mean().shift(1).values
        )

    # Optional: snow depth (metres).
    if "sd" in dataset:
        columns["snow_depth_mean_m"] = (
            dataset["sd"].resample(time="1D").mean()
            .mean(dim=spatial_dims, skipna=True).values
        )

    return pd.DataFrame(columns)


def normalize_metafilter_rules(metafilter_params):
    """Validate + flatten the rule list. Accepts either the new
    `{sensor, rules}` form (as normalized by `load_metafilter_parameters`)
    or the legacy flat form (for direct callers passing raw dicts)."""
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
                    f"Metafilter rule '{rule_name}' uses 'between' but threshold is "
                    f"not [low, high]; got {threshold!r}."
                )
            if threshold[0] > threshold[1]:
                raise MetafilterConfigurationError(
                    f"Metafilter rule '{rule_name}' has 'between' threshold "
                    f"low > high ({threshold[0]} > {threshold[1]})."
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


def process_era5_data(file_path, metafilter_params, area=AREA):
    daily_metrics = calculate_daily_metrics(file_path, area=area)
    filtered_metrics, rule_summaries = apply_metafilter(daily_metrics, metafilter_params)

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
