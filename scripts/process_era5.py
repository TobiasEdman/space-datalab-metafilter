import json
from pathlib import Path

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


def _normalize_metafilter_payload(payload: dict) -> dict:
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


def calculate_daily_metrics(file_path, area=AREA):
    dataset = xr.open_dataset(file_path)
    if "valid_time" in dataset.coords or "valid_time" in dataset.dims:
        dataset = dataset.rename({"valid_time": "time"})

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
    daily_total_precip = precipitation_mm.resample(time="1D").sum()

    if spatial_dims:
        daily_mean_temp = daily_mean_temp.mean(dim=spatial_dims, skipna=True)
        daily_total_precip = daily_total_precip.mean(dim=spatial_dims, skipna=True)

    return pd.DataFrame(
        {
            "date": pd.to_datetime(daily_mean_temp["time"].values).strftime("%Y-%m-%d"),
            "mean_temp_c": daily_mean_temp.values,
            "total_precip_mm": daily_total_precip.values,
        }
    )


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
