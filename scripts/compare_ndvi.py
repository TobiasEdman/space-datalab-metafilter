import json
from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from utils.config import AREA, NDVI_OUTPUT_DIR


def print_info(message):
    print(f"INFO {message}")


def _get_openeo_connect():
    try:
        from openeo import connect
    except ImportError as exc:
        raise RuntimeError(
            "The openeo package is required for NDVI comparison runs. "
            "Install dependencies from requirements.txt in the intended environment."
        ) from exc

    return connect


def authenticate(username, password, eo_service_url):
    print_info(f"Connecting to openEO backend: {eo_service_url}")
    connect = _get_openeo_connect()
    connection = connect(eo_service_url)
    print_info("Authenticating with basic auth")
    connection.authenticate_basic(username=username, password=password)
    print_info("openEO authentication complete")
    return connection


def to_day_window(day_string):
    current_day = date.fromisoformat(day_string)
    next_day = current_day + timedelta(days=1)
    return [current_day.isoformat(), next_day.isoformat()]


def build_ndvi_cube(connection, temporal_extent, spatial_extent=AREA):
    data_cube = connection.load_collection(
        collection_id="s2_msi_l2a",
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bands=["b04", "b08"],
    )

    try:
        return data_cube.ndvi(red="b04", nir="b08")
    except TypeError:
        return data_cube.ndvi(nir="b08", red="b04")


def download_ndvi_raster(connection, day_string, output_path, spatial_extent=AREA):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        return output_path

    ndvi_cube = build_ndvi_cube(
        connection,
        temporal_extent=to_day_window(day_string),
        spatial_extent=spatial_extent,
    )

    try:
        ndvi_cube.download(str(output_path), format="GTiff")
    except TypeError:
        ndvi_cube.download(outputfile=str(output_path), format="GTiff")

    return output_path


def summarize_ndvi_raster(raster_path):
    with rasterio.open(raster_path) as dataset:
        data = dataset.read(masked=True).astype("float32")
        if data.count() == 0:
            raise ValueError(f"No valid NDVI pixels found in {raster_path}.")

        flattened = data.compressed()
        band_means = [
            round(float(np.ma.mean(data[index])), 4)
            for index in range(dataset.count)
        ]

        return {
            "band_count": int(dataset.count),
            "valid_pixel_count": int(flattened.size),
            "mean_ndvi": round(float(flattened.mean()), 4),
            "median_ndvi": round(float(np.median(flattened)), 4),
            "p90_ndvi": round(float(np.percentile(flattened, 90)), 4),
            "max_ndvi": round(float(flattened.max()), 4),
            "band_means": json.dumps(band_means),
        }


def unique_rule_metrics(rule_summaries):
    unique_rules = []
    seen_metrics = set()

    for rule_summary in rule_summaries:
        metric_column = rule_summary["metric_column"]
        if metric_column in seen_metrics:
            continue
        seen_metrics.add(metric_column)
        unique_rules.append(rule_summary)

    return unique_rules


def format_progress(current, total, width=20):
    if total <= 0:
        return "[--------------------]"

    filled = int(width * current / total)
    if current > 0 and filled == 0:
        filled = 1
    return f"[{'#' * filled}{'-' * (width - filled)}]"


def run_strategy(connection, strategy_name, day_strings, output_dir=NDVI_OUTPUT_DIR):
    results = []
    strategy_dir = Path(output_dir) / strategy_name
    total_days = len(day_strings)

    print_info(
        f"Starting strategy '{strategy_name}' for {total_days} day(s) into {strategy_dir}"
    )

    for index, day_string in enumerate(day_strings, start=1):
        raster_path = strategy_dir / f"{day_string}.tif"
        cached = raster_path.exists()
        progress = format_progress(index, total_days)
        record = {
            "strategy": strategy_name,
            "date": day_string,
            "raster_path": str(raster_path),
            "status": "pending",
        }

        try:
            source_label = "cached" if cached else "download"
            print_info(
                f"{strategy_name} {progress} {index}/{total_days} {day_string} {source_label}"
            )
            download_ndvi_raster(connection, day_string, raster_path)
            record.update(summarize_ndvi_raster(raster_path))
            record["status"] = "ok"
            print_info(
                f"{strategy_name} {day_string} mean_ndvi={record['mean_ndvi']}"
            )
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)
            print_info(f"{strategy_name} {day_string} error: {exc}")

        results.append(record)

    print_info(f"Completed strategy '{strategy_name}'")
    return pd.DataFrame(results)


def summarize_strategy(results_df, ndvi_threshold):
    successful = results_df[results_df["status"] == "ok"].copy()
    summary = {
        "candidate_days": int(len(results_df)),
        "successful_days": int(len(successful)),
        "error_days": int((results_df["status"] == "error").sum()),
        "mean_of_mean_ndvi": None,
        "median_of_mean_ndvi": None,
        "share_above_threshold": None,
        "best_day": None,
        "best_day_mean_ndvi": None,
    }

    if successful.empty:
        return summary

    summary["mean_of_mean_ndvi"] = round(float(successful["mean_ndvi"].mean()), 4)
    summary["median_of_mean_ndvi"] = round(float(successful["mean_ndvi"].median()), 4)
    summary["share_above_threshold"] = round(
        float((successful["mean_ndvi"] >= ndvi_threshold).mean()),
        4,
    )

    best_row = successful.sort_values("mean_ndvi", ascending=False).iloc[0]
    summary["best_day"] = best_row["date"]
    summary["best_day_mean_ndvi"] = round(float(best_row["mean_ndvi"]), 4)
    return summary


def compare_ndvi_strategies(
    connection,
    all_dates,
    selected_dates,
    output_dir=NDVI_OUTPUT_DIR,
    ndvi_threshold=0.6,
):
    print_info(
        f"Preparing NDVI comparison: baseline={len(all_dates)} day(s), "
        f"metafilter={len(selected_dates)} day(s)"
    )
    baseline_results = run_strategy(
        connection,
        strategy_name="baseline_all_days",
        day_strings=all_dates,
        output_dir=output_dir,
    )
    metafilter_results = run_strategy(
        connection,
        strategy_name="metafilter_selected_days",
        day_strings=selected_dates,
        output_dir=output_dir,
    )

    all_results = pd.concat([baseline_results, metafilter_results], ignore_index=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_csv_path = output_dir / "ndvi_comparison_results.csv"
    all_results.to_csv(results_csv_path, index=False)

    baseline_summary = summarize_strategy(baseline_results, ndvi_threshold=ndvi_threshold)
    metafilter_summary = summarize_strategy(metafilter_results, ndvi_threshold=ndvi_threshold)

    comparison_summary = {
        "ndvi_threshold": ndvi_threshold,
        "baseline_all_days": baseline_summary,
        "metafilter_selected_days": metafilter_summary,
        "candidate_day_reduction": round(
            1 - (len(selected_dates) / len(all_dates)),
            4,
        ),
        "mean_ndvi_delta": (
            round(
                metafilter_summary["mean_of_mean_ndvi"]
                - baseline_summary["mean_of_mean_ndvi"],
                4,
            )
            if baseline_summary["mean_of_mean_ndvi"] is not None
            and metafilter_summary["mean_of_mean_ndvi"] is not None
            else None
        ),
        "share_above_threshold_delta": (
            round(
                metafilter_summary["share_above_threshold"]
                - baseline_summary["share_above_threshold"],
                4,
            )
            if baseline_summary["share_above_threshold"] is not None
            and metafilter_summary["share_above_threshold"] is not None
            else None
        ),
        "results_csv": str(results_csv_path),
    }

    summary_json_path = output_dir / "ndvi_comparison_summary.json"
    with open(summary_json_path, "w") as file:
        json.dump(comparison_summary, file, indent=2)

    print_info(f"Wrote NDVI comparison table: {results_csv_path}")
    print_info(f"Wrote NDVI comparison summary: {summary_json_path}")
    comparison_summary["summary_json"] = str(summary_json_path)
    return comparison_summary, all_results


def create_ndvi_comparison_plot(results_df, output_path, ndvi_threshold=0.6):
    successful = results_df[results_df["status"] == "ok"].copy()
    if successful.empty:
        raise ValueError("No successful NDVI results were available to plot.")

    successful["date"] = pd.to_datetime(successful["date"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    baseline = successful[successful["strategy"] == "baseline_all_days"].sort_values("date")
    metafilter = successful[successful["strategy"] == "metafilter_selected_days"].sort_values("date")

    figure, axis = plt.subplots(figsize=(12, 6))

    if not baseline.empty:
        axis.plot(
            baseline["date"],
            baseline["mean_ndvi"],
            color="#9AA0A6",
            linewidth=1.5,
            marker="o",
            markersize=4,
            label="All days",
        )
        axis.axhline(
            baseline["mean_ndvi"].mean(),
            color="#9AA0A6",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label="All days mean",
        )

    if not metafilter.empty:
        axis.plot(
            metafilter["date"],
            metafilter["mean_ndvi"],
            color="#2E7D32",
            linewidth=2,
            marker="o",
            markersize=6,
            label="Metafilter-selected days",
        )
        axis.axhline(
            metafilter["mean_ndvi"].mean(),
            color="#2E7D32",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label="Metafilter mean",
        )

    axis.set_title("NDVI Comparison: All Days vs Metafilter-Selected Days")
    axis.set_xlabel("Acquisition day")
    axis.set_ylabel("Mean NDVI")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print_info(f"Wrote NDVI comparison plot: {output_path}")
    return output_path


def create_era5_driver_plot(daily_metrics, rule_summaries, output_path):
    metrics = daily_metrics.copy()
    metrics["date"] = pd.to_datetime(metrics["date"])
    selected_metrics = metrics[metrics["selected"]]
    grouped_rules = {}

    for rule_summary in rule_summaries:
        grouped_rules.setdefault(rule_summary["metric_column"], []).append(rule_summary)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        len(grouped_rules),
        1,
        figsize=(13, 4 * len(grouped_rules)),
        sharex=True,
    )
    if len(grouped_rules) == 1:
        axes = [axes]

    series_colors = ["#1E4D8F", "#8A5A00", "#00695C", "#6A1B9A"]
    selected_color = "#2E7D32"
    threshold_colors = ["#B3261E", "#E65100", "#8E24AA", "#5D4037"]

    for index, (axis, (metric_column, rules_for_metric)) in enumerate(
        zip(axes, grouped_rules.items())
    ):
        series_color = series_colors[index % len(series_colors)]
        axis.plot(
            metrics["date"],
            metrics[metric_column],
            color=series_color,
            linewidth=2,
            marker="o",
            markersize=4,
            label="Daily AOI mean",
        )

        if not selected_metrics.empty:
            axis.scatter(
                selected_metrics["date"],
                selected_metrics[metric_column],
                color=selected_color,
                edgecolor="white",
                linewidth=0.6,
                s=55,
                zorder=3,
                label="Metafilter-selected day",
            )

        for rule_index, rule_summary in enumerate(rules_for_metric):
            threshold_color = threshold_colors[rule_index % len(threshold_colors)]
            axis.axhline(
                rule_summary["threshold"],
                color=threshold_color,
                linestyle="--",
                linewidth=1.5,
                label=f"Threshold {rule_summary['operator']} {rule_summary['threshold']}",
            )

        unit = rules_for_metric[0].get("unit") or metric_column
        axis.set_title(rules_for_metric[0]["name"], loc="left", fontsize=12)
        axis.set_ylabel(unit)
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")

    axes[-1].set_xlabel("Date")
    figure.suptitle("ERA5 Drivers with Metafilter-Selected Days", fontsize=16, y=0.98)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print_info(f"Wrote ERA5 driver plot: {output_path}")
    return output_path


def pick_first_unique(candidates, chosen_dates):
    for _, row in candidates.iterrows():
        if row["date"] not in chosen_dates:
            return row
    return None


def build_gallery_selection(results_df, daily_metrics):
    successful = results_df[results_df["status"] == "ok"].copy()
    baseline = successful[successful["strategy"] == "baseline_all_days"].copy()
    metafilter = successful[successful["strategy"] == "metafilter_selected_days"].copy()

    metrics = daily_metrics.copy()
    merged = successful.merge(metrics, on="date", how="left")
    baseline = merged[merged["strategy"] == "baseline_all_days"].copy()
    metafilter = merged[merged["strategy"] == "metafilter_selected_days"].copy()
    rejected = baseline[~baseline["date"].isin(metafilter["date"])].copy()

    candidates = []
    if not metafilter.empty:
        candidates.append(
            (
                "Best metafilter day",
                metafilter.sort_values("mean_ndvi", ascending=False),
            )
        )
        median_ndvi = metafilter["mean_ndvi"].median()
        candidates.append(
            (
                "Representative metafilter day",
                metafilter.assign(
                    median_distance=(metafilter["mean_ndvi"] - median_ndvi).abs()
                ).sort_values(["median_distance", "mean_ndvi"], ascending=[True, False]),
            )
        )

    if not rejected.empty:
        if "total_precip_mm" in rejected.columns:
            candidates.append(
                (
                    "Wet rejected day",
                    rejected.sort_values(
                        ["total_precip_mm", "mean_ndvi"],
                        ascending=[False, True],
                    ),
                )
            )
        if "mean_temp_c" in rejected.columns:
            candidates.append(
                (
                    "Cool rejected day",
                    rejected.sort_values(
                        ["mean_temp_c", "mean_ndvi"],
                        ascending=[True, True],
                    ),
                )
            )
        candidates.append(
            (
                "Lowest-NDVI rejected day",
                rejected.sort_values("mean_ndvi", ascending=True),
            )
        )

    candidates.extend(
        [
            ("Best baseline day", baseline.sort_values("mean_ndvi", ascending=False)),
            ("Lowest-NDVI baseline day", baseline.sort_values("mean_ndvi", ascending=True)),
        ]
    )

    selections = []
    chosen_dates = set()
    for label, candidate_frame in candidates:
        if candidate_frame.empty:
            continue
        row = pick_first_unique(candidate_frame, chosen_dates)
        if row is None:
            continue
        chosen_dates.add(row["date"])
        selections.append({"label": label, "row": row})
        if len(selections) == 4:
            break

    return selections


def load_raster_preview(raster_path):
    with rasterio.open(raster_path) as dataset:
        data = dataset.read(masked=True).astype("float32")
        if dataset.count == 1:
            return data[0]
        return np.ma.mean(data, axis=0)


def format_gallery_metrics(row, rule_summaries):
    lines = [f"Mean NDVI: {row['mean_ndvi']:.3f}"]
    for rule_summary in unique_rule_metrics(rule_summaries):
        metric_column = rule_summary["metric_column"]
        if metric_column not in row or pd.isna(row[metric_column]):
            continue
        unit = f" {rule_summary['unit']}" if rule_summary.get("unit") else ""
        lines.append(f"{rule_summary['name']}: {row[metric_column]:.2f}{unit}")
    return "\n".join(lines)


def create_ndvi_raster_gallery(
    results_df,
    daily_metrics,
    rule_summaries,
    output_path,
    color_range=(-0.2, 1.0),
):
    selections = build_gallery_selection(results_df, daily_metrics)
    if not selections:
        raise ValueError("No successful NDVI rasters were available for gallery output.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns = 2
    rows = int(np.ceil(len(selections) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(14, 5.5 * rows))
    axes = np.atleast_1d(axes).ravel()
    image_artist = None

    for axis, selection in zip(axes, selections):
        row = selection["row"]
        preview = load_raster_preview(row["raster_path"])
        image_artist = axis.imshow(
            preview,
            cmap="RdYlGn",
            vmin=color_range[0],
            vmax=color_range[1],
        )
        axis.set_title(f"{selection['label']}\n{row['date']}", loc="left", fontsize=12)
        axis.text(
            0.02,
            0.02,
            format_gallery_metrics(row, rule_summaries),
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
        axis.set_xticks([])
        axis.set_yticks([])

    for axis in axes[len(selections):]:
        axis.axis("off")

    if image_artist is not None:
        figure.subplots_adjust(right=0.9)
        colorbar_axis = figure.add_axes([0.92, 0.16, 0.018, 0.68])
        colorbar = figure.colorbar(
            image_artist,
            cax=colorbar_axis,
        )
        colorbar.set_label("NDVI")

    figure.suptitle("NDVI Raster Gallery: Selected and Rejected Days", fontsize=16, y=0.98)
    figure.text(
        0.5,
        0.01,
        "All panels use the same NDVI color scale.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=[0, 0.03, 0.9, 0.96])
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print_info(f"Wrote NDVI raster gallery: {output_path}")
    return output_path
