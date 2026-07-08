from pathlib import Path

from scripts.compare_ndvi import (
    authenticate,
    compare_ndvi_strategies,
    create_era5_driver_plot,
    create_ndvi_comparison_plot,
    create_ndvi_raster_gallery,
)
from scripts.process_era5 import (
    MetafilterError,
    format_rule_brief,
    load_metafilter_parameters,
    normalize_metafilter_rules,
    process_era5_data,
    save_daily_metrics,
)
from utils.config import AREA, NDVI_OUTPUT_DIR, eo_service_url, password, username


def print_info(message):
    print(f"INFO {message}")


def main():
    # Fail fast: the run needs openEO credentials for the NDVI stage, so
    # reject a misconfigured environment before the ERA5 processing runs.
    if not username or not password:
        print(
            "ERROR OpenEO credentials must be set via OPENEO_USERNAME and "
            "OPENEO_PASSWORD."
        )
        return 1

    output_dir = Path(NDVI_OUTPUT_DIR)
    daily_metrics_path = output_dir / "era5_daily_metrics.csv"
    metafilter_path = "filters/metafilter.json"
    metafilter_params = load_metafilter_parameters(metafilter_path)
    metafilter_rules = normalize_metafilter_rules(metafilter_params)

    print_info("Starting metafilter NDVI comparison run")
    print_info(f"Metafilter file: {metafilter_path}")
    for rule in metafilter_rules:
        print_info(f"Metafilter rule: {format_rule_brief(rule)}")
    print_info(
        "AOI: "
        f"west={AREA['west']}, east={AREA['east']}, "
        f"south={AREA['south']}, north={AREA['north']}"
    )
    print_info("Processing ERA5 daily metrics")

    try:
        era5_results = process_era5_data(
            "data/era5/era5_land_july_2024.nc",
            metafilter_params,
        )
    except MetafilterError as exc:
        if getattr(exc, "daily_metrics", None) is not None:
            save_daily_metrics(exc.daily_metrics, daily_metrics_path)
            print(f"{exc}\nSaved ERA5 diagnostics to: {daily_metrics_path}")
        else:
            print(exc)
        return 1

    save_daily_metrics(era5_results["daily_metrics"], daily_metrics_path)
    print_info(
        f"ERA5 processing complete: {len(era5_results['all_dates'])} candidate day(s), "
        f"{len(era5_results['selected_dates'])} metafilter-selected day(s)"
    )
    print_info(f"Saved ERA5 diagnostics: {daily_metrics_path}")

    connection = authenticate(username, password, eo_service_url)
    comparison_summary, comparison_results = compare_ndvi_strategies(
        connection=connection,
        all_dates=era5_results["all_dates"],
        selected_dates=era5_results["selected_dates"],
        output_dir=output_dir,
    )

    plot_path = create_ndvi_comparison_plot(
        comparison_results,
        output_path=output_dir / "ndvi_comparison_plot.png",
        ndvi_threshold=comparison_summary["ndvi_threshold"],
    )
    driver_plot_path = create_era5_driver_plot(
        era5_results["daily_metrics"],
        era5_results["rule_summaries"],
        output_path=output_dir / "era5_driver_plot.png",
    )
    gallery_path = create_ndvi_raster_gallery(
        comparison_results,
        era5_results["daily_metrics"],
        era5_results["rule_summaries"],
        output_path=output_dir / "ndvi_raster_gallery.png",
    )

    print_info(
        f"Comparison summary: mean_ndvi_delta={comparison_summary['mean_ndvi_delta']}, "
        f"share_above_threshold_delta={comparison_summary['share_above_threshold_delta']}"
    )
    print(f"Generated NDVI comparison plot: {plot_path}")
    print(f"Generated ERA5 driver plot: {driver_plot_path}")
    print(f"Generated NDVI raster gallery: {gallery_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
