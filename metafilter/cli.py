"""Installed command-line entry points."""

import argparse
from pathlib import Path

from .config import AREA
from .core import MetafilterError, load_metafilter_parameters, process_era5_data
from .open_meteo import download_open_meteo_land


def _area_from_bbox(values, parser):
    if values is None:
        return None
    west, south, east, north = values
    if not (-180 <= west < east <= 180):
        parser.error("--bbox requires -180 <= WEST < EAST <= 180")
    if not (-90 <= south < north <= 90):
        parser.error("--bbox requires -90 <= SOUTH < NORTH <= 90")
    return {"west": west, "south": south, "east": east, "north": north}


def process_era5_main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Apply a metafilter profile to ERA5 NetCDF data."
    )
    parser.add_argument(
        "input",
        help="ERA5-Land or Open-Meteo-compatible NetCDF file",
    )
    parser.add_argument(
        "--filter",
        required=True,
        help="Metafilter JSON profile",
    )
    parser.add_argument(
        "--cloud-file",
        help="Optional ERA5 single-levels cloud-cover NetCDF file",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        help="Optional CSV path for daily metrics and selection results",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Processing AOI in WGS84; defaults to the configured area",
    )
    args = parser.parse_args(argv)
    area = _area_from_bbox(args.bbox, parser) or AREA

    try:
        params = load_metafilter_parameters(args.filter)
        results = process_era5_data(
            args.input,
            params,
            area=area,
            cloud_file_path=args.cloud_file,
        )
        if args.metrics_output:
            args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
            results["daily_metrics"].to_csv(args.metrics_output, index=False)
    except (MetafilterError, OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(results["selected_dates"])


def download_open_meteo_main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Download one month of ERA5-compatible data from Open-Meteo."
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--month", type=int, choices=range(1, 13), default=8)
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="AOI in WGS84; the centroid is snapped to the ERA5 grid",
    )
    parser.add_argument(
        "--variables",
        help="Comma-separated Open-Meteo hourly variables",
    )
    args = parser.parse_args(argv)

    variables = None
    if args.variables:
        variables = [value.strip() for value in args.variables.split(",") if value.strip()]
    area = _area_from_bbox(args.bbox, parser)
    try:
        path = download_open_meteo_land(
            year=args.year,
            month=args.month,
            area=area,
            variables=variables,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(path)
