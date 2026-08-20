"""Installed command-line compatibility entry points."""

from .core import MetafilterError, load_metafilter_parameters, process_era5_data
from .open_meteo import download_open_meteo_land


def process_era5_main() -> None:
    params = load_metafilter_parameters("filters/metafilter.json")
    try:
        results = process_era5_data("data/era5/era5_land_july_2024.nc", params)
    except MetafilterError as exc:
        print(exc)
        raise SystemExit(1) from exc
    print(results["selected_dates"])


def download_open_meteo_main() -> None:
    download_open_meteo_land()
