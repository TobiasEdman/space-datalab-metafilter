"""CDS retrieval entry points.

Two retrievals are supported:

* `download_era5_land()` — `reanalysis-era5-land`, the high-res surface
  product. Default source for temperature, precipitation, soil moisture,
  snow, skin temperature, radiation.
* `download_era5_cloud()` — `reanalysis-era5-single-levels`, fetched only
  when the active filter profile references total/low cloud cover.
  ERA5-Land does not include `tcc`/`lcc`, so it has to come from the
  single-levels product.
"""
from pathlib import Path

from utils.config import AREA, OUTPUT_DIR


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


if __name__ == "__main__":
    download_era5_land()
