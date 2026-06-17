"""Download one month of ERA5-Land data from the Copernicus Climate Data Store.

This is the legacy single-purpose retrieval script (`python -m
scripts.download_era5`). When followed by `python -m scripts.process_era5`
it produces the inputs for the metafilter NDVI comparison run in `main.py`.

CDS-Beta defaults to packaging ERA5 retrieves as a zip archive containing
the requested NetCDF, even when ``data_format: 'netcdf'`` is set. The zip
is written to the user-supplied path with the user-supplied filename —
including the ``.nc`` extension — so a naive downstream
``xarray.open_dataset(...)`` raises the misleading error "did not find a
match in any of xarray's currently installed IO backends". The fix is to
explicitly request ``download_format: 'unarchived'`` in the body, which
makes CDS skip the zip wrapping and return the NetCDF directly.
"""
from pathlib import Path

from utils.config import AREA, OUTPUT_DIR


def download_era5_land():
    import cdsapi

    out_path = f"{OUTPUT_DIR}/era5/era5_land_july_2024.nc"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-land",
        {
            "variable": ["2m_temperature", "total_precipitation"],
            "year": "2024",
            "month": "08",
            "day": [f"{day:02d}" for day in range(1, 32)],
            "time": [f"{hour:02d}:00" for hour in range(24)],
            # CDS expects [north, west, south, east].
            "area": [AREA["north"], AREA["west"], AREA["south"], AREA["east"]],
            "data_format": "netcdf",
            # Without this, CDS-Beta wraps the NetCDF in a zip archive and
            # writes it under the user-supplied .nc filename — which then
            # confuses every downstream xarray.open_dataset() call.
            "download_format": "unarchived",
        },
        out_path,
    )
    return out_path


if __name__ == "__main__":
    download_era5_land()
