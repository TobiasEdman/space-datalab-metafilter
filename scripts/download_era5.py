"""Download one month of ERA5-Land data from the Copernicus Climate Data Store.

This is the legacy single-purpose retrieval script (`python -m
scripts.download_era5`). When followed by `python -m scripts.process_era5`
it produces the inputs for the metafilter NDVI comparison run in `main.py`.

Known CDS-side quirk handled here
---------------------------------
The CDS API for ERA5-Land sometimes returns a *zip archive* containing the
NetCDF file rather than the NetCDF itself, even when ``data_format: 'netcdf'``
is requested. The zip is written to the path you asked for with the
filename you chose — including the ``.nc`` extension — so a naive
downstream ``xarray.open_dataset(...)`` raises a misleading error
("did not find a match in any of xarray's currently installed IO
backends"). After every successful retrieve we therefore check the magic
bytes and, if they're a zip header, extract the inner NetCDF in place.

The behaviour was first reported externally by a user who solved their
own breakage by manually ``unzip``-ping the downloaded file. Confirmed
locally by reproducing the exact xarray error against a zip-with-.nc
extension fixture (no live CDS retrieve needed for the reproduction).
"""
import shutil
import zipfile
from pathlib import Path

from utils.config import AREA, OUTPUT_DIR

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
        },
        out_path,
    )

    # CDS sometimes returns a zip wrapping the NetCDF — normalise here so
    # downstream xarray.open_dataset() sees an actual NetCDF.
    _maybe_extract_zipped_netcdf(out_path)
    return out_path


if __name__ == "__main__":
    download_era5_land()
