from utils.config import AREA, OUTPUT_DIR

def download_era5_land():
    import cdsapi

    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-land',
        {
            'variable': ['2m_temperature', 'total_precipitation'],
            'year': '2024',
            'month': '08',
            'day': [f'{day:02d}' for day in range(1, 32)],
            'time': [f'{hour:02d}:00' for hour in range(24)],
            # CDS expects [north, west, south, east].
            'area': [AREA["north"], AREA["west"], AREA["south"], AREA["east"]],
            'data_format': 'netcdf',
            # Without this, CDS-Beta wraps the NetCDF in a zip archive
            # and writes it under the user-supplied .nc filename, which
            # then makes xarray.open_dataset() fail with the misleading
            # "did not find a match in any of xarray's currently installed
            # IO backends" error on what is actually a zip file.
            'download_format': 'unarchived',
        },
        f"{OUTPUT_DIR}/era5/era5_land_july_2024.nc"
    )

if __name__ == "__main__":
    download_era5_land()
