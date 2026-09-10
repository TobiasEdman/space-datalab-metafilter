"""Compatibility shim for installers with pre-PEP 621 setuptools."""

from setuptools import find_packages, setup


setup(
    name="space-datalab-metafilter",
    version="0.1.0",
    description="Meteorological filtering and analog-date selection",
    python_requires=">=3.9",
    packages=find_packages(include=("metafilter*",)),
    install_requires=(
        "numpy>=1.24",
        "pandas>=2.0",
        "requests>=2.33.1,<3",
        "h5netcdf>=1.4,<2",
        "cftime>=1.6,<2",
        "scipy>=1.10",
        "xarray>=2023.1",
    ),
    entry_points={
        "console_scripts": (
            "metafilter-process-era5=metafilter.cli:process_era5_main",
            "metafilter-download-open-meteo=metafilter.cli:download_open_meteo_main",
        )
    },
)
