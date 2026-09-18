import os

from dotenv import load_dotenv

load_dotenv()

AREA = {
    "west": 18.0,
    "east": 18.2,
    "south": 59.2,
    "north": 59.4,
}

# Output directories
OUTPUT_DIR = "data"
NDVI_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "ndvi_comparison")

# openEO credentials are read here but NOT validated at import time —
# the ERA5 pipeline (`scripts/download_era5`, `scripts/process_era5`) has no
# need for them, and forcing a check here meant those scripts couldn't even
# load without OPENEO_USERNAME / OPENEO_PASSWORD set. Validation now happens
# in scripts/compare_ndvi.authenticate() at the moment of actual use, so the
# error surfaces where it's relevant.
eo_service_url = os.getenv("OPENEO_SERVICE_URL", "https://openeo.digitalearth.se")
username = os.getenv("OPENEO_USERNAME")
password = os.getenv("OPENEO_PASSWORD")
