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

eo_service_url = os.getenv("OPENEO_SERVICE_URL", "https://openeo.digitalearth.se")
username = os.getenv("OPENEO_USERNAME")
password = os.getenv("OPENEO_PASSWORD")
