from pathlib import Path
from typing import Any

import cdsapi  # type: ignore[import-untyped]
import pandas as pd
import pytest

from scripts.download_era5 import download_era5_land
from scripts.process_era5 import calculate_daily_metrics
from tests.synthetic_era5 import era5_land_dataset
from utils.config import AREA, OUTPUT_DIR

ERA5_DIR = Path(OUTPUT_DIR) / "era5"


class FakeCdsClient:
    """Stands in for cdsapi.Client: records requests, writes matching files.

    Like the real client, it opens the target as given and creates no
    directories, so a missing directory raises FileNotFoundError.
    """

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def retrieve(self, name: str, request: dict[str, Any], target: str) -> None:
        self.requests.append((name, request))
        times = pd.DatetimeIndex(
            sorted(
                pd.Timestamp(f"{request['year']}-{request['month']}-{day} {hour}")
                for day in request["day"]
                for hour in request["time"]
            )
        )
        days = (times[-1].normalize() - times[0].normalize()).days + 1
        north, west, south, east = request["area"]
        dataset = era5_land_dataset(
            times, [15.0] * days, latitudes=(north, south), longitudes=(west, east)
        )
        open(target, "wb").close()
        dataset.to_netcdf(target)


@pytest.fixture
def cds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeCdsClient:
    client = FakeCdsClient()
    monkeypatch.setattr(cdsapi, "Client", lambda: client)
    monkeypatch.chdir(tmp_path)
    return client


@pytest.fixture
def era5_dir(cds: FakeCdsClient) -> Path:
    ERA5_DIR.mkdir(parents=True)
    return ERA5_DIR


def test_request_asks_for_unarchived_netcdf_over_the_area(
    cds: FakeCdsClient, era5_dir: Path
) -> None:
    download_era5_land()

    name, request = cds.requests[0]
    assert name == "reanalysis-era5-land"
    assert request["data_format"] == "netcdf"
    assert request["download_format"] == "unarchived"
    assert request["area"] == [AREA["north"], AREA["west"], AREA["south"], AREA["east"]]
    assert {"2m_temperature", "total_precipitation"} <= set(request["variable"])
    assert request["time"] == [f"{hour:02d}:00" for hour in range(24)]


def test_download_writes_a_readable_file_under_output_dir(
    cds: FakeCdsClient, era5_dir: Path
) -> None:
    download_era5_land()

    [path] = era5_dir.glob("*.nc")
    _, request = cds.requests[0]
    assert len(calculate_daily_metrics(path, area=AREA)) == len(request["day"])


@pytest.mark.xfail(
    raises=FileNotFoundError,
    strict=True,
    reason="download_era5_land does not create data/era5 before writing to it",
)
def test_download_into_a_fresh_checkout(cds: FakeCdsClient) -> None:
    download_era5_land()

    assert list(ERA5_DIR.glob("*.nc"))
