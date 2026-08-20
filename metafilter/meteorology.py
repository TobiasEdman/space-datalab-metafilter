"""Public Open-Meteo-backed daily meteorology interface."""

from __future__ import annotations

import calendar
import contextlib
import hashlib
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import xarray as xr

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    fcntl = None
    import msvcrt

from .open_meteo import (
    fetch_open_meteo_archive,
    open_meteo_json_to_dataset,
)
from .core import calculate_daily_metrics


_CACHE_LOCKS: dict[Path, threading.Lock] = {}
_CACHE_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True)
class DailyMeteorology:
    """Daily metrics plus provenance for one requested period."""

    frame: pd.DataFrame
    source: str
    bbox_wgs84: dict[str, float]
    date_start: str
    date_end: str
    cache_paths: tuple[str, ...]


def fetch_daily_meteorology(
    *,
    bbox_wgs84: dict[str, float],
    date_start: str,
    date_end: str,
    backend: str = "open-meteo",
    cache_dir: str | Path = ".cache/metafilter",
    timeout: int = 60,
) -> DailyMeteorology:
    """Fetch and aggregate daily meteorology for a WGS84 bbox.

    The requested start should include the desired lookback buffer. The result
    is clipped to the exact inclusive date range after aggregation.
    """
    if backend != "open-meteo":
        raise ValueError("the public POC API currently supports backend='open-meteo'")
    start = pd.Timestamp(date_start).normalize()
    end = pd.Timestamp(date_end).normalize()
    if start > end:
        raise ValueError("date_start must not be after date_end")

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    paths = []
    for year, month in _months_between(start, end):
        cache_path = cache_root / _cache_name(bbox_wgs84, year, month)
        with _cache_lock(cache_path), _file_lock(cache_path):
            if not _valid_cache(cache_path, year, month):
                hourly, lat, lon = fetch_open_meteo_archive(
                    year,
                    month,
                    area=bbox_wgs84,
                    timeout=timeout,
                )
                dataset = open_meteo_json_to_dataset(hourly, lat, lon)
                _write_cache_atomic(dataset, cache_path, year, month)
        paths.append(str(cache_path))

    frame = calculate_daily_metrics(paths, area=bbox_wgs84)
    dates = pd.to_datetime(frame["date"])
    frame = frame.loc[(dates >= start) & (dates <= end)].reset_index(drop=True)
    return DailyMeteorology(
        frame=frame,
        source=backend,
        bbox_wgs84=dict(bbox_wgs84),
        date_start=start.strftime("%Y-%m-%d"),
        date_end=end.strftime("%Y-%m-%d"),
        cache_paths=tuple(paths),
    )


def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[int, int]]:
    current = start.replace(day=1)
    months = []
    while current <= end:
        months.append((int(current.year), int(current.month)))
        days = calendar.monthrange(current.year, current.month)[1]
        current += pd.Timedelta(days=days)
    return months


def _cache_name(bbox: dict[str, float], year: int, month: int) -> str:
    bbox_key = ",".join(f"{bbox[key]:.6f}" for key in ("west", "south", "east", "north"))
    digest = hashlib.sha256(bbox_key.encode("utf-8")).hexdigest()[:12]
    return f"openmeteo_{digest}_{year}_{month:02d}.nc"


def _cache_lock(path: Path) -> threading.Lock:
    resolved = path.resolve()
    with _CACHE_LOCKS_GUARD:
        return _CACHE_LOCKS.setdefault(resolved, threading.Lock())


def _valid_cache(path: Path, year: int, month: int) -> bool:
    if not path.is_file():
        return False
    try:
        with xr.open_dataset(path, engine="scipy") as dataset:
            required = {"t2m", "tp", "ssrd", "swvl1"}
            if not required <= set(dataset.data_vars):
                return False
            times = pd.DatetimeIndex(dataset["time"].values)
            expected_times = pd.date_range(
                start=pd.Timestamp(year=year, month=month, day=1),
                periods=calendar.monthrange(year, month)[1] * 24,
                freq="h",
            )
            return (
                len(times) == len(expected_times)
                and times.equals(expected_times)
            )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _write_cache_atomic(dataset: xr.Dataset, path: Path, year: int, month: int) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        dataset.to_netcdf(
            temporary_path,
            engine="scipy",
            encoding={
                "time": {
                    "units": "seconds since 1970-01-01",
                    "dtype": "float64",
                }
            },
        )
        if not _valid_cache(temporary_path, year, month):
            raise ValueError("generated Open-Meteo cache is invalid")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


@contextlib.contextmanager
def _file_lock(path: Path):
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+b") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        else:  # pragma: no cover - exercised on Windows
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            else:  # pragma: no cover - exercised on Windows
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
