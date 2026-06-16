"""Tests for `_maybe_extract_zipped_netcdf` — the CDS-zip-quirk normaliser.

CDS for ERA5-Land has a behaviour where it sometimes returns a zip archive
wrapping the requested NetCDF, written to disk under the .nc filename you
asked for. xarray's auto-detect can't open such a file and raises the
misleading "did not find a match in any of xarray's currently installed IO
backends" error. download_era5.py now calls _maybe_extract_zipped_netcdf
after every retrieve to normalise that into a real NetCDF before any
downstream code touches it.
"""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from scripts.download_era5 import _maybe_extract_zipped_netcdf


def _write_minimal_netcdf3(path):
    """Write a minimal but valid NetCDF (via the scipy engine used elsewhere
    in the test suite) so we can both feed it to _maybe_extract_zipped_netcdf
    as a no-op input and as the payload inside a zip we build."""
    ds = xr.Dataset(
        data_vars={"x": (("time",), np.arange(3, dtype=np.float32))},
        coords={"time": np.arange(3)},
    )
    ds.to_netcdf(path, engine="scipy")


def _zip_with_nc_extension(inner_nc, out_path):
    """Build a zip archive at `out_path` (with whatever extension it has —
    typically `.nc` to simulate the CDS quirk) containing a single `.nc`
    file copied from `inner_nc`."""
    with zipfile.ZipFile(out_path, "w") as zf:
        zf.write(inner_nc, arcname=Path(inner_nc).name)


def test_no_op_on_plain_netcdf(tmp_path):
    """A regular NetCDF passes through bit-identically — no rewrite, no
    backup file, no extension change."""
    nc = tmp_path / "real.nc"
    _write_minimal_netcdf3(nc)
    before = hashlib.sha256(nc.read_bytes()).hexdigest()

    returned = _maybe_extract_zipped_netcdf(nc)

    assert returned == nc
    assert hashlib.sha256(nc.read_bytes()).hexdigest() == before


def test_extracts_zipped_netcdf_in_place(tmp_path):
    """The CDS quirk shape: zip-with-.nc-extension. After the call, the file
    at the same path is the inner NetCDF, openable by xarray."""
    inner = tmp_path / "_inner.nc"
    _write_minimal_netcdf3(inner)

    target = tmp_path / "era5_land.nc"
    _zip_with_nc_extension(inner, target)
    # Sanity: pre-call, xarray cannot open the zip-with-.nc-extension
    with pytest.raises(ValueError, match="did not find a match in any"):
        xr.open_dataset(target)

    _maybe_extract_zipped_netcdf(target)

    # Post-call: file at the original path is now a real NetCDF
    ds = xr.open_dataset(target)
    assert "x" in ds.data_vars
    assert ds.sizes["time"] == 3


def test_zip_with_no_nc_member_raises(tmp_path):
    """Defensive: if CDS ever returns a zip containing no .nc, fail loudly
    rather than silently leave the user with a broken file."""
    target = tmp_path / "weird.nc"
    with zipfile.ZipFile(target, "w") as zf:
        zf.writestr("readme.txt", "no netcdf here")

    with pytest.raises(RuntimeError, match="contains no .nc files"):
        _maybe_extract_zipped_netcdf(target)


def test_zip_with_multiple_nc_members_raises(tmp_path):
    """Defensive: ambiguous zip layouts are explicit errors, not heuristic
    guesses."""
    inner_a = tmp_path / "_a.nc"
    inner_b = tmp_path / "_b.nc"
    _write_minimal_netcdf3(inner_a)
    _write_minimal_netcdf3(inner_b)

    target = tmp_path / "multi.nc"
    with zipfile.ZipFile(target, "w") as zf:
        zf.write(inner_a, arcname="part_a.nc")
        zf.write(inner_b, arcname="part_b.nc")

    with pytest.raises(RuntimeError, match="multiple .nc files"):
        _maybe_extract_zipped_netcdf(target)


def test_extraction_is_atomic_no_tmp_left_behind(tmp_path):
    """The replace-from-tmp dance shouldn't leave a `.nc.tmp` file behind on
    success."""
    inner = tmp_path / "_inner.nc"
    _write_minimal_netcdf3(inner)

    target = tmp_path / "era5_land.nc"
    _zip_with_nc_extension(inner, target)

    _maybe_extract_zipped_netcdf(target)

    leftover = tmp_path / "era5_land.nc.tmp"
    assert not leftover.exists()
