from metafilter.core import _netcdf_engine


def test_netcdf3_signature_selects_scipy(tmp_path):
    path = tmp_path / "classic.nc"
    path.write_bytes(b"CDF\x02placeholder")

    assert _netcdf_engine(path) == "scipy"


def test_hdf5_signature_selects_h5netcdf(tmp_path):
    path = tmp_path / "netcdf4.nc"
    path.write_bytes(b"\x89HDF\r\n\x1a\npayload")

    assert _netcdf_engine(path) == "h5netcdf"


def test_unknown_signature_uses_xarray_default(tmp_path):
    path = tmp_path / "unknown.nc"
    path.write_bytes(b"not-netcdf")

    assert _netcdf_engine(path) is None


def test_partial_netcdf_signature_is_not_accepted(tmp_path):
    path = tmp_path / "partial.nc"
    path.write_bytes(b"CDFpayload")

    assert _netcdf_engine(path) is None


def test_cdf5_falls_through_instead_of_selecting_scipy(tmp_path):
    path = tmp_path / "cdf5.nc"
    path.write_bytes(b"CDF\x05payload")

    assert _netcdf_engine(path) is None


def test_partial_hdf5_signature_is_not_accepted(tmp_path):
    path = tmp_path / "partial.h5"
    path.write_bytes(b"\x89HDF\r\n\x1apayload")

    assert _netcdf_engine(path) is None
