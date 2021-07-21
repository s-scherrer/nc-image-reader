import numpy as np
import pandas as pd
import pytest
import time
import xarray as xr


from nc_image_reader.readers import DirectoryImageReader, XarrayImageReader

# this is defined in conftest.py
from pytest import test_data_path


def validate_reader(reader):

    expected_timestamps = pd.date_range(
        "2017-03-31 00:00", periods=8, freq="6H"
    ).to_pydatetime()
    assert len(reader.timestamps) == 8
    assert np.all(list(reader.timestamps) == expected_timestamps)

    img = reader.read(expected_timestamps[0])
    true = xr.open_dataset(
        test_data_path / "lis_noah" / "201703" / "LIS_HIST_201703310000.d01.nc"
    )["SoilMoist_tavg"].isel(SoilMoist_profiles=0)
    np.testing.assert_allclose(img.data["SoilMoist_tavg"], true.values.ravel())
    true.attrs == img.metadata["SoilMoist_tavg"]


@pytest.fixture
def default_directory_reader():
    pattern = "LIS_HIST*.nc"
    fmt = "LIS_HIST_%Y%m%d%H%M.d01.nc"
    reader = DirectoryImageReader(
        test_data_path / "lis_noah",
        "SoilMoist_tavg",
        fmt=fmt,
        pattern=pattern,
        latdim="north_south",
        londim="east_west",
        var_dim_selection={"SoilMoist_profiles": 0},
    )
    return reader


def test_setup():

    # test "normal procedure", i.e. with given fmt
    pattern = "LIS_HIST*.nc"
    fmt = "LIS_HIST_%Y%m%d%H%M.d01.nc"

    # the LIS_HIST files have dimensions north_south and east_west instead of
    # lat/lon
    start = time.time()
    reader = DirectoryImageReader(
        test_data_path / "lis_noah",
        "SoilMoist_tavg",
        fmt=fmt,
        pattern=pattern,
        latdim="north_south",
        londim="east_west",
        var_dim_selection={"SoilMoist_profiles": 0},
    )
    runtime = time.time() - start
    print(f"Setup time with fmt string: {runtime:.2e}")
    validate_reader(reader)

    # test without fmt, requires opening all files
    start = time.time()
    reader = DirectoryImageReader(
        test_data_path / "lis_noah",
        "SoilMoist_tavg",
        pattern=pattern,
        latdim="north_south",
        londim="east_west",
        var_dim_selection={"SoilMoist_profiles": 0},
    )
    runtime2 = time.time() - start
    print(f"Setup time without fmt string: {runtime2:.2e}")
    validate_reader(reader)

    assert runtime < runtime2

    # test with using a regex for the time string
    pattern = "LIS_HIST*.nc"
    fmt = "%Y%m%d%H%M"
    time_regex_pattern = r"LIS_HIST_(\d+)\..*\.nc"
    reader = DirectoryImageReader(
        test_data_path / "lis_noah",
        "SoilMoist_tavg",
        fmt=fmt,
        pattern=pattern,
        time_regex_pattern=time_regex_pattern,
        latdim="north_south",
        londim="east_west",
        var_dim_selection={"SoilMoist_profiles": 0},
    )
    validate_reader(reader)


def test_read_block(default_directory_reader):
    block = default_directory_reader.read_block()
    assert block.shape == (8, 22, 28)

    reader = XarrayImageReader(
        block.to_dataset(name="SoilMoist_tavg"), "SoilMoist_tavg"
    )
    validate_reader(reader)

    start_date = next(iter(default_directory_reader.timestamps))
    block = default_directory_reader.read_block(
        start=start_date, end=start_date
    )
    assert block.shape == (1, 22, 28)
