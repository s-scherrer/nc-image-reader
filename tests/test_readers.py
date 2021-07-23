import numpy as np
import pandas as pd
import pytest
import time
import xarray as xr

from nc_image_reader.readers import (
    DirectoryImageReader,
    XarrayImageReader,
    XarrayTSReader,
)

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

    img = reader.read(expected_timestamps[-1])
    true = xr.open_dataset(
        test_data_path / "lis_noah" / "201704" / "LIS_HIST_201704011800.d01.nc"
    )["SoilMoist_tavg"].isel(SoilMoist_profiles=0)
    np.testing.assert_allclose(img.data["SoilMoist_tavg"], true.values.ravel())
    true.attrs == img.metadata["SoilMoist_tavg"]


###############################################################################
# DirectoryImageReader
###############################################################################

# Optional features to test for DirectoryImageReader:
# - [X] var_dim_selection
#   - [X] with var_dim_selection: default_directory_reader
#   - [-] without var_dim_selection: covered in XarrayImageReader tests
# - [X] fmt: default_directory_reader
# - [X] pattern: default_directory_reader
# - [X] time_regex_pattern: test_time_regex
# - [-] timename, latname, lonname: covered in XarrayImageReader tests
# - [X] latdim, londim: default_directory_reader
# - [-] locdim: covered in XarrayImageReader tests
# - [-] landmask: covered in XarrayImageReader tests
# - [-] bbox: covered in XarrayImageReader tests
# - [-] cellsize None: covered in XarrayImageReader tests


def test_directory_reader_setup():

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


def test_time_regex():
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


###############################################################################
# XarrayImageReader
###############################################################################

# Optional features to test for DirectoryImageReader:
# - [X] var_dim_selection
#   - [-] with var_dim_selection: covered in DirectoryImageReader tests
#   - [X] without var_dim_selection: test_xarray_reader_basic
# - [X] timename, latname, lonname: test_nonstandard_names
# - [X] latdim, londim: covered in DirectoryImageReader tests
# - [X] locdim: test_locdim
# - [X] bbox, cellsize, landmask: test_landmask, test_bbox_cellsize


def test_xarray_reader_basic(default_xarray_reader):
    validate_reader(default_xarray_reader)


def test_nonstandard_names(test_dataset):
    ds = test_dataset.rename({"time": "tim", "lat": "la", "lon": "lo"})
    reader = XarrayImageReader(
        ds, "X", timename="tim", latname="la", lonname="lo"
    )
    block = reader.read_block()
    assert block.shape == (100, 10, 20)


def test_locdim(test_loc_dataset):
    reader = XarrayImageReader(test_loc_dataset, "X", locdim="location")
    block = reader.read_block()
    assert block.shape == (100, 200)


def test_bbox_landmask_cellsize(cmip_ds):
    """
    Tests the bounding box feature
    """
    num_gpis = cmip_ds["mrsos"].isel(time=0).size

    # normal reader without bbox or landmask
    reader = XarrayImageReader(cmip_ds, "mrsos", cellsize=5.0)
    assert len(reader.grid.activegpis) == num_gpis
    assert len(np.unique(reader.grid.activearrcell)) == 2592
    block = reader.read_block()
    np.testing.assert_allclose(block.values, cmip_ds.mrsos.values)

    # now with bbox using only Europe
    min_lon = -11
    min_lat = 34
    max_lon = 48
    max_lat = 72
    bbox = [min_lon, min_lat, max_lon, max_lat]
    reader = XarrayImageReader(cmip_ds, "mrsos", bbox=bbox, cellsize=5.0)
    num_gpis_europe = len(reader.grid.activegpis)
    assert num_gpis_europe < num_gpis
    assert len(np.unique(reader.grid.activearrcell)) == 117
    assert not np.any(reader.grid.arrlon < min_lon)
    assert not np.any(reader.grid.arrlat < min_lat)
    assert not np.any(reader.grid.arrlon > max_lon)
    assert not np.any(reader.grid.arrlat > max_lat)
    block = reader.read_block()
    # the sel-slice notation only works with regular grids
    europe_mrsos = cmip_ds.mrsos.sel(
        lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon)
    )
    np.testing.assert_allclose(block.values, europe_mrsos.values)

    # now additionally using a landmask
    landmask = ~np.isnan(cmip_ds.mrsos.isel(time=0))
    reader = XarrayImageReader(
        cmip_ds, "mrsos", bbox=bbox, landmask=landmask, cellsize=5.0
    )
    assert len(reader.grid.activegpis) < num_gpis_europe
    assert len(np.unique(reader.grid.activearrcell)) == 88
    assert not np.any(reader.grid.arrlon < min_lon)
    assert not np.any(reader.grid.arrlat < min_lat)
    assert not np.any(reader.grid.arrlon > max_lon)
    assert not np.any(reader.grid.arrlat > max_lat)
    block = reader.read_block()
    # the sel-slice notation only works with regular grids
    europe_mrsos = cmip_ds.mrsos.sel(
        lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon)
    )
    np.testing.assert_allclose(block.values, europe_mrsos.values)

    # with landmask as variable name
    ds = cmip_ds
    ds["landmask"] = landmask
    reader = XarrayImageReader(
        ds, "mrsos", bbox=bbox, landmask=landmask, cellsize=5.0
    )
    assert len(reader.grid.activegpis) < num_gpis_europe
    assert len(np.unique(reader.grid.activearrcell)) == 88
    assert not np.any(reader.grid.arrlon < min_lon)
    assert not np.any(reader.grid.arrlat < min_lat)
    assert not np.any(reader.grid.arrlon > max_lon)
    assert not np.any(reader.grid.arrlat > max_lat)
    new_block = reader.read_block()
    np.testing.assert_allclose(block.values, new_block.values)


###############################################################################
# XarrayTSReader
###############################################################################


def test_xarray_ts_reader(test_dataset):
    reader = XarrayTSReader(test_dataset, "X")
    _, lons, lats = reader.grid.get_grid_points()
    for lon, lat in zip(lons, lats):
        ts = reader.read(lon, lat)
        ref = test_dataset.X.sel(lat=lat, lon=lon)
        assert np.all(ts == ref)


def test_xarray_ts_reader_locdim(test_loc_dataset):
    reader = XarrayTSReader(test_loc_dataset, "X", locdim="location")
    gpis, _, _ = reader.grid.get_grid_points()
    for gpi in gpis:
        ts = reader.read(gpi)
        ref = test_loc_dataset.X.isel(location=gpi)
        assert np.all(ts == ref)
