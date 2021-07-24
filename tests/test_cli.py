import numpy as np
from pathlib import Path
import pytest
import shutil
import h5netcdf
import xarray as xr

from nc_image_reader.readers import GriddedNcOrthoMultiTs, XarrayTSReader
from nc_image_reader.cli import repurpose, transpose

# this is defined in conftest.py
from pytest import test_data_path


@pytest.fixture
def cli_args_lis(test_output_path):
    input_path = test_data_path / "lis_noah"
    output_path = test_output_path
    return [
        str(input_path),
        str(output_path),
        "2017-03-31",
        "2017-04-02",
        *("--parameter", "SoilMoist_tavg"),
        *("--pattern", "LIS_HIST*.nc"),
        *("--time_regex_pattern", r"LIS_HIST_(\d+)\..*\.nc"),
        *("--time_fmt", "%Y%m%d%H%M"),
        *("--latdim", "north_south"),
        *("--londim", "east_west"),
        *("--var_dim_selection", "SoilMoist_profiles:0"),
    ]


def test_transpose_lis(cli_args_lis, lis_noah_stacked):
    args = cli_args_lis + ["--chunks", "22", "14", "-1"]
    args[1] = args[1] + "/lis_noah_transposed.nc"
    outpath = Path(args[1])
    transpose(args)
    ref = xr.open_dataset(
        test_data_path / "lis_noah" / "201703" / "LIS_HIST_201703310000.d01.nc"
    )
    with h5netcdf.File(outpath, "r", decode_vlen_strings=False) as ds:
        assert ds["SoilMoist_tavg"].dimensions == ("lat", "lon", "time")
        assert ds["SoilMoist_tavg"].shape == (22, 28, 8)
        assert ds["SoilMoist_tavg"].chunks == (22, 14, 8)
        np.testing.assert_allclose(
            ds["SoilMoist_tavg"][..., 0],
            ref["SoilMoist_tavg"].isel(SoilMoist_profiles=0).values,
        )
        np.testing.assert_allclose(
            ds["SoilMoist_tavg"][...],
            lis_noah_stacked["SoilMoist_tavg"].transpose(..., "time").values,
        )

    # make sure that the time coordinate has nice units
    ds = xr.open_dataset(outpath)
    assert ds.time.dtype == np.dtype("datetime64[ns]")


def test_repurpose_lis(cli_args_lis, lis_noah_stacked):
    outpath = Path(cli_args_lis[1])
    repurpose(cli_args_lis)
    reader = GriddedNcOrthoMultiTs(outpath)
    ref = XarrayTSReader(lis_noah_stacked, "SoilMoist_tavg")
    assert np.all(
        np.sort(reader.grid.activegpis) == np.sort(ref.grid.activegpis)
    )
    for gpi in reader.grid.activegpis:
        ts = reader.read(gpi)
        ref_ts = ref.read(gpi)


@pytest.fixture
def cli_args_cmip(test_output_path):
    input_path = (
        test_data_path
        / "cmip6"
        / "mrsos_day_EC-Earth3-Veg_land-hist_r1i1p1f1_gr_19700101-19700131.nc"
    )
    landmask_path = test_data_path / "cmip6" / "landmask.nc"
    output_path = test_output_path
    return [
        str(input_path),
        str(output_path),
        "1970-01-01T00:00",
        "1970-01-10T00:00",
        *("--parameter", "mrsos"),
        *("--bbox", "10", "34", "43", "71"),
        *("--landmask", f"{str(landmask_path)}:landmask")
    ]


def test_transpose_cmip(cli_args_cmip, cmip_ds):
    args = cli_args_cmip
    args[1] = args[1] + "/cmip_transposed.nc"
    outpath = Path(args[1])
    transpose(args)
    ref = cmip_ds.sel(lat=slice(34, 71), lon=slice(10, 43))
    with h5netcdf.File(outpath, "r", decode_vlen_strings=False) as ds:
        assert ds["mrsos"].dimensions == ("lat", "lon", "time")
        assert ds["mrsos"].shape == (53, 47, 9)
        np.testing.assert_allclose(
            ds["mrsos"][..., 0],
            ref["mrsos"].values[0, ...],
        )
        np.testing.assert_allclose(
            ds["mrsos"][...],
            ref["mrsos"].transpose(..., "time").values[..., 0:9],
        )
    # make sure that the time coordinate has nice units
    ds = xr.open_dataset(outpath)
    assert ds.time.dtype == np.dtype("datetime64[ns]")


def test_repurpose_cmip(cli_args_cmip, cmip_ds):
    outpath = Path(cli_args_cmip[1])
    repurpose(cli_args_cmip)
    reader = GriddedNcOrthoMultiTs(outpath)
    ref_ds = cmip_ds.sel(lat=slice(34, 71), lon=slice(10, 43))
    ref = XarrayTSReader(ref_ds, "mrsos")
    # not comparing the grid GPIs here, because for "repurpose", the grid
    # started of as a global grid, from which a bbox was selected, while for
    # XarrayTSReader the grid was already only points the bbox.
    _, lons, lats, _ = reader.grid.get_grid_points()
    for lon, lat in zip(lons, lats):
        ts = reader.read(lon, lat)["mrsos"]
        assert len(ts) == 9
        ref_ts = ref.read(lon, lat)[0:9]
        assert np.all(ts == ref_ts)
