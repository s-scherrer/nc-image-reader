import numpy as np
import pandas as pd
from pathlib import Path
import pytest
import shutil
import xarray as xr


here = Path(__file__).resolve().parent


def pytest_configure():
    pytest.test_data_path = here / "test_data"


@pytest.fixture
def test_output_path(tmpdir_factory):
    # see https://stackoverflow.com/questions/51593595
    # for reference
    tmpdir = Path(tmpdir_factory.mktemp("output"))
    yield tmpdir
    shutil.rmtree(str(tmpdir))


@pytest.fixture
def test_dataset():
    nlat, nlon, ntime = 10, 20, 100
    lat = np.linspace(0, 1, nlat)
    lon = np.linspace(0, 1, nlon)
    time = pd.date_range("2000", periods=ntime, freq="D")

    X = np.zeros((ntime, nlat, nlon), dtype=float)
    for i in range(ntime):
        X[i, :, :] = float(i)

    ds = xr.Dataset(
        {"X": (["time", "lat", "lon"], X)},
        coords={"time": time, "lat": lat, "lon": lon},
    )
    return ds


@pytest.fixture
def test_loc_dataset(test_dataset):
    ds = test_dataset.stack({"location": ("lat", "lon")})
    ds["latitude"] = ds.lat
    ds["longitude"] = ds.lon
    ds = ds.drop_vars("location").rename(
        {"latitude": "lat", "longitude": "lon"}
    )
    return ds
