from datetime import datetime
import numpy as np
from pathlib import Path
import pytest

from gswp.interface import GSWPDataset


@pytest.fixture
def filename_pattern():
    here = Path(__file__).resolve().parent
    return here / "test_data" / "*.nc"


def test_datetime_compatibility(filename_pattern):
    """
    Tests whether reading using datetime and returning datetime arrays from
    tstamps_for_daterange works.
    """

    ds = GSWPDataset(filename_pattern)
    date_array = ds.tstamps_for_daterange("1970-01-01", "1970-01-31")
    time = date_array[0]

    assert isinstance(time, datetime)

    # try reading
    img = ds.read(time)
    assert img.timestamp == time


def test_only_land(filename_pattern):
    """
    Tests if the only_land feature works as expected.
    """
    ds = GSWPDataset(filename_pattern, only_land=True)
    num_gpis = ds.dataset.mrsos.isel(time=0).size

    assert len(ds.grid.activegpis) < num_gpis

    # get random image and check whether there are any nans on land
    num_times = len(ds.dataset.mrsos.time)
    t = np.random.randint(num_times)

    land_img = ds.dataset.mrsos.isel(time=t, latlon=ds.grid.activegpis)
    assert not np.any(np.isnan(land_img))


def test_bbox(filename_pattern):
    """
    Tests the bounding box feature
    """

    min_lon = -160
    min_lat = 15
    max_lon = -150
    max_lat = 25

    ds = GSWPDataset(
        filename_pattern, bbox=[min_lon, min_lat, max_lon, max_lat]
    )
    num_gpis = ds.dataset.mrsos.isel(time=0).size

    assert hasattr(ds, "bbox_gpis")
    assert len(ds.grid.activegpis) < num_gpis
    assert len(np.unique(ds.grid.activearrcell)) == 4

    assert not np.any(ds.grid.arrlon < min_lon)
    assert not np.any(ds.grid.arrlat < min_lat)
    assert not np.any(ds.grid.arrlon > max_lon)
    assert not np.any(ds.grid.arrlat > max_lat)


def test_grid_lons(filename_pattern):
    """
    Tests if the grid of the dataset has only longitudes between -180 and 180
    """

    ds = GSWPDataset(filename_pattern)

    lons = ds.grid.arrlon
    assert np.all(lons <= 180)
    assert np.all(lons > -180)
    assert np.any(lons < 0)
