from datetime import datetime
import numpy as np
from pathlib import Path
import pytest

from gwsp.interface import GWSPDataset


@pytest.fixture
def filename_pattern():
    here = Path(__file__).resolve().parent
    return here / "test_data" / "*.nc"


def test_datetime_compatibility(filename_pattern):
    """
    Tests whether reading using datetime and returning datetime arrays from
    tstamps_for_daterange works.
    """

    ds = GWSPDataset(filename_pattern)
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
    ds = GWSPDataset(filename_pattern, only_land=True)
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

    min_lon = -50
    min_lat = -50
    max_lon = 50
    max_lat = 50

    ds = GWSPDataset(
        filename_pattern, bbox=[min_lon, min_lat, max_lon, max_lat]
    )
    num_gpis = ds.dataset.mrsos.isel(time=0).size

    assert hasattr(ds, "bbox_gpis")
    assert len(ds.grid.activegpis) < num_gpis

    assert not np.any(ds.grid.arrlon < min_lon)
    assert not np.any(ds.grid.arrlat < min_lat)
    assert not np.any(ds.grid.arrlon > max_lon)
    assert not np.any(ds.grid.arrlat > max_lat)
