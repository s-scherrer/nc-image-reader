from datetime import datetime
from pathlib import Path
import pytest

from gwsp.interface import GWSPDataset


@pytest.fixture
def test_data():
    here = Path(__file__).resolve().parent
    filename_pattern = here / "test_data" / "*.nc"
    return GWSPDataset(filename_pattern)


def test_datetime_compatibility(test_data):
    """
    Tests whether reading using datetime and returning datetime arrays from
    tstamps_for_daterange works.
    """

    ds = test_data
    date_array = ds.tstamps_for_daterange("1970-01-01", "1970-01-31")
    time = date_array[0]

    assert isinstance(time, datetime)

    # try reading
    img = ds.read(time)
    assert img.timestamp == time
