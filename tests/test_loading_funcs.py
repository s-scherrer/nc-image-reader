from datetime import datetime
import numpy as np
from pathlib import Path
import pytest

from nc_image_reader.interface import GriddedXrOrthoMultiImage
from nc_image_reader.loading_funcs import load_cmip6


def test_data():
    return Path(__file__).resolve().parent / "test_data"


def test_lis_noah():
    """
    Tests whether reading using datetime and returning datetime arrays from
    tstamps_for_daterange works.
    """

    fname = test_data() / "lis_noah" / "*" / "LIS_HIST_*.nc"
    ds = GriddedXrOrthoMultiImage(str(fname), ["ssm", "rzsm"], "lis_noah")
    date_array = ds.tstamps_for_daterange("2017-03-01", "2017-04-30")
    time = date_array[0]

    assert isinstance(time, datetime)

    # try reading
    img = ds.read(time)
    assert img.timestamp == time
