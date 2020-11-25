import numpy as np
from pathlib import Path
import shutil

from gwsp.interface import GWSPTs
from gwsp.reshuffle import img2ts


def test_img2ts():

    here = Path(__file__).resolve().parent

    start = np.datetime64("1970-01-01")
    end = np.datetime64("1970-01-31")

    dataset_root = here / "test_data"
    timeseries_root = here / "output"

    img2ts(dataset_root, timeseries_root, start, end)

    assert timeseries_root.exists()
    assert (timeseries_root / "grid.nc").exists()

    # try reading time series
    ds = GWSPTs(str(timeseries_root))
    grid = ds.grid
    gpi = np.random.choice(grid.activegpis)
    ts = ds.read(gpi)

    # values are in kg/m^2, and are the upper 10cm water content
    # therefore dividing by 1000 * 0.1 should give volumetric water content
    theta = ts / (1000 * 0.1)
    assert np.all(theta < 1) or np.all(np.isnan(theta))
    assert np.all(theta > 0) or np.all(np.isnan(theta))

    # remove test output
    shutil.rmtree(timeseries_root)
