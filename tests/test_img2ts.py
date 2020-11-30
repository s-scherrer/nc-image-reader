import numpy as np
from pathlib import Path
import shutil

from gswp.interface import GSWPTs
from gswp.reshuffle import img2ts


def test_img2ts():

    here = Path(__file__).resolve().parent

    start = np.datetime64("1970-01-01")
    end = np.datetime64("1970-01-31")

    dataset_root = here / "test_data"
    timeseries_root = here / "output"

    # remove old test output (in case a test failed)
    if timeseries_root.exists():
        shutil.rmtree(timeseries_root)

    dataset = {}
    for only_land in [False, True]:

        ts_root = timeseries_root / f"only_land={only_land}"

        reshuffler = img2ts(
            dataset_root, ts_root, start, end, only_land=only_land
        )

        assert len(reshuffler.target_grid.arrcell) > 1

        # make sure the output exists
        assert ts_root.exists()
        assert (ts_root / "grid.nc").exists()

        # try reading time series
        dataset[only_land] = GSWPTs(str(ts_root))

    # get total grid size from test data
    ds = reshuffler.imgin
    num_gpis = ds.dataset.mrsos.isel(time=0).size

    # evaluation for land gpis
    land_gpis = dataset[True].grid.activegpis
    all_gpis = dataset[False].grid.activegpis
    non_land_gpis = list(set(all_gpis) - set(land_gpis))
    assert len(land_gpis) < num_gpis
    assert len(non_land_gpis) < num_gpis

    for gpi in [54169, np.random.choice(land_gpis)]:
        ts = {}
        for land in [True, False]:
            ts[land] = dataset[land].read(gpi)

            # values are in kg/m^2, and are the upper 10cm water content
            # therefore dividing by 1000 * 0.1 should give volumetric water
            # content
            theta = ts[land] / (1000 * 0.1)
            assert not np.any(np.isnan(theta))
            assert np.all(theta < 1)
            assert np.all(theta > 0)
        assert np.all(ts[True] == ts[False])

    # evaluation for gpis not on land
    gpi = np.random.choice(non_land_gpis)
    ts = dataset[False].read(gpi)
    assert np.all(np.isnan(ts))

    # remove test output
    shutil.rmtree(timeseries_root)
