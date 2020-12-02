import numpy as np
from pathlib import Path
import pytest
import shutil

from gswp.interface import GSWPTs
from gswp.reshuffle import img2ts, main, parse_args, _create_reshuffler


@pytest.fixture
def timeseries_root():
    here = Path(__file__).resolve().parent
    return here / "output"


@pytest.fixture
def dataset_root():
    here = Path(__file__).resolve().parent
    return here / "test_data"


def test_img2ts(timeseries_root, dataset_root):

    start = np.datetime64("1970-01-01")
    end = np.datetime64("1970-01-31")

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
        assert (ts_root / "1001.nc").exists()

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


@pytest.fixture
def commandline_args(dataset_root, timeseries_root):
    return [
        "--land_points", "t",
        "--bbox", "-20", "-20", "20", "20",
        "--imgbuffer", "10",
        str(dataset_root),
        str(timeseries_root),
        "1970-01-01",
        "1970-01-10",
    ]


def test_parse_args(commandline_args, dataset_root, timeseries_root):
    args = parse_args(commandline_args)

    assert np.all(args.bbox == [-20, -20, 20, 20])
    assert args.land_points == True
    assert args.imgbuffer == 10
    assert args.start == np.datetime64("1970-01-01")
    assert args.end == np.datetime64("1970-01-10")
    assert args.dataset_root == str(dataset_root)
    assert args.timeseries_root == str(timeseries_root)


def test_main(timeseries_root, commandline_args):
    # tests if running from the commandline works

    main(commandline_args)

    assert timeseries_root.exists()
    assert (timeseries_root / "grid.nc").exists()
    assert (timeseries_root / "1280.nc").exists()

    # remove test output
    shutil.rmtree(timeseries_root)


def test_bbox(commandline_args):
    args = parse_args(commandline_args)
    reshuffler = _create_reshuffler(
        args.dataset_root,
        args.timeseries_root,
        args.start,
        args.end,
        imgbuffer=args.imgbuffer,
        only_land=args.land_points,
        bbox=args.bbox,
    )

    assert len(reshuffler.target_grid.get_cells()) == 36
    assert reshuffler.imgin.lonmin == -20
    
