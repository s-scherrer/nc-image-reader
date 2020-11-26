# The MIT License (MIT)
#
# Copyright (c) 2018, TU Wien
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Module for a command line interface to convert the GLDAS data into a
time series format using the repurpose package
"""

import argparse
from pathlib import Path
import numpy as np
import sys

from repurpose.img2ts import Img2Ts
from gwsp.interface import GWSPDataset


def str2bool(val):  # pragma: no cover
    return val in ["True", "true", "t", "T", "1"]


def parse_args(args):  # pragma: no cover
    """
    Parse command line parameters for recursive download.

    Parameters
    ----------
    args : list of str
        Command line parameters as list of strings.

    Returns
    -------
    args : argparse.Namespace
        Command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Convert GLDAS data to time series format."
    )
    parser.add_argument(
        "dataset_root",
        help="Root of local filesystem where the " "data is stored.",
    )

    parser.add_argument(
        "timeseries_root",
        help="Root of local filesystem where the timeseries "
        "should be stored.",
    )

    parser.add_argument(
        "start",
        type=np.datetime64,
        help=(
            "Startdate. Either in format YYYY-MM-DD or " "YYYY-MM-DDTHH:MM."
        ),
    )

    parser.add_argument(
        "end",
        type=np.datetime64,
        help=("Enddate. Either in format YYYY-MM-DD or " "YYYY-MM-DDTHH:MM."),
    )

    # TODO
    parser.add_argument(
        "parameters",
        metavar="parameters",
        nargs="+",
        help=(
            "Parameters to reshuffle into time series format. "
            "e.g. SoilMoi0_10cm_inst SoilMoi10_40cm_inst for "
            "Volumetric soil water layers 1 to 2."
        ),
    )

    parser.add_argument(
        "--land_points",
        type=str2bool,
        default="False",
        help=(
            "Set True to convert only land points as defined"
            " in the GLDAS land mask (faster and less/smaller files)"
        ),
    )

    parser.add_argument(
        "--bbox",
        type=float,
        default=None,
        nargs=4,
        help=(
            "min_lon min_lat max_lon max_lat. "
            "Bounding Box (lower left and upper right corner) "
            "of area to reshuffle (WGS84)"
        ),
    )

    parser.add_argument(
        "--imgbuffer",
        type=int,
        default=365,
        help=(
            "How many images to read at once. Bigger "
            "numbers make the conversion faster but "
            "consume more memory. Default is 365."
        ),
    )

    args = parser.parse_args(args)
    # set defaults that can not be handled by argparse

    print(
        "Converting data from {} to"
        " {} into folder {}.".format(
            np.datetime_as_string(args.start, unit="s"),
            np.datetime_as_string(args.end, unit="s"),
            args.timeseries_root,
        )
    )

    return args


def img2ts(
    dataset_root,
    timeseries_root,
    startdate,
    enddate,
    imgbuffer=365,
    only_land=False,
    bbox=None,
):
    """
    Convert the images to time series.

    Parameters
    ----------
    dataset_root : str or Path
        Path of the directory containing the data files.
    timeseries_root : str or Path
        Path of where to store the timeseries files.
    startdate : np.datetime64
        Start date of processing
    enddate : np.datetime64
        End date of processing
    imgbuffer : int, optional (default: 365)
        Number of images to read at once.
    only_land : bool, optional (default: False)
        Use the land mask to reduce the grid to land grid points only.
    bbox : list/tuple
        Bounding box parameters in the form [min_lon, min_lat, max_lon,
        max_lat]

    Returns
    -------
    reshuffler : Img2Ts object
    """
    input_dataset = GWSPDataset(
        Path(dataset_root) / "*.nc",
        only_land=only_land,
        bbox=bbox,
    )
    Path(timeseries_root).mkdir(parents=True, exist_ok=True)

    reshuffler = Img2Ts(
        input_dataset=input_dataset,
        outputpath=timeseries_root,
        startdate=startdate,
        enddate=enddate,
        ts_attributes=input_dataset.metadata,
        zlib=True,
        imgbuffer=imgbuffer,
        # this is necessary currently due to bug in repurpose
        cellsize_lat=input_dataset.cellsize,
        cellsize_lon=input_dataset.cellsize,
    )
    reshuffler.calc()

    # returned, mainly for testing/debugging
    return reshuffler


def main(args):  # pragma: no cover
    """
    Main routine used for command line interface.

    Parameters
    ----------
    args : list of str
        Command line arguments.
    """
    args = parse_args(args)
    img2ts(
        args.dataset_root,
        args.timeseries_root,
        args.start,
        args.end,
        imgbuffer=args.imgbuffer,
        only_land=args.land_points,
        bbox=args.bbox,
    )


def run():  # pragma: no cover
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
