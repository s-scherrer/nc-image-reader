# The MIT License (MIT)
#
# Copyright (c) 2020, TU Wien
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
Module for a command line interface to convert the netcdf image data into
timeseries format or a transposed netcdf with time as last dimension
"""

import argparse
import datetime
from pathlib import Path
import numpy as np
from typing import Union, Sequence
import sys

from nc_image_reader.readers import XarrayImageReader, DirectoryImageReader
from nc_image_reader.transpose import create_transposed_netcdf


def str2bool(val):
    return val in ["True", "true", "t", "T", "1"]


def mkdate(datestring):
    if len(datestring) == 10:
        return datetime.datetime.strptime(datestring, "%Y-%m-%d")
    elif len(datestring) == 16:
        return datetime.datetime.strptime(datestring, "%Y-%m-%dT%H:%M")
    else:
        raise ValueError(f"Invalid date: {datestring}")


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, description):
        super().__init__(description=description)

        # common arguments for both scripts
        self.add_argument(
            "dataset_root",
            help=(
                "Path where the data is stored, either"
                " a directory or a netCDF file"
            ),
        )
        self.add_argument(
            "output_root", help="Path where the output should be stored."
        )
        self.add_argument(
            "start",
            type=mkdate,
            help=(
                "Startdate. Either in format YYYY-MM-DD or "
                "YYYY-MM-DDTHH:MM."
            ),
        )
        self.add_argument(
            "end",
            type=mkdate,
            help=(
                "Enddate. Either in format YYYY-MM-DD or " "YYYY-MM-DDTHH:MM."
            ),
        )
        self.add_argument(
            "--parameters",
            type=str,
            required=True,
            help="Parameter to process.",
        )
        self.add_argument(
            "--pattern",
            type=str,
            default="*.nc",
            help=(
                "If dataset_root is a directory, glob pattern to match files"
                " Default is '*.nc'"
            ),
        )
        self.add_argument(
            "--time_fmt",
            type=str,
            help=(
                "If DATASET_ROOT is a directory, strptime format string to"
                " deduce the data from the filenames. This can improve the"
                " performance significantly."
            ),
        )
        self.add_argument(
            "--time_regex_pattern",
            type=str,
            help=(
                "If dataset_root is a directory, a regex pattern to select"
                " the time string from the filename. If this is used, TIME_FMT"
                " must be chosen accordingly. See nc_image_reader.readers for"
                " more info."
            ),
        )
        self.add_argument(
            "--latname",
            type=str,
            default="lat",
            help="Name of the latitude coordinate. Default is 'lat'",
        )
        self.add_argument(
            "--latdim",
            type=str,
            help="Name of the latitude dimension (e.g. north_south)",
        )
        self.add_argument(
            "--lonname",
            type=str,
            default="lon",
            help="Name of the longitude coordinate. Default is 'lon'",
        )
        self.add_argument(
            "--londim",
            type=str,
            help="Name of the longitude dimension (e.g. east_west).",
        )
        self.add_argument(
            "--locdim",
            type=str,
            help="Name of the location dimension for non-regular grids.",
        )
        self.add_argument(
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
        self.add_argument(
            "--cellsize",
            type=float,
            default=5.0,
            help=("Size of single file cells. Default is 5.0."),
        )


class RepurposeArgumentParser(ArgumentParser):
    def __init__(self):
        super().__init__("Converts data to time series format.")
        self.add_argument(
            "--imgbuffer",
            type=int,
            default=365,
            help=(
                "How many images to read at once. Bigger "
                "numbers make the conversion faster but "
                "consume more memory. Default is 365."
            ),
        )


class TransposeArgumentParser(ArgumentParser):
    def __init__(self):
        super().__init__("Converts data to transposed netCDF.")
        self.add_argument(
            "--memory",
            type=float,
            default=2,
            help="The amount of memory to use as buffer in GB",
        )
        self.add_argument(
            "--chunks",
            type=float,
            default=None,
            nargs="+",
            help="Chunksizes for the transposed netCDF.",
        )


def parse_args(parser, args):
    """
    Parse command line parameters for recursive download.

    Parameters
    ----------
    args : list of str
        Command line parameters as list of strings.

    Returns
    -------
    reader, args
    """
    parser = argparse.ArgumentParser(
        description="Convert data to stacked transposed netCDF."
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

    common_reader_kwargs = dict(
        latname=args.latname,
        lonname=args.lonname,
        latdim=args.latdim,
        londim=args.londim,
        locdim=args.locdim,
        bbox=args.bbox,
        cellsize=args.cellsize,
    )

    input_path = Path(dataset_root)
    if input_path.is_file():
        reader = XarrayImageReader(
            input_path,
            parameter,
            fmt=args.time_fmt,
            pattern=args.pattern,
            time_regex_pattern=args.time_regex_pattern,
            **common_reader_kwargs,
        )
    else:
        reader = DirectoryImageReader(
            input_path,
            parameter,
            fmt=args.time_fmt,
            pattern=args.pattern,
            time_regex_pattern=args.time_regex_pattern,
            **common_reader_kwargs,
        )

    return reader, args


def reshuffle(args):
    parser = RepurposeArgumentParser()
    reader, args = parse_args(parser, args)
    reshuffler = Img2Ts(
        input_dataset=reader,
        outputpath=args.output_root,
        startdate=args.start,
        enddate=args.end,
        ts_attributes=reader.dataset_metadata,
        zlib=True,
        imgbuffer=imgbuffer,
        # this is necessary currently due to bug in repurpose
        cellsize_lat=args.cellsize,
        cellsize_lon=args.cellsize,
    )
    reshuffler.calc()


def transpose(args):
    parser = TransposeArgumentParser()
    reader, args = parse_args(parser, args)
    create_transposed_netcdf(
        reader,
        args.output_root,
        start=args.start,
        end=args.end,
        memory=args.memory,
        chunks=tuple(args.chunks),
    )


def run_reshuffle():  # pragma: no cover
    reshuffle(sys.argv[1:])


def run_transpose():  # pragma: no cover
    transpose(sys.argv[1:])
