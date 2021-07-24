import cftime
import datetime
import h5netcdf
import logging
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Union, Tuple, TypeVar


from .readers import XarrayImageReader


Reader = TypeVar("Reader")


def create_transposed_netcdf(
    reader: Reader,
    outfname: Union[Path, str],
    new_last_dim: str = None,
    start: datetime.datetime = None,
    end: datetime.datetime = None,
    time_units: str = "days since 1900-01-01 00:00:00",
    memory: float = 2,
    chunks: Tuple = None,
):
    """
    Creates a stacked and transposed netCDF file from a given reader.

    Parameters
    ----------
    reader : XarrayImageReaderBase
        Reader for the dataset.
    outfname : str or Path
        Output filename.
    start : datetime.datetime, optional
        If not given, start at first timestamp in dataset.
    end : datetime.datetime, optional
        If not given, end at last timestamp in dataset.
    time_units : str, optional
        The time unit to use, default is "days since 1900-01-01 00:00:00"
    memory : float, optional
        The amount of memory to be used for buffering in GB. Default is 2. Only
        used if `reader` is an instance of ``XarrayImageReader``.
    chunks : tuple
        Chunks for the output file. Must already be in the transposed order,
        i.e. the last entry corresponds to the chunksize for `new_last_dim`. -1
        indicates a chunk covering the full dimension.
    """
    new_last_dim = reader.timename
    timestamps = reader.tstamps_for_daterange(start, end)

    # first, get some info about structure of the input file
    first_img = reader.read_block(start=timestamps[0], end=timestamps[0])
    dtype = first_img.dtype
    input_dim_names = first_img.dims
    input_dim_sizes = first_img.shape
    old_pos = input_dim_names.index(new_last_dim)

    if input_dim_names[-1] == new_last_dim:  # pragma: no cover
        print(f"{new_last_dim} is already the last dimension")

    # get new dim names in the correct order
    new_dim_names = list(input_dim_names)
    new_dim_names.remove(new_last_dim)
    new_dim_names.append(new_last_dim)
    new_dim_names = tuple(new_dim_names)
    new_dim_sizes = [
        input_dim_sizes[input_dim_names.index(dim)] for dim in new_dim_names
    ]
    new_dim_sizes[-1] = len(timestamps)
    new_dim_sizes = tuple(new_dim_sizes)

    # clean the chunks argument (replace -1 entries)
    size = dtype.itemsize
    if chunks is not None:
        chunks = tuple(
            new_dim_sizes[i] if val == -1 else val
            for i, val in enumerate(chunks)
        )
        chunksize_MB = np.prod(chunks) * size / 1024 ** 2
        logging.info(
            f"create_transposed_netcdf: Creating chunks {chunks}"
            f" with chunksize {chunksize_MB:.2f} MB"
        )

    len_new_dim = new_dim_sizes[-1]
    if isinstance(reader, XarrayImageReader):
        # in case we have an XarrayImageReader, we can efficiently read a full
        # block of data, whose size is given by the `memory` argument
        imagesize_GB = np.prod(new_dim_sizes[:-1]) * size / 1024 ** 3
        # we need to divide by two, because we need intermediate storage for
        # the transposing
        stepsize = int(math.floor(memory / imagesize_GB)) // 2
        stepsize = min(stepsize, len_new_dim)
        logging.info(
            f"create_transposed_netcdf: Using {stepsize} images as buffer,"
            f" leading to buffer size of {2 * stepsize * imagesize_GB:.2f} GB"
        )
    else:
        # otherwise we'll just read single images
        stepsize = 1
    block_start = list(
        map(int, np.arange(0, len_new_dim + stepsize - 0.5, stepsize))
    )
    block_start[-1] = min(block_start[-1], len_new_dim)

    with h5netcdf.File(outfname, "w", decode_vlen_strings=False) as newfile:

        # create dimensions and coordinates
        newfile.dimensions = dict(zip(new_dim_names, new_dim_sizes))
        for dim in newfile.dimensions:
            if dim == new_last_dim:
                coord = cftime.date2num(
                    timestamps, units=time_units, calendar="standard"
                )
            else:
                coord = first_img[dim].values
            newfile.create_variable(dim, (dim,), dtype, data=coord)
        newfile[new_last_dim].attrs["units"] = time_units

        # create new variable
        newvar = newfile.create_variable(
            reader.varname, new_dim_names, dtype, chunks=chunks
        )

        # copy metadata
        for k, v in reader.dataset_metadata.items():
            newfile.attrs[k] = v
        for k, v in reader.array_metadata.items():
            newfile[reader.varname].attrs[k] = v

        # tqdm adds a nice progress bar
        for s, e in zip(tqdm(block_start[:-1]), block_start[1:]):
            transposed_block = (
                reader.read_block(start=timestamps[s], end=timestamps[e - 1])
                .transpose(..., new_last_dim)
                .values
            )
            newvar[..., slice(s, e)] = transposed_block

    logging.info("create_transposed_netcdf: Finished writing transposed file.")
