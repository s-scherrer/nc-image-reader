import logging
import sys

from nc_image_reader.readers import XarrayImageReader
from nc_image_reader.transpose import create_transposed_netcdf

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s"
)

if __name__ == "__main__":
    reader = XarrayImageReader("large_testdata_time_first.nc", "X")
    create_transposed_netcdf(
        reader, "large_testdata_time_last.nc", memory=10, chunks=(72, 144, -1)
    )
