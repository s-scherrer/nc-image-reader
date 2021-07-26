import numpy as np
from pathlib import Path
import time
from joblib import Parallel, delayed
import argparse

from nc_image_reader.readers import GriddedNcOrthoMultiTs, XarrayTSReader


parser = argparse.ArgumentParser()
parser.add_argument("--parallel", action="store_true")
args = parser.parse_args()
parallel = args.parallel

points = [
    (48.125, 16.375),  # Vienna
    # (34.125, 95.375),  # China
    # (-23.875, 24.125),  # Botswana
    # (32.375, 3.125),  # Algeria
    # (-34.125, -60.375),  # Argentina
    # (38.375, -91.875),  # USA
]

newpoints = []
for p in points:
    for i in range(-1, 2):
        for j in range(-1, 2):
            x = round(p[0] + i * 0.25, 4)
            y = round(p[1] + j * 0.25, 4)
            newpoints.append((x, y))

basepath = Path(__file__).resolve().parent


readers = {
    "pynetcf": GriddedNcOrthoMultiTs(
        str(basepath / "large_testdata_time_last_timeseries"),
        ioclass_kws=dict(read_bulk=True),
    ),
    "transposed NetCDF": XarrayTSReader(
        basepath / "large_testdata_time_last_rechunked.nc", "X"
    ),
}

print("--------------------------------------------")
print("num points =", len(newpoints))
print("parallel =", parallel)

for name, reader in readers.items():

    print("--------------------------------------------")
    start_time = time.time()
    if parallel:
        means = Parallel(n_jobs=4, prefer="processes")(
            delayed(np.mean)(reader.read(lon, lat).values)
            for lat, lon in newpoints
        )
    else:
        means = [
            np.mean(reader.read(lon, lat).values) for lat, lon in newpoints
        ]
    print(f"mean of means: {np.mean(means):4f}")
    end_time = time.time()

    duration = end_time - start_time
    print(f"{name.ljust(30)}\t{duration: <3.4f} s")

print("--------------------------------------------")
