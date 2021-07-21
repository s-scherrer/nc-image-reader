import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr


if __name__ == "__main__":
    lat = np.arange(-89.875, 90, 0.25)
    lon = np.arange(-179.875, 180, 0.25)
    time = pd.date_range("2000", "2020", freq="D")
    nlat, nlon, ntime = len(lat), len(lon), len(time)

    print("Dimension sizes:", nlat, nlon, ntime)
    print(f"Dataset size: {8 * nlat * nlon * ntime/1024**3:.2f}, GB")

    X = da.random.normal(
        loc=0, scale=1, size=(ntime, nlat, nlon), chunks=(1, nlat, nlon)
    )
    ds = xr.Dataset(
        {"X": (["time", "lat", "lon"], X)},
        coords={"time": time, "lat": lat, "lon": lon},
    )
    ds = ds.chunk({"time": 1, "lat": nlat, "lon": nlon})
    ds.to_netcdf("large_testdata_time_first.nc")
