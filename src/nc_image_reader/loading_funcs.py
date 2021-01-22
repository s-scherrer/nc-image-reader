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
Functions to load netcdf dataset from file using xarray and to convert to
format expected by ``GriddedXrOrthoMultiImage``.

IMPORTANT: Any function whose name matches "load_<something>" that is defined
here will be available via loading_func=<something> in
``GriddedXrOrthoMultiImage``.
"""


import numpy as np
import xarray as xr


def load_cmip6(fname, parameters):
    ds = xr.open_mfdataset(str(fname), parallel=True, concat_dim="time")
    ds = ds[parameters]
    ds['landmask'] = ~np.isnan(ds[parameters[0]].isel(time=0))
    return ds


def load_lis_noah(fname, parameters):
    # prepare ds
    ds = xr.open_mfdataset(fname, parallel=True, concat_dim="time")
    ds = ds.reindex(
        east_west=np.unique(ds.lon.isel(time=0)),
        north_south=np.unique(ds.lat.isel(time=0))
    )
    ds = ds.drop_vars(["lon", "lat"])
    ds = ds.rename({"east_west": "lon", "north_south": "lat"})

    # extract parameters
    for p in parameters:
        if p == "ssm":
            ds[p] = ds.SoilMoist_tavg.isel(SoilMoist_profiles=0)
        elif p == "rzsm":
            ds[p] = ds.SoilMoist_tavg.isel(SoilMoist_profiles=3)
        else:  # pragma: no cover
            raise ValueError(f"Parameter '{p}' is not available.")
    ds = ds[parameters]
    return ds
