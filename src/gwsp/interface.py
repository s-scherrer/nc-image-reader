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

from copy import copy
import os
import numpy as np
import xarray as xr

from pygeobase.object_base import Image
from pygeogrids.grids import BasicGrid
from pygeogrids.netcdf import load_grid
from pynetcf.time_series import GriddedNcOrthoMultiTs


class GWSPDataset:
    """
    Class that reads GWSP data from multiple nc files and provides
    `daily_images` iterator for Img2Ts, as well as a `CellGrid` instance.

    Parameters
    ----------
    filename_pattern : string
        Pattern matching the dataset for use in `xarray.open_mfdataset()`, for
        example "mrsos_day_EC-Earth3-Veg*.nc"
    parameter : string, optional
        Name of the parameter in the netCDF-file. Default is "mrsos".
    cellsize : float, optional
        Size of cell files in degrees. Default is 5.0.
    only_land : bool, optional (default: False)
        Use the land mask to reduce the grid to land grid points only.
    bbox : list/tuple or None
        Bounding box parameters in the form [min_lon, min_lat, max_lon,
        max_lat]
    """

    def __init__(
        self,
        filename_pattern,
        parameter="mrsos",
        cellsize=5.0,
        only_land=False,
        bbox=None,
    ):
        self.parameter = parameter
        self.cellsize = cellsize
        self.only_land = only_land

        # open dataset with xarray, using dask backend for parallel access
        ds = xr.open_mfdataset(
            str(filename_pattern), parallel=True, concat_dim="time"
        )
        self.dataset = ds.stack(dimensions={"latlon": ("lat", "lon")})

        # setup grid
        global_grid = BasicGrid(self.lon, self.lat)

        # land mask
        self.landmask = ~np.isnan(self.dataset[parameter].isel(time=0)).values
        self.land_gpis = global_grid.get_grid_points()[0][self.landmask]
        if self.only_land:
            grid = global_grid.subgrid_from_gpis(self.land_gpis)
        else:
            grid = global_grid

        # bounding box
        if bbox is not None:
            # given is: bbox = [lonmin, latmin, lonmax, latmax]
            self.lonmin, self.latmin, self.lonmax, self.latmax = (*bbox,)
            self.bbox_gpis = grid.get_bbox_grid_points(
                lonmin=self.lonmin,
                latmin=self.latmin,
                lonmax=self.lonmax,
                latmax=self.latmax
            )
            grid = grid.subgrid_from_gpis(self.bbox_gpis)

        self.grid = grid.to_cell_grid(cellsize=self.cellsize)

        # create metadata dictionary from dataset attributes
        self.metadata = copy(self.dataset.attrs)
        array_metadata = copy(self.dataset[self.parameter].attrs)
        # merging history metadata (the only common keyword)
        self.metadata["history"] = (
            "Dataset: "
            + self.metadata["history"]
            + "; DataArray: "
            + array_metadata["history"]
        )
        del array_metadata["history"]
        self.metadata.update(array_metadata)

    @property
    def lon(self):
        return self.dataset.lon.values

    @property
    def lat(self):
        return self.dataset.lat.values

    def read(self, timestamp, **kwargs):
        """
        Read a single image at a given timestamp. Raises `KeyError` if
        timestamp is not available in the dataset.

        Parameters
        ----------
        timestamp : datetime.datetime
            Timestamp of image of interest

        Returns
        -------
        img_dict : dict
            Dictionary containing the image data as numpy array, using the
            parameter name as key.

        Raises
        ------
        KeyError
        """
        try:
            data = {
                self.parameter: self.dataset[self.parameter]
                .sel(time=timestamp)
                .values
            }
            return Image(
                self.lon, self.lat, data, self.metadata, timestamp
            )
        except KeyError:  # pragma: no cover
            raise KeyError(
                f"Timestamp {timestamp} is not available in the dataset!"
            )

    def tstamps_for_daterange(self, start_date, end_date):
        """
        Timestamps available within the given date range.

        Parameters
        ----------
        start_date: datetime, np.datetime64 or str
            start of date range
        end_date: datetime, np.datetime64 or str
            end of date range

        Returns
        -------
        timestamps : array_like
            Array of datetime timestamps of available images in the date
            range.
        """
        start, end = map(np.datetime64, (start_date, end_date))
        return (
            self.dataset.time.sel(time=slice(start, end))
            .indexes["time"]
            .to_pydatetime()
        )


class GWSPTs(GriddedNcOrthoMultiTs):
    def __init__(self, ts_path, grid_path=None, **kwargs):
        """
        Class for reading GWSP time series after reshuffling.

        Parameters
        ----------
        ts_path : str
            Directory where the netcdf time series files are stored
        grid_path : str, optional (default: None)
            Path to grid file, that is used to organize the location of time
            series to read. If None is passed, grid.nc is searched for in the
            ts_path.

        Optional keyword arguments that are passed to the Gridded Base:
        ------------------------------------------------------------------------
        parameters : list, optional (default: None)
            Specific variable names to read, if None are selected, all are
            read.
        offsets : dict, optional (default:None)
            Offsets (values) that are added to the parameters (keys)
        scale_factors : dict, optional (default:None)
            Offset (value) that the parameters (key) is multiplied with
        ioclass_kws: dict

        Optional keyword arguments to pass to OrthoMultiTs class:
        ----------------------------------------------------------------
        read_bulk : boolean, optional (default:False)
            if set to True the data of all locations is read into memory,
            and subsequent calls to read_ts read from the cache and not from
            disk this makes reading complete files faster#
        read_dates : boolean, optional (default:False)
            if false dates will not be read automatically but only on specific
            request useable for bulk reading because currently the netCDF
            num2date routine is very slow for big datasets
        """
        if grid_path is None:
            grid_path = os.path.join(ts_path, "grid.nc")
        grid = load_grid(grid_path)
        super().__init__(ts_path, grid, **kwargs)
