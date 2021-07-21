"""
Timeseries and image readers for wrapping xarray.Datasets, compatible with
readers from the TUW-GEO python package universe (e.g. pygeobase, pynetcf).
"""

from abc import ABC, abstractmethod
import datetime
import fnmatch
import glob
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
from typing import Union, Iterable, List, Tuple
import xarray as xr

from pygeobase.object_base import Image
from pygeogrids.grids import gridfromdims, BasicGrid, CellGrid
from pynetcf.time_series import GriddedNcOrthoMultiTs as _GriddedNcOrthoMultiTs

from .exceptions import ReaderError


class XarrayMetadataMixin:
    """
    Provides functions to get grid and metadata from an xarray
    dataset.
    """

    def _get_lat(self, ds: xr.Dataset) -> xr.DataArray:
        lat = ds[self.latname]
        if self.latdim is not None:
            lat = lat.isel({self.londim: 0})
        return lat

    def _get_lon(self, ds: xr.Dataset) -> xr.DataArray:
        lon = ds[self.lonname]
        if self.londim is not None:
            lon = lon.isel({self.latdim: 0})
        return lon

    def _metadata_from_xarray(self, ds: xr.Dataset) -> Tuple[dict, dict]:
        dataset_metadata = dict(ds.attrs)
        array_metadata = dict(ds[self.varname].attrs)
        return dataset_metadata, array_metadata

    def _grid_from_xarray(self, ds: xr.Dataset) -> CellGrid:

        # if using regular lat-lon grid, we can use gridfromdims
        lat = self._get_lat(ds)
        lon = self._get_lon(ds)
        if self._has_regular_grid:
            grid = gridfromdims(lon, lat)
            locdim = "loc"
        else:
            grid = BasicGrid(lon, lat)

        if hasattr(self, "landmask") and self.landmask is not None:
            if self._has_regular_grid:
                landmask = self.landmask.stack(
                    dimensions={"loc": (self.latname, self.lonname)}
                )
            else:
                landmask = self.landmask
            land_gpis = grid.get_grid_points()[0][landmask]
            grid = grid.subgrid_from_gpis(land_gpis)

        # bounding box
        if hasattr(self, "bbox") and self.bbox is not None:
            # given is: bbox = [lonmin, latmin, lonmax, latmax]
            lonmin, latmin, lonmax, latmax = (*self.bbox,)
            bbox_gpis = grid.get_bbox_grid_points(
                lonmin=lonmin,
                latmin=latmin,
                lonmax=lonmax,
                latmax=latmax,
            )
            grid = grid.subgrid_from_gpis(bbox_gpis)
        num_gpis = len(grid.activegpis)
        logging.info(f"_grid_from_xarray: Number of active gpis: {num_gpis}")

        if hasattr(self, "cellsize") and self.cellsize is not None:
            grid = grid.to_cell_grid(cellsize=self.cellsize)
            num_cells = len(grid.get_cells())
            logging.info(
                f"_grid_from_xarray: Number of grid cells: {num_cells}"
            )

        return grid


class XarrayImageReaderBase(XarrayMetadataMixin, ABC):
    """
    Base class for image readers that either read a full netcdf stack or single
    files with xarray.

    This provides the methods:
    - self.read
    - self.tstamps_for_daterange
    - self._grid_from_xarray

    and therefore meets all prerequisites for Img2Ts.

    Child classes must override `_read_image` and should call
    `_grid_from_xarray` and `_metadata_from_xarray` in their constructor after
    calling the parent constructor.
    They need to set:
    - self.grid
    - self.timestamps
    - self.dataset_metadata
    - self.array_metadata
    """

    def __init__(
        self,
        varname: str,
        var_dim_selection: dict = None,
        timename: str = "time",
        latname: str = "lat",
        lonname: str = "lon",
        latdim: str = None,
        londim: str = None,
        locdim: str = None,
        landmask: xr.DataArray = None,
        bbox: Iterable = None,
        cellsize: float = None,
    ):
        self.varname = varname
        self.var_dim_selection = var_dim_selection
        self.timename = timename
        self.latname = latname
        self.lonname = lonname
        self.latdim = latdim
        self.londim = londim
        self.locdim = locdim
        self._has_regular_grid = locdim is None
        self.landmask = landmask
        self.bbox = bbox
        self.cellsize = cellsize

    @abstractmethod
    def _read_image(self, timestamp: datetime.datetime) -> xr.Dataset:
        """
        Returns a single image for the given timestamp
        """
        ...

    def read(self, timestamp: datetime.datetime, **kwargs) -> Image:
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

        if timestamp in self.timestamps:
            img = self._read_image(timestamp)[self.varname].isel(
                self.var_dim_selection
            )
            # check if dimensions are as expected and potentially select from
            # non lat/lon/time dimensions
            latdim = self.latdim if self.latdim is not None else self.latname
            londim = self.londim if self.londim is not None else self.lonname
            expected_dims = [latdim, londim, self.timename]
            for d in img.dims:
                if d not in expected_dims:
                    raise ReaderError(
                        f"Unexpected dimension {d} in image for {timestamp}."
                    )

            if self._has_regular_grid:
                img = img.stack(dimensions={"loc": (latdim, londim)})
            data = {self.varname: img.values[self.grid.activegpis]}
            metadata = {self.varname: self.array_metadata}
            return Image(
                self.grid.arrlon, self.grid.arrlat, data, metadata, timestamp
            )
        else:
            raise ReaderError(
                f"Timestamp {timestamp} is not available in the dataset!"
            )

    def tstamps_for_daterange(
        self, start: datetime.datetime, end: datetime.datetime
    ) -> List[datetime.datetime]:
        """
        Timestamps available within the given date range.

        Parameters
        ----------
        start: datetime, np.datetime64 or str
            start of date range
        end: datetime, np.datetime64 or str
            end of date range

        Returns
        -------
        timestamps : array_like
            Array of datetime timestamps of available images in the date
            range.
        """
        if start is None:
            start = self.timestamps[0]
        if end is None:
            end = self.timestamps[-1]
        return list(filter(lambda t: t >= start and t <= end, self.timestamps))

    def read_block(
        self,
        start: datetime.datetime = None,
        end: datetime.datetime = None,
    ) -> xr.DataArray:
        """
        Reads a block of the image stack.

        Parameters
        ----------
        start : datetime.datetime, optional
            If not given, start at first timestamp in dataset.
        end : datetime.datetime, optional
            If not given, end at last timestamp in dataset.

        Returns
        -------
        block : xr.DataArray
            A block of the dataset as DataArray. In case of a regular grid,
            this will have ``self.latname`` and ``self.lonname`` as dimensions.
        """
        timestamps = self.tstamps_for_daterange(start, end)
        imgs = []
        for tstamp in timestamps:
            ds = self._read_image(tstamp)
            imgs.append(ds[self.varname].isel(self.var_dim_selection))

        # concatenate and reformat so that we get a nice lat/lon/time block out
        block = xr.concat(imgs, dim=self.timename).assign_coords(
            {self.timename: timestamps}
        )
        if self.latdim is not None:
            block = block.rename({self.latdim: self.latname}).assign_coords(
                {self.latname: self._get_lat(ds).values}
            )
        if self.londim is not None:
            block = block.rename({self.londim: self.lonname}).assign_coords(
                {self.lonname: self._get_lon(ds).values}
            )
        return block


class DirectoryImageReader(XarrayImageReaderBase):
    r"""
    Image reader for a directory containing netcdf files.

    This works for any datasets which are stored as single image files within a
    directory (and its subdirectories).

    Parameters
    ----------
    directory : str or Path
        Directory in which the netcdf files are located. Any file matching
        `pattern` within this directory or any subdirectories is used.
    varname : str
        Name of the variable that should be read.
    var_dim_selection : dict, optional
        If the variable has more dimensions than latitude, longitude, time (or
        location, time), e.g. a level dimension, a single value for each
        remaining dimension must be chosen. They can be passed here as
        dictionary mapping dimension name to integer index (this will then be
        passed to ``xr.DataArray.isel``).
    fmt : str, optional
        Format string to deduce timestamp from filename (without directory
        name). If it is ``None`` (default), the timestamps will be obtained
        from the files (which requires opening all files and is therefore less
        efficient).
        This must not contain any wildcards, only the format specifiers
        from ``datetime.datetime.strptime`` (e.g. %Y for year, %m for month, %d
        for day, %H for hours, %M for minutes, ...).
        If such a simple pattern does not work for you, you can additionally
        specify `time_regex_pattern` (see below).
    pattern : str, optional
        Glob pattern to find all files to use, default is "*.nc".
    time_regex_pattern : str, optional
        A regex pattern to extract the part of the filename that contains the
        time information. It must contain a statement in parentheses that is
        extracted with ``re.findall``.
        If you are using this, make sure that `fmt` matches the the part of the
        pattern that is kept.
        Example: Consider that your filenames follow the strptime/glob pattern
        ``MY_DATASET_.%Y%m%d.%H%M.*.nc``, for example, one filename could be
        ``MY_DATASET_.20200101.1200.<random_string>.nc`` and
        ``<random_string>`` is not the same for all files.
        Then you would specify
        ``time_regex_pattern="MY_DATASET_\.([0-9.]+)\..*\.nc"``. The matched
        pattern from the above example filename would then be
        ``"20200101.1200"``, so you should set ``fmt="%Y%m%d.%H%M"``.
    latname : str, optional
        If `locdim` is given (i.e. for non-rectangular grids), this must be the
        name of the latitude data variable, otherwise must be the name of the
        latitude coordinate. Default is "lat".
    lonname : str, optional
        If `locdim` is given (i.e. for non-rectangular grids), this must be the
        name of the longitude data variable, otherwise must be the name of the
        longitude coordinate. Default is "lon"
    latdim : str, optional
        The name of the latitude dimension in case it's not the same as the
        latitude coordinate variable.
    londim : str, optional
        The name of the longitude dimension in case it's not the same as the
        longitude coordinate variable.
    locdim : str, optional
        The name of the location dimension for non-rectangular grids.
    timename : str, optional
        The name of the time coordinate, default is "time".
    landmask : xr.DataArray, optional
        A land mask to be applied to reduce storage size.
    bbox : Iterable, optional
        (lonmin, latmin, lonmax, latmax) of a bounding box.
    cellsize : float, optional
        Spatial coverage of a single cell file in degrees. Default is ``None``.
    """

    def __init__(
        self,
        directory: Union[Path, str],
        varname: str,
        var_dim_selection: dict = None,
        fmt: str = None,
        pattern: str = "*.nc",
        time_regex_pattern: str = None,
        timename: str = "time",
        latname: str = "lat",
        lonname: str = "lon",
        latdim: str = None,
        londim: str = None,
        locdim: str = None,
        landmask: xr.DataArray = None,
        bbox: Iterable = None,
        cellsize: float = None,
    ):

        super().__init__(
            varname,
            var_dim_selection=var_dim_selection,
            timename=timename,
            latname=latname,
            lonname=lonname,
            latdim=latdim,
            londim=londim,
            locdim=locdim,
            landmask=landmask,
            bbox=bbox,
            cellsize=cellsize,
        )

        # first, we walk over the whole directory subtree and find any files
        # that match our pattern
        directory = Path(directory)
        filepaths = {}
        for root, dirs, files in os.walk(directory):
            for fname in files:
                if fnmatch.fnmatch(fname, pattern):
                    filepaths[fname] = Path(root) / fname

        if not filepaths:
            raise ReaderError(
                f"No files matching pattern {pattern} in directory "
                f"{str(directory)}"
            )

        # create the grid from the first file
        ds = xr.open_dataset(next(iter(filepaths.values())))
        self.grid = self._grid_from_xarray(ds)
        (
            self.dataset_metadata,
            self.array_metadata,
        ) = self._metadata_from_xarray(ds)

        # if possible, deduce the timestamps from the filenames and create a
        # dictionary mapping timestamps to file paths
        self.filepaths = {}
        if fmt is not None:
            if time_regex_pattern is not None:
                time_pattern = re.compile(time_regex_pattern)
            for fname, path in filepaths.items():
                if time_regex_pattern is not None:
                    match = time_pattern.findall(fname)
                    if not match:
                        raise ReaderError(
                            f"Pattern {time_regex_pattern} did not match "
                            f"{fname}"
                        )
                    timestring = match[0]
                else:
                    timestring = fname
                tstamp = datetime.datetime.strptime(timestring, fmt)
                self.filepaths[tstamp] = path
        else:
            for _, path in filepaths.items():
                ds = xr.open_dataset(path)
                if timename not in ds.indexes:
                    raise ReaderError(
                        f"Time dimension {timename} does not exist in "
                        f"{str(path)}"
                    )
                time = ds.indexes[timename]
                if len(time) != 1:
                    raise ReaderError(
                        f"Expected only a single timestamp, found {str(time)} "
                        f" in {str(path)}"
                    )
                tstamp = time[0].to_pydatetime()
                self.filepaths[tstamp] = path
        # sort the timestamps according to date, because we might have to
        # return them sorted
        self.timestamps = sorted(list(self.filepaths))

    def _read_image(self, timestamp):
        ds = xr.open_dataset(self.filepaths[timestamp])
        return ds


class XarrayImageReader(XarrayImageReaderBase):
    """
    Image reader that wraps a xarray.Dataset.

    This can be used as a generic image reader for netcdf stacks, e.g. for
    reformatting the data to timeseries format using the package ``repurpose``
    (which is implemented in ``nc_image_reader.reshuffle`` and can also be done
    using the supplied script ``repurpose-ncstack``.).

    Parameters
    ----------
    ds : xr.Dataset, Path or str
        Xarray dataset (or filename of a netCDF file). Must have a time
        coordinate and either `latname`/`latdim` and `lonname`/`latdim` (for a
        regular latitude-longitude grid) or `locdim` as additional
        coordinates/dimensions.
    varname : str
        Name of the variable that should be read.
    var_dim_selection : dict, optional
        If the variable has more dimensions than latitude, longitude, time (or
        location, time), e.g. a level dimension, a single value for each
        remaining dimension must be chosen. They can be passed here as
        dictionary mapping dimension name to integer index (this will then be
        passed to ``xr.DataArray.isel``).
    timename : str, optional
        The name of the time coordinate, default is "time".
    latname : str, optional
        If `locdim` is given (i.e. for non-rectangular grids), this must be the
        name of the latitude data variable, otherwise must be the name of the
        latitude coordinate. Default is "lat".
    lonname : str, optional
        If `locdim` is given (i.e. for non-rectangular grids), this must be the
        name of the longitude data variable, otherwise must be the name of the
        longitude coordinate. Default is "lon"
    latdim : str, optional
        The name of the latitude dimension in case it's not the same as the
        latitude coordinate variable.
    londim : str, optional
        The name of the longitude dimension in case it's not the same as the
        longitude coordinate variable.
    locdim : str, optional
        The name of the location dimension for non-rectangular grids.
    landmask : xr.DataArray, optional
        A land mask to be applied to reduce storage size.
    bbox : Iterable, optional
        (lonmin, latmin, lonmax, latmax) of a bounding box.
    cellsize : float, optional
        Spatial coverage of a single cell file in degrees. Default is ``None``.
    """

    def __init__(
        self,
        ds: xr.Dataset,
        varname: str,
        var_dim_selection: dict = None,
        timename: str = "time",
        latname: str = "lat",
        lonname: str = "lon",
        latdim: str = None,
        londim: str = None,
        locdim: str = None,
        landmask: xr.DataArray = None,
        bbox: Iterable = None,
        cellsize: float = None,
    ):
        # this already sets up the grid, afterwards we only need to get the
        # timestamps
        super().__init__(
            varname,
            var_dim_selection=var_dim_selection,
            timename=timename,
            latname=latname,
            lonname=lonname,
            latdim=latdim,
            londim=londim,
            locdim=locdim,
            landmask=landmask,
            bbox=bbox,
            cellsize=cellsize,
        )

        if isinstance(ds, (str, Path)):
            ds = xr.open_dataset(ds)

        self.data = ds
        self.grid = self._grid_from_xarray(ds)
        (
            self.dataset_metadata,
            self.array_metadata,
        ) = self._metadata_from_xarray(ds)
        self.timestamps = ds.indexes[self.timename].to_pydatetime()

    def _read_image(self, timestamp):
        return self.data.sel({self.timename: timestamp})

    def read_block(
        self,
        start: datetime.datetime = None,
        end: datetime.datetime = None,
    ) -> xr.DataArray:
        """
        Reads a block of the image stack.

        Parameters
        ----------
        start : datetime.datetime, optional
            If not given, start at first timestamp in dataset.
        end : datetime.datetime, optional
            If not given, end at last timestamp in dataset.

        Returns
        -------
        block : xr.DataArray
            A block of the dataset as DataArray. In case of a regular grid,
            this will have ``self.latname`` and ``self.lonname`` as dimensions.
        """
        tstamps = self.tstamps_for_daterange(start=start, end=end)
        block = self.data.sel({self.timename: tstamps})[self.varname].isel(
            self.var_dim_selection
        )
        # reformat so that we get a nice lat/lon/time block out
        if self.latdim is not None:
            block = block.rename({self.latdim: self.latname}).assign_coords(
                {self.latname: self._get_lat(ds).values}
            )
        if self.londim is not None:
            block = block.rename({self.londim: self.lonname}).assign_coords(
                {self.lonname: self._get_lon(ds).values}
            )
        return block


class GriddedNcOrthoMultiTs(_GriddedNcOrthoMultiTs):
    def __init__(self, ts_path, grid_path=None, **kwargs):
        """
        Class for reading time series after reshuffling.

        Parameters
        ----------
        ts_path : str
            Directory where the netcdf time series files are stored
        grid_path : str, optional (default: None)
            Path to grid file, that is used to organize the location of time
            series to read. If None is passed, grid.nc is searched for in the
            ts_path.

        Optional keyword arguments that are passed to the Gridded Base:
        ---------------------------------------------------------------
        parameters : list, optional (default: None)
            Specific variable names to read, if None are selected, all are
            read.
        offsets : dict, optional (default:None)
            Offsets (values) that are added to the parameters (keys)
        scale_factors : dict, optional (default:None)
            Offset (value) that the parameters (key) is multiplied with
        ioclass_kws: dict

        Optional keyword arguments to pass to OrthoMultiTs class:
        ---------------------------------------------------------
        read_bulk : boolean, optional (default:False)
            if set to True the data of all locations is read into memory,
            and subsequent calls to read_ts read from the cache and not from
            disk this makes reading complete files faster#
        read_dates : boolean, optional (default:False)
            if false dates will not be read automatically but only on specific
            request useable for bulk reading because currently the netCDF
            num2date routine is very slow for big datasets
        """
        if grid_path is None:  # pragma: no branch
            grid_path = os.path.join(ts_path, "grid.nc")
        grid = load_grid(grid_path)
        super().__init__(ts_path, grid, **kwargs)


class XarrayTSReader(XarrayMetadataMixin):
    """
    Wrapper for xarray.Dataset when timeseries of the data should be read.

    This is useful if you are using functions from the TUW-GEO package universe
    which require a timeseries reader, but you don't have the data in the
    pynetcf timeseries format.

    Since this is reading along the time dimension, you should make sure that
    the time dimension is either the last dimension in your netcdf (the fastest
    changing dimension), or that it is chunked in a way that makes timeseries
    access fast. To move the time dimension last, you can use the function
    ``nc_image_reader.transpose.create_transposed_netcdf`` or programs like
    ``ncpdq``.


    Parameters
    ----------
    ds : xr.Dataset, Path or str
        Xarray dataset (or filename of a netCDF file). Must have a time
        coordinate and either `latname`/`latdim` and `lonname`/`latdim` (for a
        regular latitude-longitude grid) or `locdim` as additional
        coordinates/dimensions.
    varname : str
        Name of the variable that should be read.
    var_dim_selection : dict, optional
        If the variable has more dimensions than latitude, longitude, time (or
        location, time), e.g. a level dimension, a single value for each
        remaining dimension must be chosen. They can be passed here as
        dictionary mapping dimension name to integer index (this will then be
        passed to ``xr.DataArray.isel``).
    timename : str, optional
        The name of the time coordinate, default is "time".
    latname : str, optional
        If `locdim` is given (i.e. for non-rectangular grids), this must be the
        name of the latitude data variable, otherwise must be the name of the
        latitude coordinate. Default is "lat".
    lonname : str, optional
        If `locdim` is given (i.e. for non-rectangular grids), this must be the
        name of the longitude data variable, otherwise must be the name of the
        longitude coordinate. Default is "lon"
    latdim : str, optional
        The name of the latitude dimension in case it's not the same as the
        latitude coordinate variable.
    londim : str, optional
        The name of the longitude dimension in case it's not the same as the
        longitude coordinate variable.
    locdim : str, optional
        The name of the location dimension for non-rectangular grids.
    landmask : xr.DataArray, optional
        A land mask to be applied to reduce storage size.
    bbox : Iterable, optional
        (lonmin, latmin, lonmax, latmax) of a bounding box.
    cellsize : float, optional
        Spatial coverage of a single cell file in degrees. Default is ``None``.
    """

    def __init__(
        self,
        varname: str,
        var_dim_selection: dict = None,
        timename: str = "time",
        latname: str = "lat",
        lonname: str = "lon",
        latdim: str = None,
        londim: str = None,
        locdim: str = None,
        landmask: xr.DataArray = None,
        bbox: Iterable = None,
        cellsize: float = None,
    ):
        self.varname = varname
        self.var_dim_selection = var_dim_selection
        self.timename = timename
        self.latname = latname
        self.lonname = lonname
        self.latdim = latdim
        self.londim = londim
        self.locdim = locdim
        self._has_regular_grid = locdim is None
        self.landmask = landmask
        self.bbox = bbox
        self.cellsize = cellsize

        self.data = ds[self.varname].isel(self.var_dim_selection)
        self.grid = self._grid_from_xarray(ds)
        (
            self.dataset_metadata,
            self.array_metadata,
        ) = self._metadata_from_xarray(ds)

    def read(self, *args, **kwargs) -> pd.Series:
        """
        Reads a single timeseries from dataset.

        Parameters
        ----------
        args : tuple
            If a single argument, must be an integer denoting the grid point
            index at which to read the timeseries. If two arguments, it's
            longitude and latitude values at which to read the timeseries.

        Returns
        -------
        ts : pd.Series
        """

        if len(args) == 1:
            gpi = args[0]
            if self._has_regular_grid:
                lon, lat = self.grid.gpi2lonlat(gpi)

        elif len(args) == 2:
            lon = args[0]
            lat = args[1]
            if not self._has_regular_grid:
                gpi = self.grid.find_nearest_gpi(lon, lat)[0]
                if not isinstance(gpi, np.integer):
                    raise ValueError(
                        f"No gpi near (lon={lon}, lat={lat}) found"
                    )
        else:
            raise ValueError(
                f"args must have length 1 or 2, but has length {len(args)}"
            )

        if self._has_regular_grid:
            return self.data[
                {self.latname: lat, self.lonname: lon}
            ].to_pandas()
        else:
            return self.data[{self.locdim: gpi}].to_pandas()
