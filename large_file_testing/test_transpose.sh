#!/bin/bash

transpose_netcdf\
    --parameter X\
    --fmt "testdata_%Y%m%d%H%M.nc"\
    --chunks 36 72 -1\
    "large_testdata_time_first_images"\
    "large_testdata_time_last.nc"\
    "2000-01-01"\
    "2020-01-01"
