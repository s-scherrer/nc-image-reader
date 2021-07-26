#!/bin/bash

repurpose_images\
    "large_testdata_time_first_images"\
    "large_testdata_time_last_timeseries"\
    "2000-01-01"\
    "2020-01-01"\
    --parameter X\
    --time_fmt "testdata_%Y%m%d%H%M.nc"\
