#!/bin/bash

export CC=gcc-5
export CXX=g++-5

if [ ! -f "GeodeSupport.cmake" ]
then
   echo "run from top level directory"
   exit
fi

source metabuild_scripts/common_geode_build.sh

clean_build_dir


build_phase



