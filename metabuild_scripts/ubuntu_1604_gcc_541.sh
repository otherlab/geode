#!/bin/bash


if [ ! -f "GeodeSupport.cmake" ]
then
   echo "run from top level directory"
   exit
fi

source metabuild_scripts/common_geode_build.sh

# C++ things
export CC=gcc-5
export CXX=g++-5

clean_build_dir

build_phase


