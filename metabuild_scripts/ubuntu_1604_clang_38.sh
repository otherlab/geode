#!/bin/bash

if [ ! -f "GeodeSupport.cmake" ]
then
   echo "run from top level directory"
   exit
fi

source metabuild_scripts/common_geode_build.sh

# C++ things
export CC=clang-3.8
export CXX=clang++-3.8
export  CXXFLAGS="$CXXFLAGS -std=c++1y "

clean_build_dir

build_phase


