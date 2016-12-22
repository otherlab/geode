#!/bin/bash

export CC=clang-3.7
export CXX=clang++-3.7

if [ ! -f "GeodeSupport.cmake" ]
then
   echo "run from top level directory"
   exit
fi

source metabuild_scripts/common_geode_build.sh

clean_build_dir


build_phase



