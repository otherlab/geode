#!/bin/bash

if [ ! -f "GeodeSupport.cmake" ]
then
   echo "run from top level directory"
   exit
fi

source metabuild_scripts/Common_geode_build.sh

# C++ things
export  CXX="/usr/bin/g++ "
export  CXXFLAGS="$CXXFLAGS -std=c++0x "
export  LDFLAGS="$LDFLAGS "

clean_build_dir

build_phase


