#!/bin/bash

if [ ! -f "GeodeSupport.cmake" ]
then
   echo "run from top level directory"
   exit
fi

source metabuild_scripts/common_geode_build.sh


# C++ things
export  CXX="clang++-3.8 -stdlib=libc++"
export  CXXFLAGS="$CXXFLAGS -std=c++1y -I/usr/local/opt/llvm38/lib/llvm-3.8/include/c++/v1 "
export  LDFLAGS="$LDFLAGS -L/usr/local/opt/llvm38/lib/llvm-3.8/lib "

clean_build_dir

build_phase


