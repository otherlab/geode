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

# python things

PY_INC=$(./metabuild_scripts/show_py_inc.py)
PY_LIB=$(./metabuild_scripts/show_py_lib.py) 

export  CXXFLAGS="$CXXFLAGS -I$PY_INC"
export  LDFLAGS="$LDFLAGS -L$PY_LIB"

clean_build_dir

build_phase


