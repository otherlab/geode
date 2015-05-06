#!/bin/bash
# TODO: Test config for OSX
# TODO: Determine appropriate architecture (i.g. "args+=(arch=nocona)")

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
if [ "$1" = "clean" ]; then
  (cd $DIR && scons -c --config=force $lib_setup_args)
elif [ "$1" = "build" ]; then
  declare -a args=()
  SYSTEM_NAME=`uname -s`
  if [ "$SYSTEM_NAME" = "Darwin" ]; then
    # OSX specific config:
    # These were previously needed, but maybe aren't any more?:
    # args+=(cxxflags_extra='-std=c++11 -stdlib=libc++' linkflags_extra='-stdlib=libc++')
    # args+=(sse=0)
    echo "" # Can't have an empty if block
  elif [ "$SYSTEM_NAME" = "MINGW64_NT-6.1" ]; then
    # Windows specific config:
    args+=(libs_extra=psapi)
  else
    echo "ERROR: Unrecognized or unsupported platform: $SYSTEM_NAME!"
    exit 1
  fi

  args+=(use_python=0 use_libjpeg=0 use_libpng=0 use_openexr=0 use_boost=0)
  args+=(shared=0 install=0)
  args+=(use_openmesh=1 openmesh_libpath=#/../OpenMesh-2.0/build/Build/lib openmesh_publiclibs='OpenMeshCore,OpenMeshTools' openmesh_include=#/../OpenMesh-2.0/src)
  args+=(use_gmp=1 gmp_libpath=#/../mpir/.libs/ gmp_include=#/../mpir/)

  for type in debug release; do
    scons_args="--config=force -j7 prefix=#build/\$arch/\$type type=$type ${args[@]}"
    echo scons $scons_args
    (cd $DIR && scons $scons_args) || exit 1
  done
else
  echo 'Missing or unrecognized metabuild argument: '\'$1\'
  exit 1
fi
