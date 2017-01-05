DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
ARG1="$1"
shift
CMAKE_ARGS=$@
echo "CMAKE_ARGS: ${CMAKE_ARGS}"
if [ "${ARG1}" = "debug" ]; then
  CMAKE_ARGS+=" -DCMAKE_BUILD_TYPE=Debug "
elif [ "${ARG1}" = "release" ]; then
  CMAKE_ARGS+=" -DCMAKE_BUILD_TYPE=Release "
elif [ "${ARG1}" = "clean" ]; then
  (cd "${DIR}/build" && make clean)
  exit 0
else
  echo "Missing or unrecognized metabuild argument \"${ARG1}\""
  exit 1
fi

DIR_3RD_PARTY="${DIR}/../"
CMAKE_ARGS+=" -DGMP_LIB_DIR=${DIR_3RD_PARTY}/mpir/.libs/ "
CMAKE_ARGS+=" -DGMP_INCLUDE_DIR=${DIR_3RD_PARTY}/mpir/ "
CMAKE_ARGS+=" -DCMAKE_VERBOSE_MAKEFILE=On "

SYSTEM_NAME=`uname -s`
# Platform specific config
# So far I've only tested this on one OSX machine. This will probably require additional tweaking for Linux and Windows
if [ "$SYSTEM_NAME" = "Darwin" ]; then
  PKG_CONFIG_PATH="/System/Library/Frameworks/Python.framework/Versions/2.7/lib/pkgconfig:${PKG_CONFIG_PATH}"
fi
#
set -x
(cd ${DIR} && (mkdir build; cd build && PKG_CONFIG_PATH=${PKG_CONFIG_PATH} cmake $CMAKE_ARGS ../ && make))
