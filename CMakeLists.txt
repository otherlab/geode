project(geode)
cmake_minimum_required(VERSION 3.2)

include(GeodeSupport.cmake)

find_package(PkgConfig)

# Find PYTHON_EXECTUABLE first based on $PATH
find_package(PythonInterp)

# This module then uses PYTHON_EXECUTABLE to find compile flags
find_package(PythonLibs)

set(GMP_LIB_DIR "/usr/lib/" CACHE PATH "Path to libgmp.so")
set(GMP_INCLUDE_DIR "/usr/local/Cellar/ /usr/include/" CACHE PATH "Path to gmp.h")

find_library(
  GMP_LIBRARIES gmp
  HINTS ${GMP_LIB_DIR}
)
find_path(
  GMP_INCLUDE gmp.h
  HINTS ${GMP_INCLUDE_DIR}
)
if (GMP_LIBRARIES AND GMP_INCLUDE)
  set(GMP_FOUND TRUE)
endif()

if (GMP_FOUND)
  message(STATUS "GMP found in ${GMP_LIBRARIES}")
else()
  message(FATAL_ERROR "Required dependency GMP not found. Set GMP_LIB_DIR and GMP_INCLUDE_DIR to inform cmake where to look")
endif()

if (PYTHONINTERP_FOUND)
  message(STATUS "Python binary found at ${PYTHON_EXECUTABLE}. Building python bindings")
  execute_process(
    COMMAND
      ${PYTHON_EXECUTABLE} -c "import numpy, sys;sys.stdout.write(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIRS
  )
  message(STATUS "Numpy found in ${NUMPY_INCLUDE_DIRS}")
  message(STATUS "Using python libraries in ${PYTHON_LIBRARIES}")
  message(STATUS "Using python headers in ${PYTHON_INCLUDE_DIRS}")
else()
  message(STATUS "Python not found. Python bindings will not be built.")
endif()

add_subdirectory(geode)

configure_file(${CMAKE_SOURCE_DIR}/geode.pc.in ${CMAKE_BINARY_DIR}/geode.pc
  @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/geode.pc DESTINATION lib/pkgconfig/)

if(NOT GUI_DIR)
  # Check in current directory by default
  set(GUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/gui)
endif()

if (EXISTS ${GUI_DIR})
  message(STATUS "gui found at ${GUI_DIR}")
  set(Geode_DIR ${CMAKE_CURRENT_BINARY_DIR}/geode)
  add_subdirectory(${GUI_DIR} ${CMAKE_CURRENT_BINARY_DIR}/gui)
endif()
