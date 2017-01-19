project(geode)
cmake_minimum_required(VERSION 3.2)

include(GeodeSupport.cmake)

find_package(PkgConfig)

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

# Find PYTHON_EXECTUABLE first based on $PATH
find_package(PythonInterp)
if (PYTHONINTERP_FOUND)
  message(STATUS "Python binary found at ${PYTHON_EXECUTABLE}")

  execute_process(
    COMMAND
      ${PYTHON_EXECUTABLE} -c "import numpy, sys;sys.stdout.write(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIRS
  )

  # This uses sysconfig from inside PYTHON_EXECTUABLE to ensure we link against same python executable that is present in $PATH
  execute_process(
    COMMAND
      ${PYTHON_EXECUTABLE} -c "import sysconfig, sys;sys.stdout.write(sysconfig.get_config_var('LIBPC'))"
    OUTPUT_VARIABLE PYTHON_LIBPC
  )

  message(STATUS "Numpy found in ${NUMPY_INCLUDE_DIRS}")
  message(STATUS "Looking for python pkgconfig in ${PYTHON_LIBPC}")

  unset(PYTHON_FOUND) # WIP: Force pkg_check_modules to search even if there is a cached PYTHON_FOUND
  set(ENV{PKG_CONFIG_PATH} "${PYTHON_LIBPC}:$ENV{PKG_CONFIG_PATH}")
  pkg_check_modules(_TEMP_PYTHON python)
  message(STATUS "Testing if PYTHON_FOUND set after looking: ${_TEMP_PYTHON_FOUND}")
  message(STATUS "Python headers should be at: ${_TEMP_PYTHON_INCLUDEDIR}")

  message(STATUS "Expecting to find ${_TEMP_PYTHON_LIBRARIES} in ${_TEMP_PYTHON_LIBRARY_DIRS}")

  # PYTHON_LIBRARIES and PYTHON_LIBRARY_DIRS have the name and path to where python library exists
  # We need to combine those and add appropriate prefixes and suffixes so we can pass library directly later
  # Use a name other than PYTHON_LIBRARY since find_library won't do anything if variable is already set
  unset(__TEMP_PYTHON_LIBRARY)
  message(STATUS "Expecting to find ${_TEMP_PYTHON_LIBRARIES} in ${_TEMP_PYTHON_LIBRARY_DIRS}")
  find_library(__TEMP_PYTHON_LIBRARY NAMES ${_TEMP_PYTHON_LIBRARIES} PATHS ${_TEMP_PYTHON_LIBRARY_DIRS} NO_DEFAULT_PATH)
  message(STATUS "find_library returned ${__TEMP_PYTHON_LIBRARY}")
  if(__TEMP_PYTHON_LIBRARY)
    set(PYTHON_LIBRARY ${__TEMP_PYTHON_LIBRARY})
    message(STATUS "Python library found at ${PYTHON_LIBRARY}")
    set(PYTHON_FOUND TRUE)
    set(PYTHON_INCLUDEDIR ${_TEMP_PYTHON_INCLUDEDIR})
  else()
    message(WARNING " Unable to find full path to python lib")
    set(PYTHON_FOUND FALSE)
  endif()
else()
  message(WARNING "No python interpreter found!")
endif()

if(NOT PYTHON_FOUND)
  message(STATUS "Python libs not found. Python bindings will not be built.")
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
