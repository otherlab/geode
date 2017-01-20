#.rst:
# FindPathPython
# ----------------
# Search for Python libs that match Python executable that is present on $PATH
# Sets the following variables:
#
#  ::
#
#   NUMPY_INCLUDE_DIRS         - Include directory for NumPy
#   PYTHON_FOUND               - Evaluates to false if unsuccessful
#   PYTHON_INCLUDE_DIRS        - Path to directory that contains Python.h
#   PYTHON_LIBRARY             - Python standard library including full path
#

find_package(PkgConfig)
find_package(PythonInterp)

if (PYTHONINTERP_FOUND)
  message(STATUS "Python binary found at ${PYTHON_EXECUTABLE}")

  execute_process(
    COMMAND
      ${PYTHON_EXECUTABLE} -c "import numpy, sys;sys.stdout.write(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIRS
  )
  message(STATUS "Numpy found in ${NUMPY_INCLUDE_DIRS}")

  # This uses sysconfig from inside PYTHON_EXECTUABLE to ensure we build against same python executable that is present on $PATH
  execute_process(
    COMMAND
      ${PYTHON_EXECUTABLE} -c "import sysconfig, sys;sys.stdout.write(sysconfig.get_config_var('LIBPC'))"
    OUTPUT_VARIABLE PYTHON_LIBPC
  )
  message(STATUS "Looking for python pkgconfig in ${PYTHON_LIBPC}")

  unset(PYTHON_FOUND) # WIP: Force pkg_check_modules to search even if there is a cached value found
  set(ENV{PKG_CONFIG_PATH} "${PYTHON_LIBPC}:$ENV{PKG_CONFIG_PATH}")
  set(ENV{PKG_CONFIG_ALLOW_SYSTEM_LIBS} "1")
  set(ENV{PKG_CONFIG_ALLOW_SYSTEM_CFLAGS} "1")
  message(STATUS "Testing if PYTHON_FOUND set before looking: ${PYTHON_FOUND}")
  pkg_check_modules(PYTHON python)
  message(STATUS "Testing if PYTHON_FOUND set after looking: ${PYTHON_FOUND}")

  if(NOT PYTHON_INCLUDE_DIRS)
    message(STATUS "pkg-config did not set Python include dirs. Falling back to sysconfig")
    execute_process(
      COMMAND
        ${PYTHON_EXECUTABLE} -c "import sysconfig, sys;sys.stdout.write(sysconfig.get_path('include'))"
      OUTPUT_VARIABLE PYTHON_INCLUDE_DIRS
    )
  endif()

  if(NOT PYTHON_LIBRARY_DIRS)
    # Ideally pkg_check_modules should have set this, but as a fallback we ask sysconfig for LIBDIR
    message(STATUS "pkg-config did not set Python library dirs. Falling back to sysconfig")
    execute_process(
      COMMAND
        ${PYTHON_EXECUTABLE} -c "import sysconfig, sys;sys.stdout.write(sysconfig.get_config_var('LIBDIR'))"
      OUTPUT_VARIABLE PYTHON_LIBRARY_DIRS
    )
  endif()

  # PYTHON_LIBRARIES and PYTHON_LIBRARY_DIRS have the name and path to where python library exists
  # We need to combine those and add appropriate prefixes and suffixes so we can pass library directly later
  # Use a name other than PYTHON_LIBRARY since find_library won't do anything if variable is already set
  unset(_PYTHON_LIBRARIES)
  message(STATUS "Expecting to find ${PYTHON_LIBRARIES} in ${PYTHON_LIBRARY_DIRS}")
  find_library(_PYTHON_LIBRARIES NAMES ${PYTHON_LIBRARIES} PATHS ${PYTHON_LIBRARY_DIRS} NO_DEFAULT_PATH)
  message(STATUS "find_library returned ${_PYTHON_LIBRARIES}")

  if(_PYTHON_LIBRARIES)
    set(PYTHON_LIBRARIES ${_PYTHON_LIBRARIES})
  else()
    message(WARNING " Unable to find full path to python lib")
  endif()
else()
  message(WARNING " No python interpreter found!")
endif()

set(PYTHON_FOUND TRUE)

if(EXISTS "${PYTHON_INCLUDE_DIRS}/Python.h")
  message(STATUS "Python.h found in: ${PYTHON_INCLUDE_DIRS}")
else()
  set(PYTHON_FOUND PYTHON_INCLUDE_DIRS-NOTFOUND)
  message(ERROR "Unable to find python include directory")
endif()

if(EXISTS "${PYTHON_LIBRARIES}")
  message(STATUS "Python lib found at: ${PYTHON_LIBRARIES}")
else()
  set(PYTHON_FOUND PYTHON_LIBRARIES-NOTFOUND)
  message(ERROR "Unable to find python library")
endif()

