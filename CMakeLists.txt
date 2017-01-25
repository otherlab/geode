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

status("GMP library" "${GMP_LIB_DIR}")
status("GMP includes " "${GMP_INCLUDE_DIR}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/)
find_package(PathPython)

if(NOT PYTHON_FOUND)
  message(STATUS "Python libs not found. Python bindings will not be built.")
endif()

status("Building python bindings" "${PYTHON_FOUND}")
status("Python binary" "${PYTHON_EXECUTABLE}")
status("Python headers" "${PYTHON_INCLUDE_DIRS}")
status("Python library " "${PYTHON_LIBRARIES}")
status("Numpy includes" "${NUMPY_INCLUDE_DIRS}")

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
else()
  set(GUI_DIR FALSE)
endif()

status("Found GUI" "${GUI_DIR}")
status("Compilers" "${CMAKE_C_COMPILER} ${CMAKE_CXX_COMPILER}")
#status_report()
