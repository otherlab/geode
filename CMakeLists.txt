project(geode)
cmake_minimum_required(VERSION 3.2)

include(GeodeSupport.cmake)

find_package(PkgConfig)

pkg_search_module(PYTHON python)

find_library(GMP_LIBRARIES gmp)
if (GMP_LIBRARIES)
  set(GMP_FOUND TRUE)
endif()

if (GMP_FOUND)
  message(STATUS "GMP found. Building exact submodule")
else()
  message(STATUS "GMP not found. Exact module will not be built.")
endif()

if (PYTHON_FOUND)
  message(STATUS "Python found. Building python bindings")
else()
  message(STATUS "Python not found. Python bindings will not be built.")
endif()

add_subdirectory(geode)

configure_file(${CMAKE_SOURCE_DIR}/geode.pc.in ${CMAKE_BINARY_DIR}/geode.pc
  @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/geode.pc DESTINATION lib/pkgconfig/)


if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/gui)
  message(STATUS "gui found")
  set(Geode_DIR ${CMAKE_CURRENT_BINARY_DIR})
  add_subdirectory(gui)
endif()
