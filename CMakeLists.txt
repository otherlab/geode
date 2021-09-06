project(geode)
cmake_minimum_required(VERSION 3.2)

option(GEODE_REQUIRE_PYTHON "Generate an error if python bindings won't be built instead of silently disabling them" FALSE)
option(GEODE_DISABLE_PYTHON "Don't build python bindings even if available" FALSE)

if (GEODE_REQUIRE_PYTHON AND GEODE_DISABLE_PYTHON)
  message(STATUS "Geode Python bindings set to required and disabled at the same time. Please choose at most one")
endif()

include(GeodeSupport.cmake)

find_package(PkgConfig)
find_package(JPEG)
find_package(PNG)

set(GMP_LIB_DIR "/usr/lib/" CACHE PATH "Path to libgmp.so")

if(UNIX AND NOT APPLE)
  set(GMP_INCLUDE_DIR "/usr/include/" CACHE PATH "Path to gmp.h")
else()
  set(GMP_INCLUDE_DIR "/usr/local/Cellar/" CACHE PATH "Path to gmp.h")
endif()



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



set(OPENMESH_LIB_DIR "/usr/local/lib/OpenMesh/")
set(OPENMESH_INCLUDE_DIR "/usr/local/include/OpenMesh")

find_library(
  OPENMESH_LIBRARIES libOpenMeshCore.a libOpenMeshTools.a
  HINTS ${OPENMESH_LIB_DIR}
)

find_path(
  OPENMESH_INCLUDE Core/Mesh/TriMeshT.hh
  HINTS ${OPENMESH_INCLUDE_DIR}
)

if (OPENMESH_LIBRARIES AND OPENMESH_INCLUDE)
  set(OPENMESH_FOUND TRUE)
endif()

if (OPENMESH_FOUND)
  message(STATUS "OPENMESH found in ${OPENMESH_LIBRARIES}")
else()
  message(FATAL_ERROR "Required dependency OPENMESH not found. Set OPENMESH_LIB_DIR and OPENMESH_INCLUDE_DIR to inform cmake where to look")
endif()


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/)

find_package(PathPython)

add_subdirectory(geode)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/geode.pc.in ${CMAKE_BINARY_DIR}/geode.pc
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
