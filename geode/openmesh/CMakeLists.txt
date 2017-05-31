set(module_SOURCES
  curvature.cpp
  decimate.cpp
  triangulator.cpp
  TriMesh.cpp
  visualize.cpp
)

set(module_HEADERS
  color_cast.h
  curvature.h
  decimate.h
  floodfill.h
  forward.h
  triangulator.h
  TriMesh.h
  visualize.h
)

install_geode_headers(openmesh ${module_HEADERS})

add_geode_module(openmesh ${module_SOURCES})
