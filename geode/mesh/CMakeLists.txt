set(module_SRCS
  ComponentData.cpp
  decimate.cpp
  HalfedgeGraph.cpp
  HalfedgeMesh.cpp
  ids.cpp
  improve_mesh.cpp
  io.cpp
  lower_hull.cpp
  mesh_debug.cpp
  module.cpp
  PolygonSoup.cpp
  quadric.cpp
  refine_mesh.cpp
  SegmentSoup.cpp
  TriangleMesh.cpp
  TriangleSoup.cpp
  TriangleSubdivision.cpp
  TriangleTopology.cpp
  triangulate.cpp
)

set(module_HEADERS
  ComponentData.h
  decimate.h
  forward.h
  HalfedgeGraph.h
  HalfedgeMesh.h
  ids.h
  improve_mesh.h
  io.h
  lower_hull.h
  mesh_debug.h
  PolygonSoup.h
  quadric.h
  refine_mesh.h
  SegmentSoup.h
  TriangleMesh.h
  TriangleSoup.h
  TriangleSubdivision.h
  TriangleTopology.h
  triangulate.h
)

install_geode_headers(mesh ${module_HEADERS})

add_geode_module(mesh ${module_SRCS})
