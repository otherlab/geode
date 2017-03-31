#pragma once 
#include <geode/config.h>
#include <geode/mesh/TriangleTopology.h>

namespace geode {

// Generates a constrained Delaunay triangulation of the interior of a polygons (interior based on CCW winding)
// Warning: This needs improvement. For now it may return junk in some cases
//   This occurs because constrained Delaunay triangulation requires intersection free edges but polygon csg constructs approximate intersection points that can introduce new self intersections
//   I suspect we will want either a version of polygon csg that resolves all self intersections and/or enabling insertion of new vertices during Delaunay triangulation
Tuple<Ref<TriangleTopology>,Field<Vec2,VertexId>> triangulate_polygon(const Nested<Vec2>& raw_polygons);

} // geode namespace