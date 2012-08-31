#pragma once

#include <other/core/structure/Tuple.h>
#include <other/core/geometry/Segment2d.h>
#include <other/core/array/Array.h>
#include <vector>

namespace other {
  
  class SegmentMesh;
  
  typedef std::vector<Vector<real, 2> > Polygon;
  typedef std::vector<Polygon> Polygons; 
  
  Box<Vector<real,2> > bounding_box(Polygon const &poly) OTHER_EXPORT;
  Box<Vector<real,2> > bounding_box(Polygons const &polys) OTHER_EXPORT;

  Tuple<Ref<SegmentMesh>,Array<Vector<real, 2> > > to_segment_mesh(Polygons const &) OTHER_EXPORT;

  Polygon polygon_from_index_list(Array<Vector<real,2> > const &positions, RawArray<const int> indices) OTHER_EXPORT;
  Polygons polygons_from_index_list(Array<Vector<real,2> > const &positions, NestedArray<const int> indices) OTHER_EXPORT;

  // compute signed area of polygon(s)
  real polygon_area(Polygons const &polys) OTHER_EXPORT;
  real polygon_area(Polygon const &poly) OTHER_EXPORT;
  
  // compute the length of an (open) polyline
  real polyline_length(Polygon const &poly) OTHER_EXPORT;

  // compute the circumference of a closed polygon
  real polygon_length(Polygon const &poly) OTHER_EXPORT;
  
  // enforce maximum edge length along the polygon
  Polygon resample_polygon(Polygon poly, double maximum_edge_length) OTHER_EXPORT;
  
  // return whether the point is inside the polygon
  bool inside_polygon(Vector<real,2> const &p, Polygon const &poly) OTHER_EXPORT;

  // find a point inside the shape defined by polys, and inside the contour poly
  Vector<real,2> point_inside_polygon_component(Polygon const &poly, Polygons const &polys) OTHER_EXPORT;
  
  Tuple<Polygon, std::vector<int> > offset_polygon_with_correspondence(Polygon const &poly, real offset, real maxangle_deg = 20., real minangle_deg = 10.) OTHER_EXPORT;
  
}
