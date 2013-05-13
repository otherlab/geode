#pragma once

#include <other/core/structure/Tuple.h>
#include <other/core/array/Array.h>
namespace other {

class SegmentMesh;
typedef Vector<real,2> Vec2;

OTHER_CORE_EXPORT Array<Vec2> polygon_from_index_list(RawArray<const Vec2> positions, RawArray<const int> indices);
OTHER_CORE_EXPORT NestedArray<Vec2> polygons_from_index_list(RawArray<const Vec2> positions, NestedArray<const int> indices);

// Compute signed area of polygon(s)
OTHER_CORE_EXPORT real polygon_area(RawArray<const Vec2> poly);
OTHER_CORE_EXPORT real polygon_area(NestedArray<const Vec2> polys);

// Compute the length of an open polygon
OTHER_CORE_EXPORT real open_polygon_length(RawArray<const Vec2> poly);

// Compute the circumference of a closed polygon
OTHER_CORE_EXPORT real polygon_length(RawArray<const Vec2> poly);

// Enforce maximum edge length along the polygon
OTHER_CORE_EXPORT Array<Vec2> resample_polygon(RawArray<const Vec2> poly, double maximum_edge_length);

// Is the point inside the polygon?  WARNING: Not robust
OTHER_CORE_EXPORT bool inside_polygon(RawArray<const Vec2> poly, const Vec2 p);

// Find a point inside the shape defined by polys, and inside the contour poly.
// TODO: This is used only by CGAL Delaunay to compute seed points.  Our version won't use approximate seed points once, so this function should be discarded once our version exists.
OTHER_CORE_EXPORT Vec2 point_inside_polygon_component(RawArray<const Vec2> poly, NestedArray<const Vec2> polys);

OTHER_CORE_EXPORT Tuple<Array<Vec2>,Array<int>> offset_polygon_with_correspondence(RawArray<const Vec2> poly, real offset, real maxangle_deg = 20., real minangle_deg = 10.);

// Turn an array of polygons into a SegmentMesh.
OTHER_CORE_EXPORT Ref<SegmentMesh> nested_array_offsets_to_segment_mesh(RawArray<const int> offsets, bool open);
template<class TV> static inline Tuple<Ref<SegmentMesh>,Array<TV>> to_segment_mesh(const NestedArray<TV>& polys, bool open) {
  return tuple(nested_array_offsets_to_segment_mesh(polys.offsets,open),polys.flat);
}

// Make it easy to overload python functions to work with one or many polygons
OTHER_CORE_EXPORT NestedArray<const Vec2> polygons_from_python(PyObject* object);

}
