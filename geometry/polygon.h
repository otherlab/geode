#pragma once

#include <other/core/structure/Tuple.h>
#include <other/core/array/Array.h>
#include <other/core/mesh/forward.h>
#include <other/core/python/Ptr.h>

namespace other {

template<class TV, int d> class SimplexTree;

OTHER_CORE_EXPORT Array<Vec2> polygon_from_index_list(RawArray<const Vec2> positions, RawArray<const int> indices);
OTHER_CORE_EXPORT Nested<Vec2> polygons_from_index_list(RawArray<const Vec2> positions, Nested<const int> indices);

// Compute signed area of polygon(s)
OTHER_CORE_EXPORT real polygon_area(RawArray<const Vec2> poly);
OTHER_CORE_EXPORT real polygon_area(Nested<const Vec2> polys);

// Compute the length of an open polygon
OTHER_CORE_EXPORT real open_polygon_length(RawArray<const Vec2> poly);

// Compute the circumference of a closed polygon
OTHER_CORE_EXPORT real polygon_length(RawArray<const Vec2> poly);

// Enforce maximum edge length along the polygon
OTHER_CORE_EXPORT Array<Vec2> resample_polygon(RawArray<const Vec2> poly, double maximum_edge_length);

// check whether the outlines of two polygons intersect (returns false if one is completely inside the other)
// if p2_tree is NULL, a search tree is created for p2
OTHER_CORE_EXPORT bool polygon_outlines_intersect(RawArray<const Vec2> p1, RawArray<const Vec2> p2, Ptr<SimplexTree<Vec2,1>> p2_tree = Ptr<>());

// Is the point inside the polygon?  WARNING: Not robust
OTHER_CORE_EXPORT bool inside_polygon(RawArray<const Vec2> poly, const Vec2 p);

// Find a point inside the shape defined by polys, and inside the contour poly.
// TODO: This is used only by CGAL Delaunay to compute seed points.  Our version won't use approximate seed points once, so this function should be discarded once our version exists.
OTHER_CORE_EXPORT Vec2 point_inside_polygon_component(RawArray<const Vec2> poly, Nested<const Vec2> polys);

// Warning: Not robust
OTHER_CORE_EXPORT Tuple<Array<Vec2>,Array<int>> offset_polygon_with_correspondence(RawArray<const Vec2> poly, real offset, real maxangle_deg = 20., real minangle_deg = 10.);

// Turn an array of polygons into a SegmentSoup.
OTHER_CORE_EXPORT Ref<SegmentSoup> nested_array_offsets_to_segment_mesh(RawArray<const int> offsets, bool open);
template<class TV> static inline Tuple<Ref<SegmentSoup>,Array<TV>> to_segment_mesh(const Nested<TV>& polys, bool open) {
  return tuple(nested_array_offsets_to_segment_mesh(polys.offsets,open),polys.flat);
}

// Make it easy to overload python functions to work with one or many polygons
OTHER_CORE_EXPORT Nested<const Vec2> polygons_from_python(PyObject* object);

// Reorder some polygons into canonical form, assuming nondegeneracy.  Primarily for debugging and unit test purposes.
OTHER_CORE_EXPORT Nested<Vec2> canonicalize_polygons(Nested<const Vec2> polys);

// Helper routine for closed_contours_next
OTHER_CORE_EXPORT Array<int> closed_contours_next_from_offsets(RawArray<const int> offsets);

// nested.flat[i] connects to nested.flat[closed_contour_next[i]]
// This allows traversing closed contours as a graph instead of with special cases or messy modular arithmetic
template<class T,bool f> static inline Array<int> closed_contours_next(const Nested<T,f>& nested) {
  return closed_contours_next_from_offsets(nested.offsets);
}

}
