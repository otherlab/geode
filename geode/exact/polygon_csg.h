// Robust constructive solid geometry for polygons in the plane
#pragma once

// Each routines come with and without an exact_ prefix.  The exact_ versions take in and produce quantized data.
//
// Since any newly constructed points are inexactly quantized, the CSG routines are not necessarily idempotent:
// calling polygon_union multiple times may add more and more points.
//
// Warning: The worst case complexity of these algorithms is quadratic, since O(n) arbitrary line segments may
// have up to O(n^2) intersections, and contour depth is computed in a worst case O(n^2) fashion.

#include <geode/exact/config.h>
#include <geode/array/Nested.h>
namespace geode {

// Resolve all intersections between polygons, and extract the contour with given *external* depth.
// Depth starts at 0 at infinity, and increases by 1 when crossing a contour from outside to inside.
// For example, depth = 0 corresponds to polygon_union.
GEODE_CORE_EXPORT Nested<Vec2> split_polygons(Nested<const Vec2> polys, const int depth);
GEODE_CORE_EXPORT Nested<exact::Vec2> exact_split_polygons(Nested<const exact::Vec2> polys, const int depth);

// The union of possibly intersecting polygons, assuming consistent ordering
template<class... Polys> static inline Nested<Vec2> polygon_union(const Polys&... polys) {
  return split_polygons(concatenate(polys...),0);
}

// The intersection of possibly intersecting polygons, assuming consistent ordering.
template<class... Polys> static inline Nested<Vec2> polygon_intersection(const Polys&... polys) {
  return split_polygons(concatenate(polys...),sizeof...(Polys)-1);
}

enum class FillRule { Greater, Parity, NotEqual };
GEODE_CORE_EXPORT std::string str(const FillRule rule);
GEODE_CORE_EXPORT Nested<Vec2> split_polygons_with_rule(Nested<const Vec2> polys, const int depth, const FillRule rule);
GEODE_CORE_EXPORT Nested<Vec2> exact_split_polygons_with_rule(Nested<const Vec2> polys, const int depth, const FillRule rule);

// These are primarily to allow python bindings without needing to create class for FillRule
GEODE_CORE_EXPORT Nested<Vec2> split_polygons_greater(Nested<const Vec2> polys, const int depth);
GEODE_CORE_EXPORT Nested<Vec2> split_polygons_parity(Nested<const Vec2> polys, const int depth);
GEODE_CORE_EXPORT Nested<Vec2> split_polygons_neq(Nested<const Vec2> polys, const int depth);

} // namespace geode
