// Exact geometric predicates
#pragma once

#include <geode/exact/config.h>
#include <geode/vector/Vector.h>
namespace geode {

template<int axis,int d> GEODE_CORE_EXPORT GEODE_PURE bool axis_less_degenerate(const Tuple<int,Vector<Quantized,d>> a, const Tuple<int,Vector<Quantized,d>> b) GEODE_COLD;

// Is a[axis] < b[axis]?
template<int axis,int d> GEODE_PURE static inline bool axis_less(const Tuple<int,Vector<Quantized,d>> a, const Tuple<int,Vector<Quantized,d>> b) {
  if (a.y[axis] != b.y[axis])
    return a.y[axis] < b.y[axis];
  return axis_less_degenerate<axis>(a,b);
}

// Is a.x < b.x?
static inline bool rightwards(const Tuple<int,Vector<Quantized,2>> a, const Tuple<int,Vector<Quantized,2>> b) {
  if (a.y.x != b.y.x)
    return a.y.x < b.y.x;
  return axis_less_degenerate<0>(a,b);
}

// Is a.y < b.y?
static inline bool upwards(const Tuple<int,Vector<Quantized,2>> a, const Tuple<int,Vector<Quantized,2>> b) {
  if (a.y.y != b.y.y)
    return a.y.y < b.y.y;
  return axis_less_degenerate<1>(a,b);
}

// Is a 2D triangle positively oriented?
GEODE_CORE_EXPORT GEODE_PURE bool triangle_oriented(const exact::Point2 p0, const exact::Point2 p1, const exact::Point2 p2);

// Is the rotation from d0 to d1 positive?
GEODE_CORE_EXPORT GEODE_PURE bool directions_oriented(const exact::Point2 d0, const exact::Point2 d1);

// Is the rotation from a1-a0 to b1-b0 positive?
GEODE_CORE_EXPORT GEODE_PURE bool segment_directions_oriented(const exact::Point2 a0, const exact::Point2 a1, const exact::Point2 b0, const exact::Point2 b1);

// Is the rotation from a1-a0 to d positive?  This is needed for sentinel incircle tests.
GEODE_CORE_EXPORT GEODE_PURE bool segment_to_direction_oriented(const exact::Point2 a0, const exact::Point2 a1, const exact::Point2 d);

// Given segments a,b,c, does intersect(a,b) come before intersect(a,c) on segment a?  b and c are allowed to share one point (but not two).
GEODE_CORE_EXPORT GEODE_PURE bool segment_intersections_ordered(const exact::Point2 a0, const exact::Point2 a1, const exact::Point2 b0, const exact::Point2 b1, const exact::Point2 c0, const exact::Point2 c1);

// Given segments a,b and vertex c, is intersect(a,b) upwards from c
GEODE_CORE_EXPORT GEODE_PURE bool segment_intersection_upwards(const exact::Point2 a0, const exact::Point2 a1, const exact::Point2 b0, const exact::Point2 b1, const exact::Point2 c);

// Does p3 lie inside the circle defined by p0,p1,p2?  This predicate is antisymmetric.
GEODE_CORE_EXPORT GEODE_PURE bool incircle(const exact::Point2 p0, const exact::Point2 p1, const exact::Point2 p2, const exact::Point2 p3);

// Does segment a0,a1 intersect b0,b1?
GEODE_CORE_EXPORT GEODE_PURE bool segments_intersect(const exact::Point2 a0, const exact::Point2 a1, const exact::Point2 b0, const exact::Point2 b1);

}
