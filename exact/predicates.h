// Exact geometric predicates
#pragma once

#include <other/core/exact/config.h>
#include <other/core/vector/Vector.h>
namespace other {

template<int axis,int d> OTHER_CORE_EXPORT OTHER_CONST bool axis_less_degenerate(const Tuple<int,Vector<Quantized,d>> a, const Tuple<int,Vector<Quantized,d>> b) OTHER_COLD;

// Is a[axis] < b[axis]?
template<int axis,int d> OTHER_CONST static inline bool axis_less(const Tuple<int,Vector<Quantized,d>> a, const Tuple<int,Vector<Quantized,d>> b) {
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
OTHER_CORE_EXPORT OTHER_CONST bool triangle_oriented(const exact::Point2 p0, const exact::Point2 p1, const exact::Point2 p2);

// Is the rotation from d0 to d1 positive?
OTHER_CORE_EXPORT OTHER_CONST bool directions_oriented(const exact::Point2 d0, const exact::Point2 d1);

// Is the rotation from a1-a0 to b1-b0 positive?
OTHER_CORE_EXPORT OTHER_CONST bool segment_directions_oriented(const exact::Point2 a0, const exact::Point2 a1, const exact::Point2 b0, const exact::Point2 b1);

// Is the rotation from a1-a0 to d positive?  This is needed for sentinel incircle tests.
OTHER_CORE_EXPORT OTHER_CONST bool segment_to_direction_oriented(const exact::Point2 a0, const exact::Point2 a1, const exact::Point2 d);

// Given segments a,b,c, does intersect(a,b) come before intersect(a,c) on segment a?  b and c are allowed to share one point (but not two).
OTHER_CORE_EXPORT OTHER_CONST bool segment_intersections_ordered(const exact::Point2 a0, const exact::Point2 a1, const exact::Point2 b0, const exact::Point2 b1, const exact::Point2 c0, const exact::Point2 c1);

// Does p3 lie inside the circle defined by p0,p1,p2?  This predicate is antisymmetric.
OTHER_CORE_EXPORT OTHER_CONST bool incircle(const exact::Point2 p0, const exact::Point2 p1, const exact::Point2 p2, const exact::Point2 p3);

// Does segment a0,a1 intersect b0,b1?
OTHER_CORE_EXPORT OTHER_CONST bool segments_intersect(const exact::Point2 a0, const exact::Point2 a1, const exact::Point2 b0, const exact::Point2 b1);

}
