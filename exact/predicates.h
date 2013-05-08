// Exact geometric predicates
#pragma once

#include <other/core/exact/config.h>
#include <other/core/vector/Vector.h>
namespace other {

template<int axis,int d> OTHER_CORE_EXPORT bool axis_less_degenerate(const Tuple<int,Vector<exact::Int,d>> a, const Tuple<int,Vector<exact::Int,d>> b) OTHER_COLD;

// Is a[axis] < b[axis]?
template<int axis,int d> static inline bool axis_less(const Tuple<int,Vector<exact::Int,d>> a, const Tuple<int,Vector<exact::Int,d>> b) {
  if (a.y[axis] != b.y[axis])
    return a.y[axis] < b.y[axis];
  return axis_less_degenerate<axis>(a,b);
}

// Is a 2D triangle positively oriented?
OTHER_CORE_EXPORT bool triangle_oriented(const exact::Point2 p0, const exact::Point2 p1, const exact::Point2 p2);

// Is the rotation from d0 to d1 positive?
OTHER_CORE_EXPORT bool directions_oriented(const exact::Point2 d0, const exact::Point2 d1);

// Is the rotation from a1-a0 to b1-b0 positive?
OTHER_CORE_EXPORT bool segment_directions_oriented(const exact::Point2 a0, const exact::Point2 a1, const exact::Point2 b0, const exact::Point2 b1);

// Is the rotation from a1-a0 to d positive?  This is needed for sentinel incircle tests.
OTHER_CORE_EXPORT bool segment_to_direction_oriented(const exact::Point2 a0, const exact::Point2 a1, const exact::Point2 d);

// Given segments a,b,c, does intersect(a,b) come before intersect(a,c) on segment a?
// This predicate answers that question assuming that da,db and da,dc are positively oriented.
OTHER_CORE_EXPORT bool segment_intersections_ordered_helper(const exact::Point2 a0, const exact::Point2 a1, const exact::Point2 b0, const exact::Point2 b1, const exact::Point2 c0, const exact::Point2 c1);

// Does p3 lie inside the circle defined by p0,p1,p2?  This predicate is antisymmetric.
OTHER_CORE_EXPORT bool incircle(const exact::Point2 p0, const exact::Point2 p1, const exact::Point2 p2, const exact::Point2 p3);

}
