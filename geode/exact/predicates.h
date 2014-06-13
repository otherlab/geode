// Exact geometric predicates
#pragma once

#include <geode/exact/config.h>
#include <geode/vector/Vector.h>
namespace geode {

#define P2 exact::Point2
#define P3 exact::Point3

template<int axis,int d> GEODE_CORE_EXPORT GEODE_PURE bool
axis_less_degenerate(const Tuple<int,Vector<Quantized,d>> a, const Tuple<int,Vector<Quantized,d>> b) GEODE_COLD;

// Is a[axis] < b[axis]?
template<int axis,int d> GEODE_PURE static inline bool
axis_less(const Tuple<int,Vector<Quantized,d>> a, const Tuple<int,Vector<Quantized,d>> b) {
  if (a.y[axis] != b.y[axis])
    return a.y[axis] < b.y[axis];
  return axis_less_degenerate<axis>(a,b);
}

// axis_less with a runtime axis
template<int d> GEODE_PURE static inline bool
axis_less(const int axis, const Tuple<int,Vector<Quantized,d>> a, const Tuple<int,Vector<Quantized,d>> b) {
  if (a.y[axis] != b.y[axis])
    return a.y[axis] < b.y[axis];
  static_assert(d <= 3,"");
  return d<=1 || axis==0 ? axis_less_degenerate<0>(a,b)
       : d<=2 || axis==1 ? axis_less_degenerate<(d>1?1:0)>(a,b)
                         : axis_less_degenerate<(d>2?2:0)>(a,b);
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

/*** 2D predicates ***/

// Is a 2D triangle positively oriented?
GEODE_CORE_EXPORT GEODE_PURE bool triangle_oriented(const P2 p0, const P2 p1, const P2 p2);

// Is the rotation from d0 to d1 positive?
GEODE_CORE_EXPORT GEODE_PURE bool directions_oriented(const P2 d0, const P2 d1);

// Is the rotation from a1-a0 to b1-b0 positive?
GEODE_CORE_EXPORT GEODE_PURE bool segment_directions_oriented(const P2 a0, const P2 a1, const P2 b0, const P2 b1);

// Is the rotation from a1-a0 to d positive?  This is needed for sentinel incircle tests.
GEODE_CORE_EXPORT GEODE_PURE bool segment_to_direction_oriented(const P2 a0, const P2 a1, const P2 d);

// Given segments a,b,c, does intersect(a,b) come before intersect(a,c) on segment a?
// b and c are allowed to share one point (but not two).
GEODE_CORE_EXPORT GEODE_PURE bool segment_intersections_ordered(const P2 a0, const P2 a1,
                                                                const P2 b0, const P2 b1,
                                                                const P2 c0, const P2 c1);

// Given segments a,b and vertex c, is intersect(a,b) upwards from c
GEODE_CORE_EXPORT GEODE_PURE bool segment_intersection_upwards(const P2 a0, const P2 a1,
                                                               const P2 b0, const P2 b1, const P2 c);

// Does p3 lie inside the circle defined by p0,p1,p2?  This predicate is antisymmetric.
GEODE_CORE_EXPORT GEODE_PURE bool incircle(const P2 p0, const P2 p1, const P2 p2, const P2 p3);

// Does segment a0,a1 intersect b0,b1?
GEODE_CORE_EXPORT GEODE_PURE bool segments_intersect(const P2 a0, const P2 a1, const P2 b0, const P2 b1);

/*** 3D predicates ***/

// Is a 3D tetrahedron positively oriented?
GEODE_CORE_EXPORT GEODE_PURE bool tetrahedron_oriented(const P3 p0, const P3 p1,
                                                       const P3 p2, const P3 p3);

// Does segment a0,a1 intersect triangle b0,b1,b2?
GEODE_CORE_EXPORT GEODE_PURE bool segment_triangle_intersect(const P3 a0, const P3 a1,
                                                             const P3 b0, const P3 b1, const P3 b2);

// Do segment (a0,a1) and triangle (b0,b1,b2) have matching orientations?
GEODE_CORE_EXPORT GEODE_PURE bool segment_triangle_oriented(const P3 a0, const P3 a1,
                                                            const P3 b0, const P3 b1, const P3 b2);

// Given segment a and triangles b,c, does intersect(a,b) come before intersect(a,c) on segment a?
GEODE_CORE_EXPORT GEODE_PURE bool segment_triangle_intersections_ordered(const P3 a0, const P3 a1,
                                                                         const P3 b0, const P3 b1, const P3 b2,
                                                                         const P3 c0, const P3 c1, const P3 c2);

// Given triangles a,b,c, is det(na,nb,nc) > 0?
GEODE_CORE_EXPORT GEODE_PURE bool triangles_oriented(const P3 a0, const P3 a1, const P3 a2,
                                                     const P3 b0, const P3 b1, const P3 b2,
                                                     const P3 c0, const P3 c1, const P3 c2);

#undef P3
#undef P2

}
