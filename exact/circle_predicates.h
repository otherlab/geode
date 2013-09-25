#pragma once

#include <other/core/exact/circle_csg.h>
#include <other/core/exact/Interval.h>
namespace other {

// Unfortunately, circular arc CSG requires a rather large number of predicates, all doing extremely similar things.  In particular,
// most of these routines answer questions about the relative positions of intersections between arcs.  Therefore, we precompute
// these intersections wherever possible, so that most questions can be answered with fast interval queries.  This mechanism also
// allows testing to be integrated into each routine, via a compile time flag that answers a question the fast way *and* the slow way
// and asserts consistency.

// A precomputed intersection of two arcs
struct Vertex {
  int i0,i1; // Flat indices of the previous and next circle
  bool left; // True if the intersection is to the left of the (i,j) segment
  uint8_t q0,q1; // Quadrants relative to i0's center and i1's center, respectively
  exact::Vec2 rounded; // The nearly exactly rounded intersect, differing from the exact intersection by at most vertex::tolerance.

  static const Quantized tolerance() { return 2; }

  bool operator==(const Vertex v) const {
    return i0==v.i0 && i1==v.i1 && left==v.left;
  }

  bool operator!=(const Vertex v) const {
    return !(*this==v);
  }

  friend Hash hash_reduce(const Vertex v) {
    return Hash(v.i0,2*v.i1+v.left);
  }

  OTHER_UNUSED friend ostream& operator<<(ostream& output, const Vertex v) {
    return output<<format("(%d,%d,%c)",v.i0,v.i1,v.left?'L':'R');
  }

  // Probably not worth accounting for, but p()/box() must be inside Box<EV2>(arcs[i].center).thickened(arcs[i].radius) for i in [i1,i2]

  // A conservative interval containing the true intersection
  Vector<Interval,2> p() const {
    return Vector<Interval,2>(Interval(rounded.x-tolerance(),rounded.x+tolerance()),
                              Interval(rounded.y-tolerance(),rounded.y+tolerance()));
  }

  // The same as p(), but with a different type
  Box<exact::Vec2> box() const {
    return Box<exact::Vec2>(rounded).thickened(tolerance());
  }

  // Reverse the vertex to go from i1 to i0
  Vertex reverse() const {
    Vertex r;
    r.i0 = i1;
    r.i1 = i0;
    r.left = !left;
    r.q0 = q1;
    r.q1 = q0;
    r.rounded = rounded;
    return r;
  }
};

typedef RawArray<const ExactCircleArc> Arcs;

// Cautionary tale: An earlier version of this routine had an option to negate the radius,
// which was used to reduce code duplication in circles_intersect.  This would have been
// a disaster, as it wouldn't have flipped the sign of the symbolic perturbation.
static inline exact::Point3 aspoint(Arcs arcs, const int arc) {
  const auto& a = arcs[arc];
  return tuple(a.index,exact::Vec3(a.center,a.radius));
}
static inline exact::Point2 aspoint_center(Arcs arcs, const int arc) {
  const auto& a = arcs[arc];
  return tuple(a.index,a.center);
}

// Check if arcs are both parts of the same circle
static inline bool arcs_from_same_circle(Arcs arcs, const int i0, const int i1) {
  if (i0 == i1)
    return true;
  const auto &a0 = arcs[i0],
             &a1 = arcs[i1];
  assert(a0.index!=a1.index || (a0.center==a1.center && a0.radius==a1.radius && a0.positive==a1.positive));
  return a0.index==a1.index;
}

bool arc_is_repeated_vertex(Arcs arcs, const Vertex& v01, const Vertex& v12);

// Do two circles intersect (degree 2)?
OTHER_CORE_EXPORT bool circles_intersect(Arcs arcs, const int arc0, const int arc1);

// Is intersection (a0,a1).y < (b0,b1).y?  We require a0==b0, which allows a degree 8 implementation.
// If add = true, check whether ((0,a1)+(0,b1)).y > 0.
template<bool add=false> bool circle_intersections_upwards(Arcs arcs, const Vertex a, const Vertex b);

// Tests if an arc segment is less then a half circle
bool circle_intersections_ccw(Arcs arcs, const Vertex v0, const Vertex v1);

// Does the piece of a1 between a01 and a12 intersect the piece of b1 between b01 and b12 (degree 6)?  a1 and b1 are assumed to intersect at ab.
bool circle_arcs_intersect(Arcs arcs, const Vertex a01, const Vertex a12,
                                      const Vertex b01, const Vertex b12,
                                      const Vertex ab);

// Does the (a1,b) intersection occur on the piece of a1 between a0 and a2 (degree 6)?  a1 and b are assumed to intersect.
bool circle_arc_intersects_circle(Arcs arcs, const Vertex a01, const Vertex a12, const Vertex a1b);

// Construct both of the intersections of two circular arcs, assuming they do intersect.
// The two intersections are to the right and the left of the center segment, respectively, so result[left] is correct.
// The results differ from the true intersections by at most 2.
// Degrees 3/2 for the nonsqrt part and 6/4 for the part under the sqrt.
Vector<Vertex,2> circle_circle_intersections(Arcs arcs, const int arc0, const int arc1);

// Compute a bounding box for the arc between two vertices
Box<exact::Vec2> arc_box(Arcs arcs, const Vertex& v01, const Vertex& v12);

// Precompute all intersections.  result[i] is the start of arcs[i]
Array<Vertex> compute_vertices(Arcs arcs, RawArray<const int> next);

// An intersection between an arc and a horizontal line
struct HorizontalVertex {
  int arc;
  bool left; // True if the intersection is the left of the *vertical* line through the arc's center
  uint8_t q0; // Quadrant relative to arc's center
  Quantized y; // y-coordinate of horizontal line
  Interval x; // x-coordinate of intersection

  bool operator==(const HorizontalVertex h) const {
    assert(y==h.y);
    return arc==h.arc && left==h.left;
  }

  bool operator!=(const HorizontalVertex v) const {
    return !(*this==v);
  }

  OTHER_UNUSED friend ostream& operator<<(ostream& output, const HorizontalVertex h) {
    return output<<format("(%d,%s,%c)",h.arc,repr(h.y),h.left?'L':'R');
  }
};

// Does a circle intersect a horizontal line (degree 1)?
bool circle_intersects_horizontal(Arcs arcs, const int arc, const Quantized y);

// Assuming the circle intersects the horizontal line, generate structs for each intersection.
Vector<HorizontalVertex,2> circle_horizontal_intersections(Arcs arcs, const int arc, const Quantized y);

// Does the piece of a1 between a01 and a12 contain the given horizontal intersection (degree 3)?
bool circle_arc_contains_horizontal_intersection(Arcs arcs, const Vertex a01, const Vertex a12, const HorizontalVertex a1y);

// Are the two horizontal circle intersections in rightwards order (degree 4)?  We require ay.y==by.y.
bool horizontal_intersections_rightwards(Arcs arcs, const HorizontalVertex ay, const HorizontalVertex by);

// Compute winding(local_outside) - winding(rightwards), where local_outside is immediately outside of h.arc and rightwards
// is immediately to the right of h.  Thus, the result will be either 0 or -1, since locally winding(local_outside) = 0 and winding(rightwards) = 0 or 1.
static inline int local_horizontal_depth(Arcs arcs, const HorizontalVertex h) {
  return -(arcs[h.arc].positive==h.left);
}

// Count the depth change along the horizontal ray at the given intersection with an arc.
// The change is -1 if we go out of an arc, +1 if we go into an arc.
static inline int horizontal_depth_change(Arcs arcs, const HorizontalVertex h) {
  return arcs[h.arc].positive==h.left ? 1 : -1;
}

} // namespace other
