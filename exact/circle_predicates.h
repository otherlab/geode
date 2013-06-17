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

  static Quantized tolerance() { return 2; }

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
// an absolute disaster, as wouldn't have flipped the sign of the symbolic perturbation.
static inline exact::Point3 aspoint(Arcs arcs, const int arc) {
  const auto& a = arcs[arc];
  return tuple(a.index,exact::Vec3(a.center,a.radius));
}
static inline exact::Point2 aspoint_center(Arcs arcs, const int arc) {
  const auto& a = arcs[arc];
  return tuple(a.index,a.center);
}

// Check if arcs are both parts of the same circle
bool arcs_from_same_circle(const Arcs& arcs, int i0, int i1);

// Do two circles intersect (degree 2)?
OTHER_CORE_EXPORT bool circles_intersect(Arcs arcs, const int arc0, const int arc1);

// Is intersection (a0,a1).y < (b0,b1).y?  This is degree 20 as written, but can be reduced to 6.
bool circle_intersections_upwards(Arcs arcs, const Vertex a, const Vertex b);

// Does the piece of a1 between a0 and a1 intersect the piece of b1 between b0 and b2?  a1 and b1 are assumed to intersect.
bool circle_arcs_intersect(Arcs arcs, const Vertex a01, const Vertex a12,
                                      const Vertex b01, const Vertex b12,
                                      const Vertex ab);
// Does the (a1,b) intersection occur on the piece of a1 between a0 and a2?  a1 and b are assumed to intersect.
bool circle_arc_intersects_circle(Arcs arcs, const Vertex a01, const Vertex a12, const Vertex a1b);

// Construct both of the intersections of two circular arcs, assuming they do intersect.
// The two intersections are to the right and the left of the center segment, respectively, so result[left] is correct.
// The results differ from the true intersections by at most 2.
// Degrees 3/2 for the nonsqrt part and 6/4 for the part under the sqrt.
Vector<Vertex,2> circle_circle_intersections(Arcs arcs, const int arc0, const int arc1);

// Compute a bounding box for the arc between two verticies
Box<exact::Vec2> arc_box(RawArray<const ExactCircleArc> arcs, const Vertex& v01, const Vertex& v12);

// Precompute all intersections
// result[i] is the start of arcs[i]
Array<Vertex> compute_verticies(RawArray<const ExactCircleArc> arcs, RawArray<const int> next);

// Compute winding(local_outside) - winding(rightwards), where local_outside is immediately outside of a12 and rightwards
// is far to the right of a12, taking into account only arcs a1 and a2.  Thus, ignoring secondary intersections with arcs a1 and a2,
// the result will be either 0 or -1, since locally winding(local_outside) = 0 and winding(rightwards) = 0 or 1.
int local_x_axis_depth(Arcs arcs, const Vertex a01, const Vertex a12, const Vertex a23);

// Count the depth change along the horizontal ray from (a0,a1) to (a0,a1+(inf,0) due to the arc from (b0,b1) to (b1,b2).
// The change is -1 if we go out of an arc, +1 if we go into an arc.  Degree 8 as written, but can be eliminated entirely.
int horizontal_depth_change(Arcs arcs, const Vertex a, const Vertex b01, const Vertex b12);

} // namespace other
