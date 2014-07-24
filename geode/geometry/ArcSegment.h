#pragma once
#include <geode/vector/Vector.h>
#include <geode/vector/Rotation.h>
#include <geode/geometry/Box.h>

namespace geode {

// This struct contains various utility functions for circular arcs represented by the two endpoints x0,x1 and q = 2*sagitta/|x1-x0|.
// q > 0 is counterclockwise, q < 0 is clockwise, q == 0 is a straight line
// q == 1 will be a counterclockwise half circle, q == -1 will be a clockwise half circle
struct ArcSegment {
  Vec2 x0, x1; // These are the start and end points of the arc
  real q; // This value is 2*sagitta/|x1-x0|

  ArcSegment(const Vec2 _x0, const Vec2 _x1, const real _q) : x0(_x0), x1(_x1), q(_q) {}

  Vec2 x_avg() const { return 0.5*(x0 + x1); } // Midpoint on straight line between ends
  Vec2 d() const { return 0.5*(x1 - x0); } // Delta vector between x_avg and endpoints
  Vec2 d_perp() const { return rotate_right_90(d()); }
  Vec2 arc_mid() const { return x_avg() + q*d_perp(); } // Midpoint on arc segment
  real l() const { return d().magnitude(); }
  real c_times_l() const { return 2*q / (1 + sqr(q)); }

  real arc_length() const; // Length along perimeter of arc
  real angle() const { return 4.*atan(q); }
  real half_q() const { return q / (1 + sqrt(1 + sqr(q))); } // q value for ArcSegments between endpoints and arc_mid
  real c() const { return c_times_l() / l(); } // Curvature of arc (unstable if l = 0)

  Rotation<Vec2> d_to_tangent_rotation() const; // Rotation between direction of d and tangent of arc at x1

  Vec2 t0() const { return d_to_tangent_rotation().inverse_times(d()); } // Unnormalized tangent vector at x0 (Might have zero length for degenerate arcs)
  Vec2 t1() const { return d_to_tangent_rotation() * d(); } // Unnormalized tangent vector at x1 (Might have zero length for degenerate arcs)

  Box<Vec2> bounding_box() const; // This is susceptable to floating point rounding (i.e. arc may extend a very small distance outside of box)

  real closest_endpoint_distance(const Vec2 p) const { return sqrt(min((p-x0).sqr_magnitude(),(p-x1).sqr_magnitude())); }
  Vec2 closest_endpoint(const Vec2 p) const { return (p-x0).sqr_magnitude() <= (p-x1).sqr_magnitude() ? x0 : x1; }

  real distance_to_any_part_of_circle(const Vec2 p) const;
  real distance_to_arc(const Vec2 p) const;

  // These functions are unstable when q is close to zero:
  real r() const { return abs(l() / c_times_l()); } // Multiply by l last so that repeated points have radius 0
  real h_over_l() const { return (sqr(q) - 1)/(2*q); }
  Vec2 arc_center() const { return x_avg() + d_perp()*h_over_l(); }
};

real arc_angle(const Vec2 x0, const Vec2 x1, const real q);
real arc_length(const Vec2 x0, const Vec2 x1, const real q);
real arc_curvature(const Vec2 x0, const Vec2 x1, const real q);
Vec2 arc_center(const Vec2 x0, const Vec2 x1, const real q); // Unstable/undefined if q is close to zero
Box<Vec2> arc_bounding_box(const Vec2 x0, const Vec2 x1, const real q);
Vec2 arc_t0_vector(const Vec2 x0, const Vec2 x1, const real q);
Vec2 arc_t1_vector(const Vec2 x0, const Vec2 x1, const real q);
real point_to_arc_distance(const Vec2 p, const Vec2 x0, const Vec2 x1, const real q);

} // namespace geode
