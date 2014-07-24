#include <geode/geometry/ArcSegment.h>
#include <geode/geometry/Segment.h>
#include <geode/structure/Tuple.h>

namespace geode {

Rotation<Vec2> ArcSegment::d_to_tangent_rotation() const {
  // For d = 0.5*(x1 - x0); // Vector to/from midpoint to ends
  // Angle d_to_t between d and tangent = arc_angle / 2
  // We also know that q = tan(arc_angle/4) = tan(d_to_t / 2) = sin(d_to_t) / (1 + cos(d_to_t))
  // Solve to get:
  //   sin(d_to_t) =         2*q / (1 + q_sqr)
  //   cos(d_to_t) = (1 - q_sqr) / (1 + q_sqr)
  // const real q_sqr = sqr(q);
  // return Matrix<real,2>(1-q_sqr,2*q,-2*q,1-q_sqr) / (1 + q_sqr);
  return Rotation<Vec2>::from_complex(std::complex<real>(1-sqr(q),2*q)); // We ignore the 1/(1+q_sqr) term since the rotation will be normalized anyway
}

Box<Vec2> ArcSegment::bounding_box() const {
  // Result will at least contain both endpoints
  Box<Vec2> result = geode::bounding_box(x0, x1);

  const real r = this->r();
  if(!isfinite(r)) {
     // If r isn't finite, we approximate arc as a straight line and just use endpoints
  }
  else {
    const Vec2 arc_center = this->arc_center();
    const Vec2 normal = d_perp() * sign(q);
    for(int i = 0; i < 4; ++i) {
      const Vec2 compass_pt = arc_center + rotate_left_90_times(Vec2(r,0),i);
      if(dot(compass_pt - x0, normal) >= 0) {
        result.enlarge_nonempty(compass_pt);
      }
    }
  }
  return result;
}

real ArcSegment::distance_to_any_part_of_circle(const Vec2 p) const {
  const Vec2 m = arc_mid();
  const Vec2 mp = p - m;
  const real w_sqr = mp.sqr_magnitude();
  //const real al = dot(mp, d());
  const real bl = dot(mp, d_perp());
  const real l_sqr = d().sqr_magnitude();
  const real l = sqrt(l_sqr);
  const real f = this->c_times_l();
  const int sign_bl = sign(bl);
  
  const real temp = (2*bl + f*w_sqr);
  return sign_bl * temp / (l + sqrt(l_sqr + f * temp));
}

real ArcSegment::arc_length() const {
  const real p_over_l = ((sqr(q) + 1)*2.*atan(q)/q);
  if(!isfinite(p_over_l)) {
    assert(abs(q) < 1e-6);
    return l() * 2.; // assume we go to the limit as q approaches zero
  }
  else {
    return l() * p_over_l;
  }
}

real ArcSegment::distance_to_arc(const Vec2 p) const {
  if(abs(q) > 0.4) { // If curvature of arc is significant, we should be able to safely construct center
    const Vec2 e = arc_center();
    const Vec2 projected = e + normalized(p - e)*r();
    const real s_intercept = dot(projected-x0, d_perp()); // Check which side of chord projected point ends up on
    if(s_intercept*sign(q) < 0.) {
      // If projected point ends up on wrong side of chord from x0 to x1 it is on part of the circle not in the arc, so endpoint will be best
      return closest_endpoint_distance(p);
    }
    else {
      return (projected - p).magnitude();
    }
  }
  else if(q == 0) { // q will often be set to zero so we will handle that case seperately
    auto closest = Segment<Vec2>(x0,x1).closest_point(p).x;
    return (p-closest).magnitude();
  }
  else { // If arc has low curvature we need to be more careful
    
    assert(abs(q) < 1); // assert that angle is less than pi so we don't need to check special cases

    // Use tangent directions to check if closest point is an endpoint or somewhere in the middle of the arc
    if((dot(t0(), p - x0) <= 0) || (dot(t1(), p - x1) > 0)) {
      // || cross(t1, p - x1) <= 0) ^ angle_greater_than_pi) {
      return closest_endpoint_distance(p);
    }
    else {
      return distance_to_any_part_of_circle(p);
    }
  }
}

Vec2 arc_t0_vector(const Vec2 x0, const Vec2 x1, const real q) {
  return ArcSegment({x0,x1,q}).t0();
}
Vec2 arc_t1_vector(const Vec2 x0, const Vec2 x1, const real q) {
  return ArcSegment({x0,x1,q}).t1();
}

real point_to_arc_distance(const Vec2 p, const Vec2 x0, const Vec2 x1, const real q) {
  const auto e = ArcSegment({x0,x1,q});
  return e.distance_to_arc(p);
}

Vec2 arc_center(const Vec2 x0, const Vec2 x1,  const real q) {
  return ArcSegment({x0,x1,q}).arc_center();
}

real arc_angle(const Vec2 x0, const Vec2 x1,  const real q) {
  return ArcSegment({x0,x1,q}).angle();
}

real arc_length(const Vec2 x0, const Vec2 x1,  const real q) {
  return ArcSegment({x0,x1,q}).arc_length();
}

real arc_curvature(const Vec2 x0, const Vec2 x1,  const real q) {
  return ArcSegment({x0,x1,q}).c();
}

Box<Vec2> arc_bounding_box(const Vec2 x0, const Vec2 x1, const real q) {
  return ArcSegment({x0,x1,q}).bounding_box();
}

} // namespace geode
