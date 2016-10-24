// Set to 1 to enable checks in quantization that ensure all tolerances were met
#define CHECK_CONSTRUCTIONS 0

#include <geode/exact/circle_quantization.h>
#include <geode/exact/Exact.h>
#include <geode/exact/math.h>
#include <geode/exact/perturb.h>

#if CHECK_CONSTRUCTIONS
#include <geode/exact/circle_objects.h>
#endif

namespace geode {

Quantizer<real,2> make_arc_quantizer(const Box<Vector<real,2>> arc_bounds) {
  // Enlarge box quite a lot so that we can closely approximate lines.
  // The error in approximating a straight segment of length L by a circular arc of radius R is
  //   er = L^2/(8*R)
  // If the maximum radius is R, the error due to quantization is
  //   eq = R/bound
  // Equating these, we get
  //   R/bound = L^2/(8*R)
  //   R^2 = L^2*bound/8
  //   R = L*sqrt(bound/8)
  const real max_radius = sqrt(exact::bound/8)*arc_bounds.sizes().max();
  return quantizer(arc_bounds.thickened(max_radius));
}

// Return integers num,denom both with magnitude less than or equal to exact::bound such that num/denom approximately equals x.
// If abs(x) is very large result may be clamped to +/- exact::bound
static Vector<Exact<1>,2> rational_approximation(const real x) {
  assert(!isnan(x));

  // x == x/1 == x*denom / denom == num / denom
  // We want to choose denom as big as possible such that abs(x)*denom <= exact::bound
  const real scale = round(exact::bound / max(1.,ceil(abs(x))));
  assert(0 <= scale && scale <= exact::bound); // scale is exact::bound divided by some value >= 1, so must be <= exact::bound
  const ExactInt denom = ExactInt(scale);
  const real approx_num = round(x*scale);
  // Since scale*ceil(abs(x)) <= exact::bound we know scale*abs(x) <= exact::bound
  // Thus abs(approx_num) will be <= exact::bound
  assert(abs(approx_num) <= exact::bound);
  if(denom > 0) {
    assert(sign(approx_num) == sign(x) || approx_num == 0);
    return vec(Exact<1>(approx_num), Exact<1>(denom));
  }
  else {
    // If abs(x) is on the order of 0.5*exact::bound or larger, denom could have been rounded to zero
    // Compute a numerator in the valid range and use 1 for the denom
    const ExactInt safe_num = ExactInt(clamp(round(x), real(-exact::bound), real(exact::bound)));
    assert(sign(safe_num) == sign(x));
    return vec(Exact<1>(safe_num), Exact<1>(1));
  }
}

#if CHECK_CONSTRUCTIONS
static bool test_circles_intersect(const Vector<Quantized,2> c0, const Quantized r0, const Vector<Quantized,2> c1, const Quantized r1) {
  typedef ExactCircle<Perturbation::Implicit> CircleImp;
  typedef ExactCircle<Perturbation::Explicit> CircleExp;
  
  const bool intersect_imp = has_intersections(CircleImp(c0, r0), CircleImp(c1, r1));
  const bool intersect_exp = has_intersections(CircleExp(c0, r0, 0), CircleExp(c1, r1, 1));
  GEODE_ASSERT(intersect_imp == intersect_exp);
  return intersect_imp && intersect_exp;
}
#endif

// A circle that is centered at x0 or x1 with a radius of constructed_arc_max_endpoint_error() will always intersect
// a circle with radius and center as returned from construct_circle_radius_and_center(x0,x1,q)
Quantized constructed_arc_endpoint_error_bound() { return 2; } // ceil(sqrt(2)/2)

// Returns a center and radius for a circle that passes within constructed_arc_endpoint_error_bound() units of each quantized vertex and has approxamently the correct curvature
// x0 and x1 should be integer points (from quantizer)
// WARNING: If endpoints have been quantized to the same point, a radius of 0 (invalid for an ExactCircleArc) will be returned to indicate no arc is needed
// As long as x0 != x1, a circle of radius constructed_arc_endpoint_error_bound() centered at x0 or x1 will always intersect the returned circle
Tuple<Vector<Quantized,2>, Quantized> construct_circle_center_and_radius(const Vector<Quantized, 2> x0, const Vector<Quantized, 2> x1, const real q) {
  if(x0 == x1) {
    return tuple(x0, Quantized(0));
  }

  const Vector<Exact<1>,2> ex0(x0),
                           ex1(x1);

  const Vector<Exact<1>,2> delta = ex1 - ex0;

  const Exact<1> max_r = Exact<1>(exact::bound/8); // FIXME: I'm not convinced this is correct, but it will work for now

  const Vector<Exact<1>,2> q_fract = rational_approximation(q);
  assert(sign(q_fract.y) == 1);
  assert(sign(q_fract.x) == sign(q) || !is_nonzero(q_fract.x));

  const SmallShift exact_two = (One()<<1);
  const Vector<Exact<1>,2> d_perp = delta.orthogonal_vector();
  const Vector<Exact<3>,2> H_num = emul(sqr(q_fract.y) - sqr(q_fract.x), d_perp);
  const Exact<2> H_den_times_half = q_fract.y * q_fract.x * exact_two;
  const Exact<2> H_den = H_den_times_half * exact_two;
  const Exact<3> H_num_l1 = H_num.L1_Norm();

  // Compare length of H (using l-one norm for simplicity) and max_r to ensure we aren't going to generate a center that is out of bounds
  const bool must_scale_H = (H_num_l1 >= max_r * abs(H_den));

  Vector<Quantized,2> center;
  if(must_scale_H) {
    // If H is too big, we need to scale it so that biggest component is <= max_r
    // We need: (H_num_l_inf / abs(H_den)) * s = max_r
    // Solve for s: s = (max_r * abs(H_den)) / H_num_l1
    // Plug in s: H * s = (H_num / H_den) * ((max_r * abs(H_den)) / H_num_l1)
    // H_den cancels except for sign and we have: H_prime = ((H_den < 0 : -1 : 1) * max_r * H_num) / H_num_l1
    // C = (0.5 * (x0 + x1)) + H_prime
    // We use sign of q in case rational_approximation rounded it to zero
    assert(!is_nonzero(H_den) || is_negative(H_den) == signbit(q));
    center = snap_div(emul((signbit(q) ? -max_r : max_r)*exact_two, H_num) + emul(H_num_l1, ex0 + ex1), H_num_l1*exact_two, false);
  }
  else {
    // Otherwise we do:
    // C = (0.5 * (x0 + x1)) + H
    // Combine into a single fraction and snap
    center = snap_div(H_num + emul(H_den_times_half, ex0 + ex1), H_den, false);
  }

  assert(is_nonzero(H_den) || must_scale_H);

  // Since C is exact up to rounding, we know that distance to endpoints will differ by at most sqrt(2)
  // Instead of multiple calls to snap_div, we just take average of squared distances to each endpoint
  // This value will be at or between the two options and is symmetric with respect to x0 and x1
  auto ecenter = Vector<Exact<1>,2>(center);
  const Exact<2> sum_dist_sqr = esqr_magnitude(ecenter - ex0) + esqr_magnitude(ecenter - ex1);
  Quantized r = snap_div(sum_dist_sqr, Exact<1>(2), true);

  // We need a circle centered at x0 or x1 with radius == constructed_arc_endpoint_error_bound() to intersect the constructed circle
  // As computed above, center and r will ensure this happens except when r is too small and the constructed arc is fully inside the endpoint error circles
  if(r <= constructed_arc_endpoint_error_bound()) {
    // For an endpoint outside the circle, it is safe to increase r since that will decrease error to that endpoint
    // We only grow radius up to the error bound so it is impossible to overshoot by too much
    // Any endpoints that start inside the circle will be ok unless they are exactly at the center of the constructed arc
    if(ecenter == ex0 || ecenter == ex1) {
      // Center point was exactly on the line of points equidistant to x0 and x1
      // Rounding it moved it by at most sqrt(0.5) so distance from x0 to x1 is <= 2*sqrt(0.5) == sqrt(2)
      // We move center by the orthogonal delta between x0 and x1 to ensure we don't move it on top of the other endpoint
      center += ((q >= 0) ? (x1 - x0) : (x0 - x1)).orthogonal_vector();
      assert(sqrt(2) < constructed_arc_endpoint_error_bound());
      // Distance from center to endpoint that center was on top of will be equal to length of orthogonal vector which is >= 1 and <= sqrt(2)
      // Distance from center to other endpoint will be sqrt(2) * length of orthogonal vector <= sqrt(2)*sqrt(2) == 2
      // Since we used the orthogonal vector, we won't have moved center on top of the other endpoint
    }
    r = constructed_arc_endpoint_error_bound();
  }


#if CHECK_CONSTRUCTIONS
  GEODE_WARNING("Expensive circle construction tests enabled");
  const auto est_mid = 0.5*((x0 + x1) + q*rotate_right_90(x1 - x0));

  GEODE_ASSERT(test_circles_intersect(center, r, x0, constructed_arc_endpoint_error_bound()));
  GEODE_ASSERT(test_circles_intersect(center, r, x1, constructed_arc_endpoint_error_bound()));

  // Limits on maximum radius can result in larger errors for midpoint of straight (or almost so) arcs (when q is close to zero)
  // If we were forced to scale H we need to allow a larger error margin to account for added curvature.
  // I haven't been able to work out a guaranteed error bound at the midpoint, but found no errors larger than 6 units during testing.
  GEODE_ASSERT(test_circles_intersect(center, r, vec(round(est_mid.x),round(est_mid.y)), constructed_arc_endpoint_error_bound()*(must_scale_H ? 3 : 1)));
#endif

  return tuple(center, r);
}

} // namespace geode

