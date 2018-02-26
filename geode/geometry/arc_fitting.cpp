#include "arc_fitting.h"
#include <geode/math/constants.h>
#include <geode/python/wrap.h>
#include <geode/utility/prioritize.h>

#include <queue>

namespace geode {

static void add_discretized_arc_low_curvature(Array<Vec2>& result, const Vec2 p1, const real q, const real max_deviation) {
  // assorted arc parameters
  const real q_sqr = sqr(q);
  const Vec2 p0 = result.back();
  const Vec2 d = 0.5*(p1 - p0); // delta
  const Vec2 d_perp = rotate_right_90(d);
  const Vec2 p_avg = 0.5*(p0 + p1);
  const real l = d.magnitude();
  const real c = 2*q / (l*q_sqr + l);
  const Vec2 d_perp_unit = d_perp.normalized();

  // As the curvature of an arc approaches 0, the center becomes numerically unstable
  // Instead of constructing the center and sampling and center + polar(theta)*r, this function samples along the axis from p0 to p1
  // The sampling won't be linear, but is probably close enough

  // To stay under a given error we need to sample with small enough segments:
  // max_allowed_seg_length = sqrt(max_deviation * (2*r - max_deviation))

  // seg_length <= c0*chord_step_size
  const real c0 = (1 + q_sqr) / (1 - q_sqr); // abs(cosine) between d and the arc's tangent at p0

  // chord_step_size = 2*l / num_steps;
  // For the worst case: seg_length = c0*chord_step_size which needs to be <= max_allowed_seg_length
  // c0*chord_step_size <= sqrt(max_deviation * (2*r - max_deviation))
  // c0*2*l / num_steps <= sqrt(max_deviation * (2*r - max_deviation))
  // num_steps >= c0*2*l / sqrt(max_deviation * (2*r - max_deviation))
  // To avoid numerical issues, use lc == l/r --> r == l / lc
  // num_steps >= c0*2*l / sqrt(max_deviation * (2*(l/lc) - max_deviation))
  // num_steps >= c0*2*l*sqrt(1. / (max_deviation*2*l/lc - sqr(max_deviation)))
  //   expand lc = 2*q / (sqr(q) + 1);
  // num_steps >= c0*2*l*sqrt(q / ((sqr(q) + 1)*max_deviation*l - sqr(max_deviation)*q))
  const real root = abs(q) / ((sqr(q) + 1)*max_deviation*l - sqr(max_deviation)*abs(q));
  const int num_steps = (root > 0) ? ceil(2*c0*l*sqrt(root)) : 1;

  const real du = 2./num_steps; // we will paramatrize and sample from u = -1 to u = 1

  const Vec2 arc_mid = p_avg + q*d_perp;

  for(int i = 1; i < num_steps; ++i) { // start at i = 1 since p0 is already added
    const real u = -1 + i*du;
    const real x2 = sqr(u*l);
    const real y = -c*x2 / (1 + sqrt(1 - x2*sqr(c)));

    const Vec2 p_u = arc_mid + u*d + y*d_perp_unit;

    result.append(p_u);
  }
  result.append(p1); // use original point for end
}

static void add_subdivided_arc(Array<Vec2>& result, const Vec2 p1, const real q, const real max_deviation) {
  assert(!result.empty());
  assert(isfinite(q));
  GEODE_ASSERT(isfinite(q)); // This is a hard assert to avoid infinite recursion
  const Vec2 p0 = result.back();
  const Vec2 d = 0.5*(p1 - p0); // delta
  const real l = d.magnitude();

  if(l < max_deviation) {
    result.append(p1);
  }
  else if(abs(q) < sqrt(1./3.)) { // For less than a third of an arc, low curvature will work
    add_discretized_arc_low_curvature(result, p1, q, max_deviation);
  }
  else {
    // If curvature is high could explicitly compute center, but just subdividing is more robust
    // subdivide and recurse
    const Vec2 m = 0.5*(p0 + p1); // mid point
    const Vec2 arc_mid = m + q*rotate_right_90(d);
    const real half_q = q / (1 + sqrt(1 + sqr(q)));
    add_subdivided_arc(result, arc_mid, half_q, max_deviation);
    add_subdivided_arc(result,      p1, half_q, max_deviation);
  }
}

Array<Vec2> discretize_arcs(const RawArray<const CircleArc> arc_points, const bool closed, const real max_deviation) {
  Array<Vec2> result;
  if(arc_points.empty()) return result;
  result.append(arc_points.front().x);
  for(int i : range(1,arc_points.size())) {
    const auto next_q = arc_points[i - 1].q;
    const auto next_x = arc_points[i].x;
    add_subdivided_arc(result, next_x, next_q, max_deviation);
  }
  if(closed) {
    add_subdivided_arc(result, arc_points.front().x, arc_points.back().q, max_deviation);
  }
  return result;
}

Nested<Vec2> discretize_nested_arcs(const Nested<const CircleArc> arc_points, const bool closed, const real max_deviation) {
  Nested<Vec2, false> result;
  for(const auto& p : arc_points) {
    result.append(discretize_arcs(p, closed, max_deviation));
  }
  return result.freeze();
}

real fit_q(const Vec2 p0, const Vec2 p1, const Vec2 p3) {
  const Vec2 A = (p3-p0), B = (p3-p1); // legs for triangle circumscribed by this arc
  const real inscribed_angle = angle_between(A,B);
  const real arc_angle =  2 * (pi - inscribed_angle);
  return -tan(0.25 * arc_angle); // q = tan(arc_angle / 4)
}

// Some conservative approximations are used (error will be within given bounds, but some q values outside of bounds might still be valid)
Box<real> fit_q_range(const Vec2 p0, const Vec2 p1, const Vec2 p3, const real allowed_error) {
  const Vec2 A = (p3-p0), B = (p3-p1); // legs for triangle circumscribed by this arc
  const real a = A.magnitude(), b = B.magnitude();

  // Can't do worse than distance to an endpoint regardless of q
  if(a <= allowed_error || b <= allowed_error) {
    return Box<real>::full_box();
  }

  const real inscribed_angle = angle_between(A,B);

  const Vec2 bisection_dir = polar(angle(A) + 0.5*inscribed_angle); // Gives perpendicular in case A and B are antiparallel unlike (a*B + b*A).normalized();

  // Maximum deviation will be approxamently along bisection direction. Assume it is and sample two points
  const Vec2 p3_in = p3 + bisection_dir*allowed_error;
  const Vec2 p3_out = p3 - bisection_dir*allowed_error;

  const real fit_in = fit_q(p0, p1, p3_in);
  const real fit_out = fit_q(p0, p1, p3_out);

  return bounding_box(fit_in, fit_out);
}

// Longer straight segments aren't approximated well by high curvature arcs.
// Each straight segment deviates from it's arc by the sagitta.
// This function computes range of curvature that will stay within an acceptable tolerance
// RESULTS ARE APPROXIMATE WHEN USED FOR FITTING SINCE IT ASSUMES EACH SEGMENT IS EXACTLY ON ARC!!!
static real abs_allowed_arc_curvature(const real allowed_error, const real l_sq_max) {
  // l_sq_max = sqr(longest segment_length / 2)
  // abs(c) <=  allowed_arc_curvature
  // for simplicity this assumes that the start and end of each segment are exactly on the arc
  // if abs(c) > allowed_arc_curvature, deviation will be greater than error
  // actual allowed curvature is from -abs_allowed_arc_curvature to +abs_allowed_arc_curvature
  return 2*allowed_error / (sqr(allowed_error) + l_sq_max);
}

static Box<real> q_curvature_based_limits(const real l, const real allowed_error, const real max_seg_l_sq) {
  const real c_max = abs_allowed_arc_curvature(allowed_error, max_seg_l_sq);
  const real lcm = (l*c_max);
  const real disc = 1 - sqr(lcm);

  if(disc < 0) {
    return Box<real>::full_box();
  }

  const real sqrt_disc = sqrt(disc);
  assert(sqrt_disc <= 1);
  // q values that give max allowed curvature will be at +/-lcm / (1 +/- sqrt_disc)
  // An arc with a very large abs(q) will have small curvature, but is a giant circle far from the segment we are trying to fit
  // We want the range of values that approximate a straight line
  // With some simplification: lcm / (1 - sqrt_disc) == (1 + sqrt_disc) / lcm
  // In order these should be:
  // const real q_crit_0 = -(1 + sqrt_disc) / lcm; // == -lcm / (1 - sqrt_disc)
  // const real q_crit_1 = -lcm / (1 + sqrt_disc);
  // const real q_crit_2 =  lcm / (1 + sqrt_disc);
  // const real q_crit_3 =  (1 + sqrt_disc) / lcm; // == lcm / (1 - sqrt_disc)
  // Values between q_crit_0 and q_crit_1 give negative curvature that is too low (abs value too high)
  // Values between q_crit_2 and q_crit_3 give positive curvature that is too high
  // We keep the range of values between q_crit_1 and q_crit_0 which are sufficiently close to flat
  const real q_small = lcm / (1 + sqrt_disc); // == +/- q_crit_2/1
  return Box<real>{-q_small, q_small};
}

static real longest_seg_sq(const RawArray<const Vec2> points, const int start_point, const int count) {
  real result = 0;
  for(int i : range(start_point, start_point + count)) {
    const Vec2 prev = points[wrap(i    , points.size())];
    const Vec2 next = points[wrap(i + 1, points.size())];
    const real mag_sq = (next - prev).sqr_magnitude();
    if(mag_sq > result) result = mag_sq;
  }
  return result;
}

// find acceptable range of q values for an arc from points[start_point] and points[(start_point + count - 1) % points.size()]
// each point in between will be <= allowed_error from the arc
// will only use q in [-1,1] (i.e. will only fit half circles or less)
static Box<real> fit_q_range(const RawArray<const Vec2> points, const real allowed_error, const int start_point, const int count) {

  const Vec2 p0 = points[start_point], p1 = points[wrap(start_point + count, points.size())];
  const Vec2 u = (p1 - p0);
  const real l = 0.5 * u.magnitude();
  Box<real> result = q_curvature_based_limits(l, allowed_error, longest_seg_sq(points, start_point, count));

  // dot(p1, u) - dot(p0, u) = dot(p1 - p0, u) = dot(u,u) = u.magnitude_sq() > 0
  const auto u_range = Box<real>{dot(p0, u), dot(p1, u)};
  assert(u_range.min <= u_range.max);

  // We limit fitting to half circles since numerical accuracy of circle arcs starts to degrade for larger arc angles
  // Limiting to half circles also makes it easier to check that arc doesn't have any 'kinks' by ensuring dot(p,u) is monotonic)
  result = Box<real>::intersect(result, Box<real>{-1.,1.});

  real last_u = u_range.min;
  //loop over the points between p0 and p1
  for(int c : range(start_point + 1, start_point + count)) {
    const int i = wrap(c, points.size());
    const Vec2& p = points[i];
    const real new_u = dot(p, u);
    if(new_u < last_u) // Check if projection onto chord is monotonic
      return Box<real>::empty_box(); // If not, there's a 'kink' where points go backwards relative to arc which we probably want to avoid
    last_u = new_u;
    result = Box<real>::intersect(result, fit_q_range(p0, p1, p, allowed_error));
    if(result.empty()) break;
  }
  // Bonus check needed for final point
  if(last_u > u_range.max) // Check if projection onto chord is monotonic
    return Box<real>::empty_box();

  return result;
}

static real best_q(const RawArray<const Vec2> points, const int start_point, const int count, const real allowed_error, const Box<real> allowed_q) {
  const Vec2 p0 = points[start_point], p1 = points[wrap(start_point + count, points.size())];
  if(count <= 1) return 0; // use a straight line if only 2 points

  real prev_l = (points[wrap(start_point + 1, points.size())] - points[start_point]).magnitude();
  real sum_q = 0.;
  real sum_weights = 0.;

  Box<real> q_values = Box<real>::empty_box();
  Box<real> safe_q = Box<real>::full_box();

  //loop over the points between p0 and p1
  for(int c : range(start_point + 1, start_point + count)) {
    const int i = wrap(c, points.size());
    const Vec2& p3 = points[i];
    const real next_l = (points[wrap(c+1, points.size())] - p3).magnitude();

    const Box<real> q_range = fit_q_range(p0, p1, p3, allowed_error);
    if(q_range != Box<real>::full_box()) { // only operate on points that don't have degenerate geometry
      safe_q = Box<real>::intersect(safe_q, q_range);
      const real q = fit_q(p0, p1, p3);
      q_values.enlarge(q);
      const real w = prev_l + next_l;
      sum_q += q*w;
      sum_weights += w;
    }
    prev_l = next_l;
  }

  const real target_q = (sum_weights == 0.) ? 0. : sum_q / sum_weights;
  const real closest_q = allowed_q.clamp(target_q);
  return closest_q;
}

namespace {
  struct Cluster {
    int start, count;
    Cluster *prev, *next;
    Box<real> q;
    bool erased;

    void unlink() {
      if(prev) prev->next = next;
      if(next) next->prev = prev;
      prev = 0;
      next = 0;
      q = Box<real>::empty_box();
      erased = true;
    }
  };
}

Array<CircleArc> fit_arcs(const RawArray<const Vec2> points, real allowed_error, bool closed) {
  //const real min_r = 0.5*(max_error + sqr(l)/max_error);

  typedef Prioritize<Cluster*, int> MergeOp; // cluster and size of merged cluster
  std::priority_queue<MergeOp, std::vector<MergeOp>, std::greater<MergeOp> > queue;

  const int num_seed_clusters = max(0, points.size()-!closed);
  Array<Cluster> clusters(num_seed_clusters);
  for(const int i : range(clusters.size())) {
    auto& c = clusters[i];
    c.start = i;
    c.count = 1;
    c.prev = &c - 1;
    c.next = &c + 1;
    c.q = Box<real>::zero_box(); // only allow straight lines to start
    c.erased = false;
    queue.push(prioritize(&c, 2));
  }
  if(!clusters.empty()) {
    clusters.front().prev = closed ? &clusters.back() : 0;
    clusters.back().next = closed ? &clusters.front() : 0;
  }

  while(!queue.empty()) {
    const int op_count = queue.top().p;
    Cluster& lhs = *(queue.top().a);
    queue.pop();
    if(!lhs.next) continue; // if neighbor cluster unlinked then can't merge
    if(lhs.next == &lhs) continue; // make sure we don't try and merge cluster with itself if it wraps all the way around
    Cluster& rhs = *(lhs.next);
    const int new_count = lhs.count + rhs.count;
    if(new_count != op_count) continue; // if either cluster has been changed ignore this merge operation

    assert((lhs.start + lhs.count) % points.size() == rhs.start); // clusters ought to be adjacent
    assert(lhs.next == &rhs && rhs.prev == &lhs); // clusters should be linked

    const Box<real> new_range = fit_q_range(points, allowed_error, lhs.start, new_count);

    if(new_range.empty()) continue; // if no viable q values for merged cluster then ignore this merge op

    lhs.q = new_range;
    lhs.count = new_count;
    rhs.unlink();
    if(lhs.prev) queue.push(prioritize(lhs.prev, lhs.prev->count + lhs.count));
    if(lhs.next) queue.push(prioritize(&lhs, lhs.count + lhs.next->count));
  }

  Array<CircleArc> result;
  Cluster* last_cluster = 0;
  for(auto& c : clusters) {
    if(c.q.empty()) continue;
    assert(!last_cluster || last_cluster == c.prev);

    CircleArc new_arc;
    new_arc.x = points[c.start]; // arc will start here
    new_arc.q = best_q(points, c.start, c.count, allowed_error, c.q); // fit a single q value
    result.append(new_arc); // add the new point
    last_cluster = &c;
  }

  assert(points.empty() || last_cluster); // should have found at least one cluster if there were any points

  // for open arc polygons need to add the last vertex
  if(!closed && last_cluster) {
    // the last cluster should end at the last point
    const int final_point_i = last_cluster->start + last_cluster->count;
    assert(final_point_i == (points.size() - 1));
    CircleArc new_arc;
    new_arc.x = points[final_point_i];
    new_arc.q = 0.; // last q value will be ignored
    result.append(new_arc);
  }

  // If this is a closed polygon with at least one cluster, make sure the last arc connects back to the first
  assert(!(last_cluster && closed) || result.front().x == points[wrap(last_cluster->start + last_cluster->count, points.size())]);

  // NOTE: If the input is a degenerate polygon, result will have a single point (with zero q) in it
  return result;
}

Nested<CircleArc> fit_polyarcs(const Nested<const Vec2> polys, real allowed_error, bool closed) {
  Nested<CircleArc, false> result;
  for(const auto p : polys) {
    result.append(fit_arcs(p,allowed_error,closed));
  }
  return result.freeze();
}
} // geode namespace
using namespace geode;

void wrap_arc_fitting() {
  GEODE_FUNCTION(discretize_arcs)
  GEODE_FUNCTION(discretize_nested_arcs)
  GEODE_FUNCTION(fit_arcs)
  GEODE_FUNCTION(fit_polyarcs)
}
