#include "exact_circle_offsets.h"
#include <geode/exact/circle_quantization.h>
#include <geode/exact/Interval.h>
#include <geode/exact/scope.h>
namespace geode {

Quantized quantize_offset(const Quantizer<Quantized,2>& quant, const real d) {
  auto result = quant.quantize_length(d);
  if(result == 0 && d != 0) {
    // Offsets close to the quantization resolution are dangerous since errors in quantization, constructions, and unquantization can outweigh offset
    // This can cause bad things like offsetting by a negative amount causing geometry to grow
    // Any usage that triggers this warning should be sanity checked since errors from quantization and constructions will be larger than offsets
    GEODE_DEBUG_ONLY(GEODE_WARNING(format("Arc offset amount was below numerical representation threshold! Rounded to +/-%e", quant.inverse.unquantize_length(1.))));
    // This rounds small values away from zero to preserve sign of offset to provide a chance of better behavior
    result = (d > 0) ? 1 : -1;
  }
  return result;
}

#ifdef NDEBUG
#else
// For asserts to check that a value is actually quantized rather than some non-integer value
static bool is_quantized(const Quantized x) { return ExactInt(x) == x; }
static bool is_quantized(const exact::Vec2 v) { return is_quantized(v.x) && is_quantized(v.y); }
#endif

static Vec2 round(const Vec2 x) { return vec(std::round(x.x),std::round(x.y)); }

static void add_capsule_helper(ArcAccumulator<Pb::Implicit>& result, const ExactCircle<Pb::Implicit> c, const exact::Vec2 x0, const exact::Vec2 x1, const bool left_flags_safe, const bool prefer_full_circle, const Quantized signed_offset) {
  static constexpr Pb PS = Pb::Implicit;
  const Quantized abs_offset = abs(signed_offset);
  const ArcDirection capsule_sign = (signed_offset > 0) ? ArcDirection::CCW : ArcDirection::CW; // Final area of capsule should have same sign
  assert(is_quantized(c.center));
  assert(is_quantized(c.radius));
  assert(is_quantized(x0));
  assert(is_quantized(x1));
  assert(abs_offset != 0);

  const Quantized inner_r = c.radius - abs_offset;
  const Quantized outer_r = c.radius + abs_offset;

  // Tolerance is for each dimension, so we need to multiply by sqrt(2) to get distance error
  const Interval intersection_error_bound = assume_safe_sqrt(Interval(2))*ApproxIntersection::tolerance();
  // If we grow endcap for an arc by this amount, we ensure intersection or coverage of inner and outer offset arcs
  const Quantized endcap_safety_margin = ceil(intersection_error_bound.box().max); // Since offset circles don't need to construct new centers and have exactly correct radius, we only need to account for error of intersections

  if(!left_flags_safe) {
    if(prefer_full_circle) {
      result.add_full_circle(ExactCircle<PS>(c.center, outer_r), capsule_sign);
      if(inner_r > 0) {
        result.add_full_circle(ExactCircle<PS>(c.center, inner_r), -capsule_sign); // Subtract inside to get an annulus if needed
      }
    }
    else {
      // Use midpoint of vertices to minimize largest error
      const auto new_center = round(0.5*(x0 + x1));
      const Quantized endcap_r = abs_offset + 2*endcap_safety_margin;
      result.add_full_circle(ExactCircle<PS>(new_center, endcap_r), capsule_sign); // Approximate as a single circle
    }
  }
  else {
    const Quantized endcap_r = abs_offset + endcap_safety_margin;

    // Find or build circles for endcaps
    const auto src_cap = ExactCircle<PS>(x0, endcap_r);
    const auto dst_cap = ExactCircle<PS>(x1, endcap_r);
    const CircleId src_cap_cid = result.vertices.get_or_insert(src_cap);
    const CircleId dst_cap_cid = result.vertices.get_or_insert(dst_cap);

    IncidentId outer_src_cap; // endpoint of arc (if any) along outer circle at intersection with src_cap
    IncidentId dst_cap_outer; // endpoint of arc along dst_cap at outer circle (if any) or src_cap

    // Check if outer intersect endcaps to see if we need it
    // Endcap safety margin could (just barely and in very rare cases) make them fully enclose the outer offset arc
    // If this happens we can just use the endcaps

    const auto outer_c = ExactCircle<PS>(c.center, outer_r);

    if(has_intersections(outer_c, src_cap) && has_intersections(outer_c, dst_cap)) {
      const auto outer_src = outer_c.intersection_min(src_cap);
      const auto dst_outer = dst_cap.intersection_min(outer_c);
      const CircleId outer_cid = result.vertices.get_or_insert(outer_c);
      outer_src_cap = result.vertices.get_or_insert(outer_src, outer_cid, src_cap_cid);
      dst_cap_outer = result.vertices.get_or_insert(dst_outer, dst_cap_cid, outer_cid);
    }
    else {
      assert(has_intersections(src_cap, dst_cap)); // Outer should only be enclosed in cases where endcaps are intersecting
      const auto i = dst_cap.intersection_max(src_cap); // Use src_cap as a proxy for outer
      dst_cap_outer = result.vertices.get_or_insert(i, dst_cap_cid, src_cap_cid);
      // outer_src_cap will be invalid
    }

    IncidentId inner_dst_cap; // endpoint of arc (if any) along inner circle at intersection with dst_cap
    IncidentId src_cap_inner; // endpoint of arc along src_cap at inner circle (if any) or dst_cap

    const auto inner_c = ExactCircle<PS>(c.center, inner_r); // Warning: inner_r could be negative
    if((inner_r > 0) && has_intersections(inner_c, src_cap) && has_intersections(inner_c, dst_cap)) {
      const auto src_inner = src_cap.intersection_max(inner_c);
      const auto inner_dst = inner_c.intersection_max(dst_cap);
      const CircleId inner_cid = result.vertices.get_or_insert(inner_c);
      src_cap_inner = result.vertices.get_or_insert(src_inner, src_cap_cid, inner_cid);
      inner_dst_cap = result.vertices.get_or_insert(inner_dst, inner_cid, dst_cap_cid);
    }
    else {
      assert(has_intersections(src_cap, dst_cap)); // Should either have an inner circle or endcaps should intersect
      const auto i = src_cap.intersection_max(dst_cap); // Use dst_cap instead of inner
      src_cap_inner = result.vertices.get_or_insert(i, src_cap_cid, dst_cap_cid);
      // inner_dst_cap will be invalid
    }

    result.contours.start_contour();
    if(capsule_sign == ArcDirection::CCW) {
      if(outer_src_cap.valid()) result.append_to_back({outer_src_cap, ArcDirection::CCW});
                                result.append_to_back({dst_cap_outer, ArcDirection::CCW});
      if(inner_dst_cap.valid()) result.append_to_back({inner_dst_cap, ArcDirection::CW});
                                result.append_to_back({src_cap_inner, ArcDirection::CCW});
    }
    else {
      // To get the reversed capsule we have to reverse order of endpoints, switch reference circles, reverse direction flags, and move direction flags to neighbors
      // This is rather messy...

      if(inner_dst_cap.valid()) {
        result.append_to_back({opposite(src_cap_inner), ArcDirection::CCW}); // becomes inner_src_cap, CCW since this is the arc along inner circle
        result.append_to_back({opposite(inner_dst_cap), ArcDirection::CW}); // becomes dst_cap_inner
      }
      else {
        // 'inner' is actually src_cap so this becomes dst_cap_src_cap
        result.append_to_back({opposite(src_cap_inner), ArcDirection::CW});
      }
      // Either branch will end with dst_cap as the reference circle

      if(outer_src_cap.valid()) {
        result.append_to_back({opposite(dst_cap_outer), ArcDirection::CW}); // becomes outer_dst_cap
        result.append_to_back({opposite(outer_src_cap), ArcDirection::CW}); // becomes src_cap_outer
      }
      else {
        // 'outer' is actually dst_cap so this becomes src_cap_dst_cap
        result.append_to_back({opposite(dst_cap_outer), ArcDirection::CW});
      }
    }
    result.contours.end_closed_contour();
  }
}

void add_capsule(ArcAccumulator<Pb::Implicit>& g, const exact::Vec2 x0, const real q, const exact::Vec2 x1, const Quantized signed_offset) {
  const Tuple<exact::Vec2, Quantized> orig_center_and_radius = construct_circle_center_and_radius(x0, x1, q);
  const auto c = ExactCircle<Pb::Implicit>(orig_center_and_radius.x, orig_center_and_radius.y);

  // We need to know if constructed arc intersects endpoints in the correct order
  // First we need to make sure c isn't degenerate (c.radius > 0)
  // If it isn't degenerate our constructed circle must intersect 'helper' circles placed at x0 and x1 with radius of constructed_arc_endpoint_error_bound()
  // If these 'helper' circles don't overlap, their intersections with the constructed circle should to be in the correct ccw/cw order for q
  // This bound is probably more conservative than necessary, but is still small enough that it shouldn't impact accuracy
  const auto ix0 = Vector<Interval,2>(x0);
  const auto ix1 = Vector<Interval,2>(x1);
  const Interval approx_dist = assume_safe_sqrt(sqr_magnitude(ix0 - ix1)); // Use intervals to get conservative bounds of distance
  const bool left_flags_safe = (c.radius > 0) && certainly_less(Interval(2*constructed_arc_endpoint_error_bound()), approx_dist); // Check that helper circles
  const bool prefer_full_circle = (abs(q) >= 1) && (c.radius > 0);

  assert(left_flags_safe || !prefer_full_circle); // This would likely be wildly unstable and so we explode here (in debug builds at least) to avoid headache later
  add_capsule_helper(g, c, (q >= 0) ? x0 : x1, (q >= 0) ? x1 : x0, left_flags_safe, prefer_full_circle, signed_offset);
}

void add_capsule(ArcAccumulator<Pb::Implicit>& g, const ExactArc<Pb::Implicit>& arc, const Quantized signed_offset) {
  const Interval intersection_error_bound = assume_safe_sqrt(Interval(2))*ApproxIntersection::tolerance();
  const Interval approx_dist = assume_safe_sqrt(sqr_magnitude(arc.src.p() - arc.dst.p()));
  // Since the endpoints are approximate, their intersections could be in the wrong order. If we don't catch this, a small section of arc can get turned into a complete circle
  // This checks that neither endpoint could cross the ray through midpoint of original arc (from center of arcs circle)
  const bool left_flags_safe = certainly_less(Interval(2)*intersection_error_bound, approx_dist);

  bool prefer_full_circle = false; // This value won't be used unless !left_flags_safe
  if(!left_flags_safe) { // only call arc_ccw_angle_more_than_pi if result will be used
    prefer_full_circle = arc.is_full_circle() // Repeated vertex is for full circle
                      || !arc.circle.intersections_ccw(arc.src, arc.dst);
  }

  add_capsule_helper(g, arc.circle, arc.src.approx.snapped(), arc.dst.approx.snapped(), left_flags_safe, prefer_full_circle, signed_offset);
}

Tuple<Ref<PlanarArcGraph<Pb::Implicit>>, Nested<HalfedgeId>> offset_closed_exact_arcs(const PlanarArcGraph<Pb::Implicit>& src_g, const Nested<HalfedgeId>& contours, const Quantized signed_offset) {
  IntervalScope scope;

  const auto arc_contours = src_g.combine_concentric_arcs(src_g.edges_to_closed_contours(contours));
  ArcAccumulator<Pb::Implicit> minkowski_terms;

  // Add the original contours
  minkowski_terms.copy_contours(arc_contours, src_g.vertices);

  for(const auto c : arc_contours) {
    for(const auto sa : c) {
      const auto ccw_a = src_g.vertices.arc(src_g.vertices.ccw_arc(sa));
      // Add a capsule around the arc
      add_capsule(minkowski_terms, ccw_a, signed_offset);
    }
  }
  return minkowski_terms.split_and_union();
}

} // geode namespace