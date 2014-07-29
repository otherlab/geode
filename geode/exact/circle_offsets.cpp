#include <geode/exact/scope.h>
#include <geode/exact/circle_offsets.h>
#include <geode/exact/circle_quantization.h>
#include <geode/exact/ExactArcGraph.h>

namespace geode {

static constexpr Pb PS = Pb::Implicit;
typedef ExactArcGraph<PS>::EdgeValue EdgeValue;

#ifdef NDEBUG
#else
static bool is_quantized(const Quantized x) { return ExactInt(x) == x; }
static bool is_quantized(const exact::Vec2 v) { return is_quantized(v.x) && is_quantized(v.y); }
#endif

static void add_capsule_helper(ExactArcGraph<PS>& g, const ExactCircle<PS> c, const exact::Vec2 x0, const exact::Vec2 x1, const bool left_flags_safe, const bool prefer_full_circle, const Quantized signed_offset) {
  const Quantized abs_offset = abs(signed_offset);
  const int capsule_sign = (signed_offset >= 0) ? 1 : -1; // Final area of capsule should have same sign
  assert(is_quantized(c.center));
  assert(is_quantized(c.radius));
  assert(is_quantized(x0));
  assert(is_quantized(x1));
  assert(abs_offset != 0);

  // Tolerance is for each dimension, so we need to multiply by sqrt(2) to get distance error
  const Interval intersection_error_bound = assume_safe_sqrt(Interval(2))*ApproxIntersection::tolerance();
  // If we grow endcap for an arc by this amount, we ensure intersection or coverage of inner and outer offset arcs
  const Quantized endcap_safety_margin = ceil(intersection_error_bound.box().max); // Since offset circles don't need to construct new centers and have exactly correct radius, we only need to account for error of intersections

  const Quantized inner_r = c.radius - abs_offset;
  const Quantized outer_r = c.radius + abs_offset;
  const Quantized endcap_r = abs_offset + endcap_safety_margin;

  if(!left_flags_safe) {
    if(prefer_full_circle) {
      g.add_full_circle(ExactCircle<PS>(c.center, outer_r), EdgeValue(1, capsule_sign));
      if(inner_r > 0) {
        g.add_full_circle(ExactCircle<PS>(c.center, inner_r), EdgeValue(1, -capsule_sign)); // Subtract inside to get an annulus if needed
      }
    }
    else {
      // Use midpoint of verticies to minimize largest error
      const auto new_center = floor(0.5*(x0 + x1));
      g.add_full_circle(ExactCircle<PS>(new_center, endcap_r), EdgeValue(1, capsule_sign)); // Approximate as a single circle
    }
  }
  else {
    // Find or build circles for endcaps
    const auto src_cap = ExactCircle<PS>(x0, endcap_r);
    const auto dst_cap = ExactCircle<PS>(x1, endcap_r);

    IncidentCircle<PS> src_cap_inner;
    IncidentCircle<PS> src_cap_outer;

    IncidentCircle<PS> dst_cap_outer;
    IncidentCircle<PS> dst_cap_inner;

    // Check if outer intersect endcaps to see if we need it
    // Endcap safety margin could (just barely and in very rare cases) make them fully enclose the outer offset arc
    // If this happens we can just use the endcaps
    const auto outer_c = ExactCircle<PS>(c.center, outer_r);
    if(has_intersections(outer_c, src_cap) && has_intersections(outer_c, dst_cap)) {
      const auto outer_src = outer_c.intersection_min(src_cap);
      const auto outer_dst = outer_c.intersection_max(dst_cap);
      const auto outer_arc = ExactArc<PS>({outer_c, outer_src, outer_dst});
      g.add_arc(outer_arc, EdgeValue(1,capsule_sign));
      src_cap_outer = outer_src.reference_as_incident(outer_c);
      dst_cap_outer = outer_dst.reference_as_incident(outer_c);
    }
    else {
      assert(has_intersections(src_cap, dst_cap)); // Outer should only be enclosed in cases where endcaps are intersecting
      const auto i = CircleIntersection<PS>::first(src_cap, dst_cap);
      src_cap_outer = i.incident(ReferenceSide::cl);
      dst_cap_outer = i.incident(ReferenceSide::cr);
    }

    const auto inner_c = ExactCircle<PS>(c.center, inner_r); // Warning: inner_r could be negative
    if((inner_r > 0) && has_intersections(inner_c, src_cap) && has_intersections(inner_c, dst_cap)) {
      const auto inner_src = inner_c.intersection_min(src_cap);
      const auto inner_dst = inner_c.intersection_max(dst_cap);
      const auto inner_arc = ExactArc<PS>({inner_c, inner_src, inner_dst});
      g.add_arc(inner_arc, EdgeValue(1,-capsule_sign)); // This edge travels clockwise so has a negative weight
      src_cap_inner = inner_src.reference_as_incident(inner_c);
      dst_cap_inner = inner_dst.reference_as_incident(inner_c);
    }
    else {
      assert(has_intersections(src_cap, dst_cap)); // Should either have an inner circle or endcaps should intersect
      const auto i = CircleIntersection<PS>::first(dst_cap, src_cap);
      src_cap_inner = i.incident(ReferenceSide::cr);
      dst_cap_inner = i.incident(ReferenceSide::cl);
    }

    g.add_arc(ExactArc<PS>({src_cap, src_cap_inner, src_cap_outer}), EdgeValue(1, capsule_sign));
    g.add_arc(ExactArc<PS>({dst_cap, dst_cap_outer, dst_cap_inner}), EdgeValue(1, capsule_sign));
  }
}

static void add_capsule(ExactArcGraph<PS>& g, const exact::Vec2 x0, const real q, const exact::Vec2 x1, const Quantized signed_offset) {
  const Tuple<exact::Vec2, Quantized> orig_center_and_radius = construct_circle_center_and_radius(x0, x1, q);
  const auto c = ExactCircle<PS>(orig_center_and_radius.x, orig_center_and_radius.y);

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

static void add_capsule(ExactArcGraph<PS>& g, const ExactArc<PS>& arc, const Quantized signed_offset) {
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

template<Pb PS> static Tuple<ExactArcGraph<PS>, Nested<HalfedgeId>> exact_offset_arcs(const ExactArcGraph<PS>& src_g, const Nested<HalfedgeId>& contours, const Quantized signed_offset) {
  IntervalScope scope;
  auto minkowski_terms = ExactArcGraph<PS>();

  for(const auto& c : contours) {
    for(const HalfedgeId he : c) {
      const auto a = src_g.arc(HalfedgeGraph::edge(he));

      // Add the origional segment
      minkowski_terms.add_arc(a, EdgeValue(1, HalfedgeGraph::is_forward(he) ? 1 : -1));

      // Add a capsule around the arc
      add_capsule(minkowski_terms, a, signed_offset);
    }
  }

  minkowski_terms.split_edges();

  const auto contour_edges = extract_region(minkowski_terms.graph, faces_greater_than(minkowski_terms, 0));
  return tuple(minkowski_terms, contour_edges);
}

Nested<CircleArc> offset_arcs(const Nested<const CircleArc> arcs, const real d) {
  const auto bounds = approximate_bounding_box(arcs).thickened(max(d,0));
  IntervalScope scope;
  const auto quant = make_arc_quantizer(bounds);
  const Quantized signed_offset = quant.quantize_length(d);
  if(signed_offset == 0) {
    GEODE_DEBUG_ONLY(GEODE_WARNING("Arc offset amount was below numerical representation threshold! (this should be a few um for geometry that fits in meter bounds)"));
    return arcs.copy();
  }
  // We perform a CSG union of inputs to ensure input to exact_offset_arcs has well behaved inputs
  auto g = ExactArcGraph<PS>();
  g.quantize_and_add_arcs(quant, arcs);
  g.split_edges();
  const auto contours = extract_region(g.graph, faces_greater_than(g, 0));

  const auto new_g_and_contours = exact_offset_arcs(g, contours, signed_offset);
  return new_g_and_contours.x.unquantize_circle_arcs(quant, new_g_and_contours.y);
}

Nested<CircleArc> offset_open_arcs(const Nested<const CircleArc> arcs, const real d) {
  const auto bounds = approximate_bounding_box(arcs).thickened(max(d,0));
  const auto quant = make_arc_quantizer(bounds);
  const Quantized signed_offset = quant.quantize_length(d);
  if(signed_offset == 0) {
    GEODE_DEBUG_ONLY(GEODE_WARNING("Arc offset amount was below numerical representation threshold! (this should be a few um for geometry that fits in meter bounds)"));
    return Nested<CircleArc>(); // Union of 0 width shapes is empty
  }

  IntervalScope scope;
  auto minkowski_terms = ExactArcGraph<PS>();

  for(const auto& c : arcs) {
    assert(c.size() > 0);
    for(const int i : range(c.size() - 1)) {
      add_capsule(minkowski_terms, quant(c[i].x), c[i].q, quant(c[i+1].x), signed_offset);
    }
  }

  minkowski_terms.split_edges();
  const auto contour_edges = extract_region(minkowski_terms.graph, faces_greater_than(minkowski_terms, 0));
  return minkowski_terms.unquantize_circle_arcs(quant, contour_edges);
}

} // namespace geode