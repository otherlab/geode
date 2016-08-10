#include <geode/array/sort.h>
#include <geode/exact/circle_quantization.h>
#include <geode/exact/PlanarArcGraph.h>
#include <geode/exact/scope.h>
#include <geode/mesh/ComponentData.h>
#include <geode/python/Class.h>
#include <geode/utility/curry.h>

namespace geode {

template<Pb PS> static bool valid_contours(const VertexSet<PS>& vertices, const ArcContours& contours) {
  for(const auto c : contours) {
    for(const auto a : c) {
      if(!vertices.circle_ids(a.tail()).contains(vertices.reference_cid(a.head()))) {
        return false;
      }
    }
  }
  return true;
}

namespace {
// We use this in some cases where we might not have an existing IncidentId
template<Pb PS> struct IncidentVertexInfo {
  IncidentCircle<PS> i;
  CircleId ref_cid;
  CircleId inc_cid;
};

template<Pb PS, ArcDirection D=ArcDirection::CCW> struct IncComp {
  static const IncidentCircle<PS>& get_incident(const IncidentCircle<PS>& i) { return i; }
  static const IncidentHorizontal<PS>& get_incident(const IncidentHorizontal<PS>& i) { return i; }
  static const IncidentCircle<PS>& get_incident(const IncidentVertexInfo<PS>& i) { return i.i; }
  const ExactCircle<PS> circle;

  template<class TL, class TR> inline bool operator()(const TL& lhs, const TR& rhs) const {
    if(D == ArcDirection::CCW) {
      return circle.intersections_sorted(get_incident(lhs),get_incident(rhs));
    }
    else {
      return circle.intersections_sorted(get_incident(rhs),get_incident(lhs));
    }
  }
};
} // anonymous namespace

template<Pb PS, ArcDirection D> static void sort_relative(RawArray<IncidentVertexInfo<PS>> incs, const IncComp<PS,D>& comp, const IncidentHorizontal<PS>& first) {
  // Instead of wrapping around just before start of quadrant 0 we want to wrap around just before 'first'

  // Start by partitioning elements relative to 'first'
  // Incidents greater than 'first' (i.e. between 'first' and end of quadrant 3) belong in the low range of our new wrapped order
  // Incidents less than 'first' (i..e. between start of quadrant 0 and 'first') belong in the hi range of our new wrapped order
  const auto mid = std::partition(incs.begin(), incs.end(), curry(comp,first));
  // Now we sort the two sub ranges
  std::sort(incs.begin(),mid,comp);
  std::sort(mid,incs.end(),comp);
}

template<Pb PS> static Field<Box<Vec2>,CircleId> get_circle_clipping(const VertexSet<PS>& vertices, const ArcContours& contours) {
  auto circle_bounds = Field<Box<Vec2>,CircleId>(vertices.n_circles());
  for(const auto c : contours) {
    for(const auto a : c) {
      assert(vertices.circle_ids(a.tail()).contains(vertices.reference_cid(a.head())));
      circle_bounds[vertices.reference_cid(a.head())].enlarge(bounding_box(vertices.arc(vertices.ccw_arc(a))));
    }
  }
  return circle_bounds;
}

template<Pb PS> static Vector<IncidentVertexInfo<PS>,2> get_intersections(const VertexSet<PS>& vertices, const CircleId ref_cid, const CircleId inc_cid) {
  const auto i = vertices.circle(ref_cid).get_intersections(vertices.circle(inc_cid));
  Vector<IncidentVertexInfo<PS>,2> result;
  result.x.i = i.x;
  result.y.i = i.y;
  for(auto& r : result) {
    r.ref_cid = ref_cid;
    r.inc_cid = inc_cid;
  }
  return result;
}

template<Pb PS> IncidentId VertexField<PS>::append_unique(const ExactCircle<PS>& ref, const IncidentCircle<PS>& inc) {
  IncidentId result;
  incidents_.preallocate(incidents_.size() + 2);
  if(inc.side == first_iid_side) {
    incidents_.flat.append_assuming_enough_space(inc);
    incidents_.flat.append_assuming_enough_space(inc.reference_as_incident(ref));
    result = IncidentId(incidents_.size() - 2);
  }
  else {
    incidents_.flat.append_assuming_enough_space(inc.reference_as_incident(ref));
    incidents_.flat.append_assuming_enough_space(inc);
    result = IncidentId(incidents_.size() - 1);
  }
  assert(incidents_[result].side == inc.side);
  assert(incidents_[opposite(result)].side == opposite(inc.side));
  assert(side(result) == inc.side);
  return result;
}

// Find edge (if any) along ref_cid that contains i
template<Pb PS> EdgeId PlanarArcGraph<PS>::find_edge(const CircleId ref_cid, const IncidentHorizontal<PS>& i) const {
  const EdgeId result = outgoing_edge(*find_prev(ref_cid, i));
  assert(!result.valid() || circle_id(result) == ref_cid);
  return result;
}

// Find edge (if any) along v.inc_cid that contains v.i or has an endpoint at v.i and extends into the given face of v.ref_cid
template<Pb PS> static HalfedgeId find_incident_edge(const PlanarArcGraph<PS>& g, const IncidentVertexInfo<PS>& v, const CircleFace ref_face, const ArcDirection ref_dir) {
  const IncidentId iid_if_any = g.vertices.try_find(v.ref_cid, v.inc_cid, v.i.side);
  EdgeId eid;
  if(iid_if_any.valid()) {
    const IncidentId inc_id = opposite(iid_if_any);
    assert(g.vertices.reference_cid(inc_id) == v.inc_cid);
    assert(g.vertices.incident_cid(inc_id) == v.ref_cid);
    // We have a vertex exactly at v which could have different edges in ccw or cw directions
    // Checking which of the 4 combinations of v.i.side and ref_face tells us which we want
    if(cl_is_reference(v.i.side) != (ref_face == CircleFace::interior)) {
      eid = g.outgoing_edge(inc_id);
    }
    else {
      eid = g.outgoing_edge(g.incident_order.ccw_prev(v.inc_cid, inc_id));
    }
  }
  else {
    // We don't have an exact match, so use find_prev to do a binary search to find correct edge
    // We search along inc_cid to find the intersecting edge
    const IncidentCirculator inc_iter = g.find_prev(v.inc_cid, v.i.reference_as_incident(g.vertices.circle(v.ref_cid)));
    eid = g.outgoing_edge(*inc_iter);
    // This is only used after edge splitting so we shouldn't have a valid edge along ref_cid and inc_cid without having a vertex there
    assert(eid.valid() ? !g.outgoing_edge(*(g.find_prev(v.ref_cid, v.i))).valid() : true);
    // It might be possible to have another edge along ref_cid, but it couldn't be incident since it is parallel
    // If that edge had an endpoint at v, try_find should have detected it
  }

  if(!eid.valid())
    return HalfedgeId();
  assert(g.circle_id(eid) == v.inc_cid);
  const bool ref_entering_inc = (ref_dir == ArcDirection::CCW) ^ cr_is_reference(v.i.side); // Check if arc along ref is entering or exiting inc
  return directed_edge(eid, ref_entering_inc ? exterior_edge_dir : interior_edge_dir);
}

template<Pb PS> static ExactHorizontal<PS> select_horizontal(const ExactArc<PS>& arc) {
  const auto arc_bounds = bounding_box(arc);
  const auto safe_y = Box<Quantized>(arc.circle.center.y).thickened(arc.circle.radius - 1); // This range will always have a non-degenerate intersection with arc.circle
  const auto clamped_center = safe_y.clamp(std::round(arc_bounds.vertical_box().center().x));
  // We need to ensure we quantize as 0. instead of -0. so that value can be safely hashed
  return ExactHorizontal<PS>((clamped_center == -0.) ? 0. : clamped_center);
}
template<Pb PS> static Tuple<IncidentHorizontal<PS>,ArcDirection> closest_intersection(const ExactArc<PS>& a, const Vector<IncidentHorizontal<PS>,2>& hits) {
  for(const auto& h : hits) {
    if(a.contains_horizontal(h))
      return tuple(h, ArcDirection::CCW);
  }
  const bool arc_below = a.circle.intersections_upwards(a.src, hits.x);
  const bool dst_closer = a.circle.intersections_upwards(a.src, a.dst) == arc_below;
  const bool left_side_closer = left_of_center(dst_closer ? a.dst : a.src);
  assert(hits[left_side_closer].left == left_side_closer);
  return tuple(hits[left_side_closer], dst_closer ? ArcDirection::CW : ArcDirection::CCW);
}

template<Pb PS> CircleTree<PS>::CircleTree(const VertexSet<PS>& verts, const ArcContours& contours)
 : tree(new_<BoxTree<Vec2>>(get_circle_clipping(verts,contours).flat, 1))
{ }

template<Pb PS> CircleTree<PS>::CircleTree(Uninit)
 : tree(new_<BoxTree<Vec2>>(RawArray<const Box<Vec2>>(0,nullptr),1))
{ }

template<Pb PS> CircleId CircleTree<PS>::prim(const int n) const {
  assert(tree->is_leaf(n) && tree->prims(n).size()==1);
  return CircleId(tree->prims(n)[0]);
}

namespace { template<Pb PS> struct ActiveCirclesHelper {
  const CircleTree<PS>& tree;
  const CircleSet<PS>& circles;
  const Box<Vec2> search_bounds;
  Array<CircleId> result;
  bool cull(const int n) const { return !search_bounds.intersects(tree.tree->boxes[n]); }
  void leaf(const int n) {
    assert(!cull(n));
    result.append(tree.prim(n));
  }
};}

template<Pb PS> Array<CircleId> CircleTree<PS>::circles_active_near(const CircleSet<PS>& circles, const Box<Vec2> bounds) const {
  auto helper = ActiveCirclesHelper<PS>({*this, circles, bounds});
  single_traverse(*tree, helper);
  return helper.result;
}


namespace { template<Pb PS> struct IntersectionHelper {
  const CircleTree<PS>& tree;
  VertexSet<PS>& vertices;
  bool cull(const int n) const { return false; }
  bool cull(const int n0, const int n1) const { return false; }
  void leaf(const int n) const { assert(tree.tree->prims(n).size()==1); }
  void leaf(const int n0, const int n1) {
    if(n0 == n1) // Only check unique arcs
      return;
    const auto b = Box<Vec2>::intersect(tree.tree->boxes[n0],tree.tree->boxes[n1]);
    assert(!b.empty());
    const CircleId cid0 = tree.prim(n0);
    const CircleId cid1 = tree.prim(n1);
    const auto& c0 = vertices.circle(cid0);
    const auto& c1 = vertices.circle(cid1);

    for(const auto& i : c0.intersections_if_any(c1)) {
      if(b.intersects(i.approx.box()))
        vertices.get_or_insert(i, cid0, cid1);
    }
  }

};}

template<Pb PS> static CircleTree<PS> insert_circle_intersections(VertexSet<PS>& vertices, const ArcContours& contours) {
  const auto tree = CircleTree<PS>(vertices, contours);
  IntersectionHelper<PS> helper({tree, vertices});
  double_traverse(*(tree.tree), helper);
  // This doesn't ensure added intersections are on contours so some spurious vertices can be added
  // In practice, bounding boxes seem to be tight enough that it is faster to allow a few spurious vertices rather then adding a filtering step
  return tree;
}

namespace {
template<Pb PS> ExactCircle<PS> construct_circle(const exact::Vec2 x0, const exact::Vec2 x1, const real q) {
  const auto c_and_r = construct_circle_center_and_radius(x0,x1,q);
  return ExactCircle<PS>(c_and_r.x, c_and_r.y);
}

static Quantized helper_circle_radius() {
  return constructed_arc_endpoint_error_bound();
}
template<Pb PS> static ExactCircle<PS> helper_circle(const exact::Vec2 center) {
  return ExactCircle<PS>(center, helper_circle_radius());
}

namespace {
template<Pb PS> struct NewArc {
  exact::Vec2 x0;
  real q;
  CircleId cid;
  ArcDirection direction;
  IncidentId x0_inc;
  IncidentId x1_inc;

  void set_circle_and_direction(VertexSet<PS>& verts, const exact::Vec2 x1) {
    cid = verts.get_or_insert(construct_circle<PS>(x0, x1, q));
    direction = (q >= 0.) ? ArcDirection::CCW : ArcDirection::CW;
  }

  // These find the first/last intersection relative to direction of arc
  // first_intersection is where arc enters incident and last_intersection is where arc leaves incident
  // Assuming start of arc isn't inside incident, using first_intersection as end will have a shorter arc than last_intersection
  IncidentCircle<PS> first_intersection(const ExactCircle<PS>& circle, const ExactCircle<PS>& incident) const {
    return (direction == ArcDirection::CCW) ? circle.intersection_min(incident) : circle.intersection_max(incident);
  }
  IncidentCircle<PS> last_intersection(const ExactCircle<PS>& circle, const ExactCircle<PS>& incident) const {
    return (direction != ArcDirection::CCW) ? circle.intersection_min(incident) : circle.intersection_max(incident);
  }

  void set_x0_inc(VertexSet<PS>& verts, NewArc& prev) {
    const ExactCircle<PS> prev_circle = verts.circle(prev.cid);
    const ExactCircle<PS> curr_circle = verts.circle(cid);
    for(const auto& i : intersections_if_any(prev_circle, curr_circle)) {
      if(i.is_inside(helper_circle<PS>(x0))) {
        // If we find an intersection inside of the helper circle use it
        x0_inc = verts.get_or_insert(i.as_incident_to(curr_circle), cid, prev.cid);
        prev.x1_inc = opposite(x0_inc);
        return;
      }
    }
    // By default we just use the intersection furthest along this arc
    x0_inc = verts.get_or_insert(last_intersection(curr_circle, helper_circle<PS>(x0)),
                                cid,
                                verts.get_or_insert(helper_circle<PS>(x0)));
  }
  void set_x1_inc(VertexSet<PS>& verts, const NewArc& next) {
    if(x1_inc.valid())
      return;

    if(cid == next.cid) { // If the next arcs continues along the current circle
      // Merging these two arcs is hard since we need to track cases where we wrap all the way around or reverse directions
      // Instead of merging we just reuse intersection with a helper circle on next arc
      assert(is_same_circle(verts.incident(next.x0_inc), helper_circle<PS>(next.x0))); // We should already have intersection from when we computed next.x0_inc
      // WARNING: It's important that we reuse same intersection point (otherwise we would have a gap between this and next)
      x1_inc = next.x0_inc;
      return;
    }

    // If the above cases fall through we compute a new intersection at the helper circle
    const CircleId helper_cid = verts.incident_cid(next.x0_inc);
    assert(is_same_circle(verts.circle(helper_cid),helper_circle<PS>(next.x0)));
    x1_inc = verts.get_or_insert(first_intersection(verts.circle(cid),verts.circle(helper_cid)), cid, helper_cid);
  }

};
} // end anonymous namespace
template<Pb PS> bool is_same_intersection(const ExactCircle<PS>& ref0, const IncidentCircle<PS>& inc0, const ExactCircle<PS>& ref1, const IncidentCircle<PS>& inc1) {
  if(inc0.side == inc1.side)
    return is_same_circle(ref0, ref1) && is_same_circle(inc0, inc1);
  else
    return is_same_circle(ref0, inc1) && is_same_circle(inc0, ref1);
}

} // anonymous namespace

static Array<int> make_next_i(const int n) {
  auto result = Array<int>(n, uninit);
  for(int i_prev = n-1, i_curr = 0; i_curr < n; i_prev = i_curr++) {
    result[i_prev] = i_curr;
  }
  return result;
}


// Construct a circle to generate an intersection with input circle at a point close to x0
// Result should on or inside a helper_circle centered at x0
template<Pb PS> static IncidentCircle<PS> construct_intersection_for_endpoint(const ExactCircle<PS>& circle, const exact::Vec2 x0) {
  assert(has_intersections(circle, helper_circle<PS>(x0))); // This should only be called for points that are 'close' to circle

  // Using this for multiple iterations of quantization/unquantization might behave poorly since any given endpoint will tend to be approximated in the same direction each iteration
  // It if turns out to be a problems we could hash x0 to select between intersection_min and intersection_max via threefry(first_arc.x0.x, first_arc.x0.y) which might help
  // I suspect we could construct another circle that will have an intersection with the input circle somewhere inside the helper circle in most (all?) cases
  // I don't think any of this is actually necessary so I'm using the simple version for now instead of debugging this
#if 0

  // We rotate offset from target point to center of input circle to get center of a new circle that would intersect the input circle at x0 if x0 was exactly on input circle
  const auto ortho_circle = ExactCircle<PS>{x0 + rotate_left_90(x0-circle.center), circle.radius};

  // Distance from x0 should be same for both circles so ortho_circle should also intersect a helper circle at x0

  // There should be an intersection between ortho_circle and the input circle
  assert(has_intersections(circle, ortho_circle));

  const auto result = circle.intersection_min(ortho_circle);

  // Intersection of ortho_circle and input circle should be inside helper circle
  assert(ExactArc<PS>{circle,
                      circle.intersection_min(helper_circle<PS>(x0)),
                      circle.intersection_max(helper_circle<PS>(x0))}.interior_contains(result));
  return result;
#else
  // I don't think this is likely to be a problem in practice so I'm just using the simple implementation for now
  return circle.intersection_min(helper_circle<PS>(x0));
#endif
}


template<Pb PS> void VertexSet<PS>::quantize_circle_arcs(const Quantizer<real,2>& quant, const RawArray<const CircleArc> src_arcs, ArcContours& result, const bool src_arcs_open) {
  const int src_n = src_arcs.size();
  const auto src_next = make_next_i(src_n);

  auto new_arcs = Array<NewArc<PS>>();
  if(src_n == 0) {
    return; // Ignore empty input
  }

  exact::Vec2 tail_point = quant(src_arcs.front().x);
  for(const int src_i : range(src_n - src_arcs_open)) {
    const exact::Vec2 x0 = tail_point;
    const exact::Vec2 x1 = quant(src_arcs[src_next[src_i]].x);
    if(x0 == x1 || circles_overlap(helper_circle<PS>(x0), helper_circle<PS>(x1)))
      continue; // Skip over small degenerate arcs
    // If we skipped arcs, x0 won't be same as src_arcs[src_i].x, but it should have changed by at most the helper circle diameter + 1 so it should be safe to use
    new_arcs.append({x0, src_arcs[src_i].q});
    tail_point = x1;
  }

  // If after filtering we have a single point, ignore it and return without adding anything to result
  if(new_arcs.size() <= 1 - src_arcs_open)
    return;

  const int new_n = new_arcs.size();
  const int before_start = src_arcs_open ? 0 : new_n-1;
  const int after_start = src_arcs_open ? 1 : 0;

  for(int curr_i = before_start, next_i = after_start; next_i < new_n; curr_i = next_i++) {
    new_arcs[curr_i].set_circle_and_direction(*this, new_arcs[next_i].x0); // Construct all of the circles
  }

  // For open arcs need to treat front and back differently
  if(src_arcs_open) {
    auto& first_arc = new_arcs.front();
    auto& last_arc = new_arcs.back();

    last_arc.set_circle_and_direction(*this, tail_point);

    const auto first_intersection = construct_intersection_for_endpoint(this->circle(first_arc.cid), first_arc.x0);

    // Helper circles constructed at endpoints of each arc don't overlap
    // This lets us assume constructed endpoints will safely avoid hitting other side of arcs
    first_arc.x0_inc = get_or_insert(first_intersection, first_arc.cid,
                                     get_or_insert(first_intersection.as_circle()));

    const auto last_intersection = construct_intersection_for_endpoint(this->circle(last_arc.cid), tail_point);
    last_arc.x1_inc = get_or_insert(last_intersection, last_arc.cid,
                                    get_or_insert(last_intersection.as_circle()));
  }

  for(int prev_i = before_start, curr_i = after_start; curr_i < new_n; prev_i = curr_i++) {
    // Construct starting vertices
    new_arcs[curr_i].set_x0_inc(*this, new_arcs[prev_i]);
  }

  for(int curr_i = before_start, next_i = after_start; next_i < new_n; curr_i = next_i++) {
    // Either reuse starting vertices or construct new end ones
    new_arcs[curr_i].set_x1_inc(*this, new_arcs[next_i]);
  }

  for(int prev_i = before_start, next_i = after_start; next_i < new_n; prev_i = next_i++) {
    // Although we have a series of arcs and their endpoints, some neighbors might need helper arcs inserted between them
    auto& prev = new_arcs[prev_i];
    auto& next = new_arcs[next_i];

    // Check if end of previous arc isn't start of next arc in which case we need to add a helper arc
    if(!is_same_vid(prev.x1_inc, next.x0_inc)) {
      // If endpoints aren't the same, we will need to construct a helper arc

      const auto helper_cid = incident_cid(prev.x1_inc); // We should already have a helper circle
      assert(incident_cid(next.x0_inc) == helper_cid); // Check that next arc connects here

      const auto hc = this->circle(helper_cid); // Helper arc will use this circle
      assert(is_same_circle(hc, helper_circle<PS>(next.x0))); // Make sure we have a helper circle with the right center

      // Inputs to quantization will usually need to undergo a union (to remove self intersections) before further operations are safe
      // In this case quantization will usually be on positive area polyarcs. Under the assumption that eroding small features is better
      // than adding them we want to add helper arcs with a negative winding

      // Canonical arcs are ccw so we connect prev to dst and next to src if we are going to use a negative winding
      // We compute all options and choose the best one
      const auto src_options = hc.get_intersections(this->circle(next.cid));
      const auto dst_option_0 = this->incident(opposite(prev.x1_inc));
      const auto dst_option_1 = hc.other_intersection(dst_option_0);

      assert(!hc.is_same_intersection(dst_option_0, dst_option_1));

      // Guess first option on each side
      auto helper_arc = ExactArc<PS>({hc, src_options[0], dst_option_0});

      // If next and prev had an intersection inside we wouldn't need the helper arc
      // This implies that the ccw order or intersections will be a rotation of src_option[a], src_option[!a], dst_option[b], dst_option[!b] for some a and b
      if(helper_arc.unsafe_contains(src_options[1]))
        helper_arc.src = src_options[1]; // If src_option[0] was first, switch to it
      if(helper_arc.unsafe_contains(dst_option_1)) {
        helper_arc.dst = dst_option_1; // If dst_option[1] was first, switch to it
        prev.x1_inc = get_or_insert(dst_option_1.reference_as_incident(hc), prev.cid, helper_cid);
      }

      // We need to make sure next starts at end of helper_arc
      next.x0_inc = get_or_insert(helper_arc.src.reference_as_incident(helper_arc.circle), next.cid, helper_cid);

      assert(is_same_intersection(this->circle(prev.cid), this->incident(prev.x1_inc), helper_arc.circle, helper_arc.dst));
      assert(is_same_intersection(this->circle(next.cid), this->incident(next.x0_inc), helper_arc.circle, helper_arc.src));

      // We add the helper arcs in the next pass along with the regular arcs
    }
  }

  result.start_contour();
  if(src_arcs_open) {
    result.append_to_back({new_arcs[0].x0_inc, new_arcs[0].direction});
  }
  for(int prev_i = before_start, curr_i = after_start; curr_i < new_n; prev_i = curr_i++) {
    const auto& prev = new_arcs[prev_i];
    const auto& curr = new_arcs[curr_i];

    // Check if we need to bridge to previous arcs with a helper arc
    if(!is_same_vid(prev.x1_inc, curr.x0_inc)) {
      // As explained above, we use negative winding to try to minimize features added by quantization
      result.append_to_back({opposite(prev.x1_inc), ArcDirection::CW}); // Add the helper arc
    }
    result.append_to_back({curr.x0_inc, curr.direction});
  }
  if(src_arcs_open) {
    result.end_open_contour(to_vid(new_arcs.back().x1_inc));
  }
  else {
    result.end_closed_contour();
  }
}

template<Pb PS> void VertexSet<PS>::quantize_circle_arcs(const Quantizer<real,2>& quant, const Nested<const CircleArc> src_arcs, ArcContours& result, const bool src_arcs_open) {
  for(const auto contour : src_arcs) {
    quantize_circle_arcs(quant, contour, result, src_arcs_open);
  }
  assert(valid_contours(*this, result));
}

template<Pb PS> ArcContours VertexSet<PS>::quantize_circle_arcs(const Quantizer<real,2>& quant, const Nested<const CircleArc> src_arcs, const bool src_arcs_open) {
  ArcContours result;
  quantize_circle_arcs(quant, src_arcs, result, src_arcs_open);
  return result;
}

template<Pb PS> PlanarArcGraph<PS>::PlanarArcGraph(Uninit)
 : circle_tree(uninit)
 , incident_order(uninit)
 , topology(new_<HalfedgeGraph>())
{ }

namespace {
// This class provides iterators over an infinite sequence of ones
// Used as default for weights in init_topology_and_windings
struct AlwaysOneSequence {
  struct End {};
  struct Iter
  {
    constexpr One operator*() const { return One{}; }
    constexpr Iter operator++() const { return *this; }
    GEODE_UNUSED inline friend constexpr bool operator!=(const Iter, const End) { return true; }
  };
  constexpr Iter begin() const { return Iter{}; }
  constexpr End end() const { return End{}; }
};
} // anonymous namespace

template<Pb PS> void PlanarArcGraph<PS>::embed_arcs(const ArcContours& contours, const RawArray<const int8_t> weights) {
  circle_tree = insert_circle_intersections(vertices, contours);
  incident_order = VertexSort<PS>(vertices);
  if(weights.empty()) {
    edge_srcs = init_topology_and_windings(topology, edge_windings, outgoing_edges, vertices, contours, AlwaysOneSequence{}, incident_order);
  }
  else {
    assert(weights.size() == contours.size());
    edge_srcs = init_topology_and_windings(topology, edge_windings, outgoing_edges, vertices, contours, weights, incident_order);
  }
  init_borders_and_faces();
}

template<Pb PS> PlanarArcGraph<PS>::PlanarArcGraph(const VertexSet<PS>& _vertices, const ArcContours& contours, const RawArray<const int8_t> weights)
 : circle_tree(uninit)
 , vertices(_vertices)
 , incident_order(uninit)
 , edge_windings()
 , outgoing_edges()
 , edge_srcs()
 , topology(new_<HalfedgeGraph>())
{
  embed_arcs(contours, weights);
}

namespace { template<Pb PS> struct LeftwardRaycastHelper {
  const PlanarArcGraph<PS>& graph;
  const HorizontalIntersection<PS> dst; // Line for ray and its leftmost point
  const CircleId ref_cid;
  typedef Tuple<HorizontalIntersection<PS>, HalfedgeId> Hit;
  Array<Hit> unsorted_hits;

  bool ray_x_intersects(const Box<real>& b) const {
    return !(b.max < dst.x.box().min);
  }
  bool cull(const int n) const {
    const auto b = graph.circle_tree.tree->boxes[n];
    return !ray_x_intersects(b[0]) || !b[1].inside(dst.line.y, Zero());
  }
  void leaf(const int n) {
    assert(!cull(n));
    const auto hit_bounds = graph.circle_tree.tree->boxes[n];
    const auto hit_cid = graph.circle_tree.prim(n);
    const auto hit_c = graph.vertices.circle(hit_cid);

    for(const HorizontalIntersection<PS>& i : intersections_if_any(hit_c, dst.line)) {
      if(!hit_bounds.intersects(i.box())) {
        assert(!graph.find_edge(hit_cid, i).valid()); // Make sure we don't have valid edges outside of our boxes
        continue; // Skip any intersection that is outside the clipped bounding box of the circle (It won't be on a valid edge)
      }
      // Check that intersection is on the ray
      if(!(i == dst) && intersections_rightwards(i, dst))
        continue;
      const EdgeId eid = graph.find_edge(hit_cid, i);
      if(!eid.valid())
        continue;
      unsorted_hits.append(tuple(i, i.left ? ccw_edge(eid) : cw_edge(eid)));
    }
  }
  Array<HalfedgeId> hits() {
    std::sort(unsorted_hits.begin(), unsorted_hits.end(), [] (const Hit& lhs, const Hit& rhs) { return rhs.x < lhs.x; }); // CAUTION: We swap argument order to get reversed sort!
    return unsorted_hits.template project<HalfedgeId, &Hit::y>().copy();
  }
};}

template<Pb PS> Array<HalfedgeId> PlanarArcGraph<PS>::leftward_raycast_to(const CircleId ref_cid, const IncidentHorizontal<PS>& i) const {
  auto helper = LeftwardRaycastHelper<PS>({*this, {i, vertices.circle(ref_cid) }, ref_cid});
  single_traverse(*circle_tree.tree, helper);
  return helper.hits();
}

template<Pb PS> Array<HalfedgeId> PlanarArcGraph<PS>::path_from_infinity(const EdgeId target_eid) const {
  const auto target_arc = arc(target_eid);
  const auto h = select_horizontal(target_arc);
  const auto h_inc_and_dir = closest_intersection(target_arc, target_arc.circle.get_intersections(h));
  const IncidentHorizontal<PS> h_inc = h_inc_and_dir.x;
  const CircleId target_cid = circle_id(target_eid);
  Array<HalfedgeId> result = leftward_raycast_to(target_cid, h_inc);

  assert(!find_edge(target_cid, h_inc).valid() // Either no edge on circle at h
          || (!result.empty() && circle_id(result.back()) == target_cid)); // Or raycast path ends at edge on circle
  if(result.empty() || HalfedgeGraph::edge(result.back()) != target_eid) {
    // If we didn't find the edge follow circle to ensure we get the right edge
    // In effect, we perform a 'raycast' from h_inc along the target edge's circle until we hit the target edge

    // This is the circle we will follow along
    const ExactCircle<PS> ref_circle = target_arc.circle;

    // This is the shorter direction (computed above in closest_intersection) that from h_inc to target_arc
    const ArcDirection path_dir = h_inc_and_dir.y;

    // This is the closer endpoint of target_arc that we will hit first
    const IncidentId target_iid = (path_dir == ArcDirection::CCW) ? src(target_eid) : dst(target_eid);
    const IncidentCircle<PS> target_inc = vertices.incident(target_iid);
    assert(left_of_center(target_inc) == left_of_center(h_inc));

    // This is the arc between h_inc and target_arc
    const auto h_to_target = (path_dir == ArcDirection::CCW) ? ExactHorizontalArc<PS>(ref_circle, h_inc, target_inc)
                                                             : ExactHorizontalArc<PS>(ref_circle, target_inc, h_inc);
    const auto arc_path_bounds = bounding_box(h_to_target);

    Array<IncidentVertexInfo<PS>> path_incidents; // We will collect incident intersections on our path here
    for(const CircleId inc_cid : circle_tree.circles_active_near(vertices, arc_path_bounds)) {
      for(const IncidentCircle<PS>& i : target_arc.circle.intersections_if_any(vertices.circle(inc_cid))) {
        if(arc_path_bounds.intersects(i.approx.box())
         && !ref_circle.is_same_intersection(i, target_inc)
         && h_to_target.contains(i)) {
          path_incidents.append(IncidentVertexInfo<PS>({i, target_cid, inc_cid}));
        }
      }
    }

    if(path_dir == ArcDirection::CCW)
      sort_relative<PS,ArcDirection::CCW>(path_incidents, IncComp<PS,ArcDirection::CCW>({target_arc.circle}), h_inc);
    else
      sort_relative<PS,ArcDirection::CW>(path_incidents, IncComp<PS,ArcDirection::CW>({target_arc.circle}), h_inc);

    // path_incidents should now be all incidents along h_to_target in order, starting at h_inc and ending just before target_inc
    // Add one more for target_inc which we excluded above
    path_incidents.append(IncidentVertexInfo<PS>({target_inc, vertices.reference_cid(target_iid), vertices.incident_cid(target_iid)}));

    // We need to be careful that we correctly handle edges with endpoints on ref_circle or that follow along it
    // To do this, we assume that we stay either slightly inside or outside based on side just after raycast hit at h_inc (even if there wasn't an edge there)
    const CircleFace f = h_inc.left ? CircleFace::exterior : CircleFace::interior;

    for(const auto& inc : path_incidents) {
      const HalfedgeId hid = find_incident_edge(*this, inc, f, path_dir);
      if(hid.valid()) {
        result.append(hid);
      }
    }
    result.append(directed_edge(target_eid, f == CircleFace::exterior ? exterior_edge_dir : interior_edge_dir));
  }

  return result;
}

// Check that Euler characteristic is consistent with a planar graph
GEODE_UNUSED static bool euler_characteristic_planar(const HalfedgeGraph& g, const ComponentData& cd) {
  const int V = g.n_vertices();
  const int E = g.n_edges();
  const int B = g.n_borders();
  const int F = g.n_faces();
  const int C = cd.n_components();
  if(F == 0) { // HalfedgeGraph treats empty graphs as having zero faces, but to be consistent with normal definition of euler characteristic empty graph should have 1 face
    if(E != 0 || B != 0 || C != 0) {
      assert(false);
      return false;
    }
    return true;
  }

  // We don't define components for isolated vertices, which doesn't match normal definition. Account for that here
  int isolated_v = 0;
  for(const VertexId vid : g.vertices()) {
    if(g.isolated(vid)) ++isolated_v;
  }

  if((1 + B - F) != C) { // We can relate number of borders to number of components
    return false;
  }

  if((V - E + F - (C+isolated_v)) != 1) {
    return false;
  }

  return true;
}

template<Pb PS> void PlanarArcGraph<PS>::init_borders_and_faces() {
  auto& t = *topology;
  t.initialize_borders();
  auto cd = ComponentData(t);
  assert(cd.n_components() != 0 || t.n_edges() == 0);
  FaceId infinity_face;

  for(const ComponentId seed_c : cd.components()) {
    if(cd[seed_c].exterior_face.valid())
      continue;
    const EdgeId seed_eid = HalfedgeGraph::edge(t.halfedge(cd.border(seed_c)));
    const auto p = path_from_infinity(seed_eid);
    initialize_path_faces(p, infinity_face, t, cd);
    assert(cd[seed_c].exterior_face.valid()); // Path ought to have set the exterior face
  }

  // Now that we have safely set exterior faces for each component, we need to generate any interior faces that weren't traversed
  t.initialize_remaining_faces();

  assert(!infinity_face.valid() || infinity_face == boundary_face());
  // At this point all borders should have a valid face id

  assert(euler_characteristic_planar(topology, cd));
}

template<Pb PS> EdgeId PlanarArcGraph<PS>::outgoing_edge(const IncidentId src) const {
  for(const HalfedgeId hid : topology->outgoing(to_vid(src))) {
    if(HalfedgeGraph::is_forward(hid) && edge_srcs[HalfedgeGraph::edge(hid)] == src)
      return HalfedgeGraph::edge(hid);
  }
  return EdgeId();
}

namespace { template<Pb PS> struct CompareIncId {
  const VertexSet<PS>& verts; const ExactCircle<PS> c;
  bool operator()(const IncidentId iid0, const IncidentId iid1) const {
    assert(is_same_circle(c, verts.reference(iid0)) && is_same_circle(c, verts.reference(iid1)));
    assert((iid0 == iid1) == c.is_same_intersection(verts.incident(iid0), verts.incident(iid1)));
    return (iid0 != iid1) && c.unique_intersections_sorted(verts.incident(iid0), verts.incident(iid1));
  }
  bool operator()(const IncidentId i0, const IncidentHorizontal<PS>& i1) const {
    assert(is_same_circle(c, verts.reference(i0)));
    return c.unique_intersections_sorted(verts.incident(i0), i1);
  }
  bool operator()(const IncidentHorizontal<PS>& i0, const IncidentId i1) const {
    assert(is_same_circle(c, verts.reference(i1)));
    return c.unique_intersections_sorted(i0, verts.incident(i1));
  }
  bool operator()(const IncidentId iid0, const IncidentCircle<PS>& i1) const {
    assert(is_same_circle(c, verts.reference(iid0)));
    const auto& i0 = verts.incident(iid0);
    return c.intersections_sorted(i0, i1);
  }
  bool operator()(const IncidentCircle<PS>& i0, const IncidentId iid1) const {
    assert(is_same_circle(c, verts.reference(iid1)));
    const auto& i1 = verts.incident(iid1);
    return c.intersections_sorted(i0, i1);
  }
};} // end anonymous namespace


// std::lower_bound --> first element, s.t. !(element < value) or !comp(element, value) (i.g. Will return element == value if present)
// std::upper_bound --> first element, s.t. (value < element) or comp(value, element) (i.g. Will not return element == value if present)

template<Pb PS> IncidentCirculator PlanarArcGraph<PS>::find_prev(const CircleId ref_cid, const IncidentHorizontal<PS>& i) const {
  // Get IncidentId for all incidents on the circle
  const RawArray<const IncidentId> circle_incidents = incident_order.circle_incidents(ref_cid);
  // Do a binary search on circle_incidents to bracket 'i'
  assert(!circle_incidents.empty());
  // We use upper_bound to get next incident if we are searching for a duplicate (Although upper_bound/lower_bound only matters for version of this function where 'i' happens to be an IncidentCircle)
  IncidentId const* first_after = std::upper_bound(circle_incidents.begin(), circle_incidents.end(), i, CompareIncId<PS>({vertices, vertices.circle(ref_cid)}));
  int prev_i = int(first_after - circle_incidents.begin()) - 1; // Subtract 1 to get previous incident (this also handles first_after == circle_incidents.end())
  if(prev_i == -1) prev_i = circle_incidents.size() - 1; // Handle fact that circle_incidents wrap around
  auto result = IncidentCirculator({circle_incidents, prev_i});
  assert(vertices.arc(*result,result.next()).contains_horizontal(i));
  return result;
}

template<Pb PS> IncidentCirculator PlanarArcGraph<PS>::find_prev(const CircleId ref_cid, const IncidentCircle<PS>& i) const {
  assert(!is_same_circle(vertices.circle(ref_cid), i.as_circle())); // Crude check that user didn't swap reference/incident
  // Get IncidentId for all incidents on the circle
  const RawArray<const IncidentId> circle_incidents = incident_order.circle_incidents(ref_cid);
  // Do a binary search on circle_incidents to bracket 'i'
  assert(!circle_incidents.empty());
  // We use upper_bound to get next incident if we are searching for a duplicate (Although upper_bound/lower_bound only matters for version of this function where 'i' happens to be an IncidentCircle)
  IncidentId const* first_after = std::upper_bound(circle_incidents.begin(), circle_incidents.end(), i, CompareIncId<PS>({vertices, vertices.circle(ref_cid)}));
  int prev_i = int(first_after - circle_incidents.begin()) - 1; // Subtract 1 to get previous incident (this also handles first_after == circle_incidents.end())
  if(prev_i == -1) prev_i = circle_incidents.size() - 1; // Handle fact that circle_incidents wrap around
  auto result = IncidentCirculator({circle_incidents, prev_i});
  assert(vertices.arc(*result, result.next()).half_open_contains(i));
  assert(vertices.reference_cid(*result) == ref_cid);
  assert(vertices.reference_cid(result.next()) == ref_cid);
  return result;
}

template<Pb PS> static IncidentId ccw_prev(const VertexSet<PS>& verts, const VertexSort<PS>& incident_order, const IncidentId iid) {
  return incident_order.ccw_prev(verts.reference_cid(iid), iid);
}

// Get the halfedges with src at vid
template<Pb PS> static SmallArray<HalfedgeId,4> outgoing_edges(const VertexId vid, const VertexSet<PS>& verts, const VertexSort<PS>& incident_order, const Field<EdgeId, IncidentId>& ccw_next_edges) {
  const IncidentId curr_il = iid_cl(vid);
  const IncidentId curr_ir = iid_cr(vid);
  assert(ccw_next_edges.valid(curr_ir));
  const EdgeId r_ccw_next = ccw_next_edges[curr_ir];
  assert(ccw_next_edges.valid(curr_il));
  const EdgeId l_ccw_next = ccw_next_edges[curr_il];
  assert(ccw_next_edges.valid(ccw_prev(verts, incident_order, curr_ir)));
  const EdgeId r_ccw_prev = ccw_next_edges[ccw_prev(verts, incident_order, curr_ir)];
  assert(ccw_next_edges.valid(ccw_prev(verts, incident_order, curr_il)));
  const EdgeId l_ccw_prev = ccw_next_edges[ccw_prev(verts, incident_order, curr_il)];
  SmallArray<HalfedgeId,4> result;

  if(r_ccw_next.valid()) {result.append(ccw_edge(r_ccw_next));}
  if(l_ccw_next.valid()) {result.append(ccw_edge(l_ccw_next)); }
  if(r_ccw_prev.valid()) {result.append(cw_edge(r_ccw_prev)); }
  if(l_ccw_prev.valid()) {result.append(cw_edge(l_ccw_prev)); }

  return result;
}

template<Pb PS, class Weights> static Field<IncidentId, EdgeId> init_topology_and_windings(HalfedgeGraph& topology, Field<int, EdgeId>& edge_windings, Field<EdgeId, IncidentId>& ccw_next_edges, const VertexSet<PS>& verts, const ArcContours& contours, const Weights weights, const VertexSort<PS>& incident_order) {
  assert(topology.n_vertices() == 0 && topology.n_edges() == 0);
  assert(edge_windings.empty());
  assert(ccw_next_edges.empty());
  auto incident_values = Field<int, IncidentId>(verts.n_incidents()); // Winding of edge between iid and ccw_next[iid]
  auto incident_active = Field<bool, IncidentId>(verts.n_incidents()); // True if there should be an edge (possibly with zero weight) between iid and ccw_next[iid]

  {
    auto weight_iter = weights.begin();
    for(const auto c : contours) {
      assert(weight_iter != weights.end());
      const auto weight = *weight_iter;
      ++weight_iter;
      for(const SignedArcInfo sa : c) {
        const UnsignedArcInfo ua = verts.ccw_arc(sa);
        const CircleId cid = verts.reference_cid(ua.src);
        auto ic = incident_order.circulator(cid, ua.src);
        do {
          incident_values[*ic] += sign(sa.direction()) * weight;
          incident_active[*ic] = true;
          ++ic;
        } while(*ic != ua.dst);
      }
    }
  }

  ccw_next_edges.flat.exact_resize(verts.n_incidents());

  Field<IncidentId, EdgeId> edge_srcs;
  for(const IncidentId src_iid : verts.incident_ids()) {
    if(!incident_active[src_iid]) {
      assert(incident_values[src_iid] == 0); // If incident value was touched, should have also marked active
      continue;
    }
    // Note: This used to skip creation of zero weight edges since they aren't important when only looking at winding numbers (which is most of the time)
    //   However, preserving zero weight edges is important if caller wants to later query the fate of input contours
    //   My intuition is that even with lots of degenerate inputs, the overhead of maintaining the zero weight edges will be a small fraction of total time/memory
    //   For now, we always preserve zero weight edges since letting caller control this seems messy and error prone
    edge_windings.append(incident_values[src_iid]);
    const EdgeId eid = edge_srcs.append(src_iid);
    assert(edge_windings.size() == edge_srcs.size());
    ccw_next_edges[src_iid] = eid;
  }

  // We manually perform surgery on the halfedge graph to avoid incrementally rewriting links
  auto& topology_halfedges = topology.halfedges_;
  auto& topology_verts = topology.vertices_;

  topology_verts.flat.exact_resize(verts.n_vertices(), uninit);
  topology_halfedges.flat.exact_resize(2*edge_srcs.size(),uninit);

  // Some crude checks that mucking about with internal data structures is at least roughly correct
  static_assert(sizeof(HalfedgeGraph::HalfedgeInfo) == 4*sizeof(HalfedgeId), "Sizes don't match. Probably need to update graph construction to handle changed fields");
  static_assert(sizeof(HalfedgeGraph::VertexInfo) == 1*sizeof(VertexId), "Sizes don't match. Probably need to update graph construction to handle changed fields");
  assert(topology.n_vertices() == verts.n_vertices());
  assert(topology.n_edges() == edge_srcs.size());

  for(auto& e : topology_halfedges.flat) {
    e.border = BorderId();
  }

  for(const VertexId vid : verts.vertex_ids()) {
    const auto outgoing = outgoing_edges(vid, verts, incident_order, ccw_next_edges);
    if(!outgoing.empty()) {
      topology_verts[vid].halfedge = outgoing.front();
      HalfedgeId prev_o = outgoing.back();
      for(const HalfedgeId next_o : outgoing) {
        const HalfedgeId next_i = HalfedgeGraph::reverse(next_o);
        topology_halfedges[next_i].next = prev_o;
        topology_halfedges[prev_o].prev = next_i;
        topology_halfedges[next_o].src = vid;
        prev_o = next_o;
      }
    }
    else {
      topology_verts[vid].halfedge = HalfedgeId();
    }
  }

  return edge_srcs;
}

// Note: Faces are constructed so that this is always FaceId(0), but future optimizations might change this
template<Pb PS> FaceId PlanarArcGraph<PS>::boundary_face() const {
  return FaceId(0);
}

template<Pb PS> IncidentId VertexSet<PS>::try_find(const CircleId ref_cid, const CircleId inc_cid, const ReferenceSide side) const {
  const auto ordered_cids = cl_is_reference(side) ? vec(ref_cid, inc_cid) : vec(inc_cid, ref_cid);
  const VertexId vid = vid_cache.get_default(ordered_cids);
  const IncidentId result = vid.valid() ? incident_id(vid, side) : IncidentId();
  assert(!result.valid() || (reference_cid(result) == ref_cid && incident_cid(result) == inc_cid));
  return result;
}

template<Pb PS> IncidentId VertexSet<PS>::get_or_insert(const IncidentCircle<PS>& inc, const CircleId ref_cid, const CircleId inc_cid) {
  assert(iid_to_cid.size() == this->incidents_.size());
  assert(ref_cid.valid() && inc_cid.valid());
  assert(is_same_circle(inc.as_circle(), this->circle(inc_cid)));
  const auto ordered_cids = cl_is_reference(inc.side) ? vec(ref_cid, inc_cid) : vec(inc_cid, ref_cid);
  VertexId& vid = vid_cache.get_or_insert(ordered_cids);
  IncidentId result;
  if(!vid.valid()) {
    result = this->append_unique(this->circle(ref_cid), inc);
    vid = to_vid(result);
    iid_to_cid.flat.extend(ordered_cids.swap());
    assert(iid_to_cid.size() == this->incidents_.size());
    assert(reference_cid(result) == ref_cid);
    assert(incident_cid(result) == inc_cid);
  }
  else {
    result = incident_id(vid, inc.side);
  }
  return result;
}

template<Pb PS> CircleId VertexSet<PS>::reference_cid(const IncidentId iid) const {
  return iid_to_cid[opposite(iid)];
}

template<Pb PS> CircleId VertexSet<PS>::incident_cid(const IncidentId iid) const {
  const auto result = iid_to_cid[iid];
  assert(is_same_circle(this->circle(result), this->incident(iid).as_circle()));
  return result;
}

// Check if arc from prev_head to curr_head can be concatonated with arc from curr_head to next_head into a single arc from prev_head to next_head
template<Pb PS> static bool is_middle_head_redundant(const SignedArcHead prev_head, const SignedArcHead curr_head, const SignedArcHead next_head, const VertexSet<PS>& vertices, const VertexSort<PS>& incident_order) {

  // We need to check for a change in direction or circle and that we don't 'overflow' past 360 degrees

  if(prev_head.direction != curr_head.direction)
    return false; // Arcs must continue in the same direction at curr

  if(vertices.reference_cid(prev_head.iid) != vertices.reference_cid(curr_head.iid))
    return false; // Arcs must continue along the same circle at curr


  if(prev_head.iid == curr_head.iid) // Check if prev_head to curr_head is a full circle
    return false;  // Can't represent more than a full circle with a single arc, so we need to keep intermediate head

  // If and only if next_head is inside the previous arc, the new angle will be greater than 360 degrees
  // We need to get the CCW ordered endpoints...
  const auto ccw_arc = (prev_head.direction == ArcDirection::CCW) ? UnsignedArcInfo({prev_head.iid, curr_head.iid}) : UnsignedArcInfo({curr_head.iid, prev_head.iid});
  // ...so that we can test if next_head is inside that arc
  if(incident_order.arc_interior_contains(ccw_arc, vertices.find_iid(to_vid(next_head.iid), vertices.reference_cid(curr_head.iid)))) {
    return false;
  }

  // As long as we don't have any of the above issues, curr_head is redundant and can be erased
  return true;
}

template<Pb PS> ArcContours VertexSet<PS>::combine_concentric_arcs(const ArcContours& contours, const VertexSort<PS>& incident_order) const {
  ArcContours result;
  for(const auto contour : contours) {
    RawArray<const SignedArcHead> heads = contour.heads; // We break the abstraction barrier of ArcContours to grab heads directly
    assert(heads.size() >= 1); // Shouldn't have an empty contour
    assert(is_same_vid(heads.front().iid, heads.back().iid)); // We assume first head is repeated at tail for a closed contour

    // Get a pair of iterators for our input range to track which inputs we have already processed
    auto input_front = heads.begin();
    auto input_back = heads.end()-1; // Input back is the last input, not the end

    if(contour.is_closed()) {
      --input_back; // For a closed contour we have a redundant head at the end which we can ignore

      // We start by lopping off any redundant heads at back of input which need special treatment to handle wrap around
      while(input_front < input_back && is_middle_head_redundant(*(input_back-1),*input_back,*input_front,*this,incident_order)) {
        --input_back; // 'Pop' off redundant heads from our input range
      }

      result.start_contour();
      // We walk over all inputs up until (but not including) input_back (which we handle separately)
      for(SignedArcHead prev_head = *input_back;input_front != input_back; ++input_front) {
        if(!is_middle_head_redundant(prev_head, *input_front, *(input_front+1),*this,incident_order)) {
          result.append_to_back(*input_front);
          prev_head = *input_front;
        }
      }
      assert(input_front == input_back);
      result.append_to_back(*input_back); // Because we trimmed end of input before, we know that we need to keep input_back
      result.end_closed_contour(); // Add back the redundant head to make the new result closed like the input was
    }
    else {
      result.start_contour();
      SignedArcHead prev_head = *input_front;
      result.append_to_back(*input_front); // Always keep first vertex
      ++input_front;
      for(;input_front != input_back;++input_front) {
        if(!is_middle_head_redundant(prev_head, *input_front, *(input_front+1), *this, incident_order)) {
          result.append_to_back(*input_front);
          prev_head = *input_front;
        }
      }
      result.end_open_contour(to_vid(input_back->iid)); // And always keep last vertex
    }
  }
  assert(result.size() == contours.size());
  return result;
}

template<Pb PS> ArcContours PlanarArcGraph<PS>::edges_to_closed_contours(const Nested<const HalfedgeId> edges) const {
  ArcContours result;
  for(const auto c : edges) {
    result.start_contour();
    for(const auto e : c) {
      result.append_to_back({src(e), arc_direction(e)});
    }
    result.end_closed_contour();
  }
  return result;
}

template<Pb PS> static Vector<CircleArc, 2> unquantize_circle(const Quantizer<real,2>& quant, const ExactCircle<PS>& c, const ArcDirection d) {
  const auto q = real(sign(d));
  const Vec2 rhs = quant.inverse(c.center + Vec2(c.radius, 0));
  const Vec2 lhs = quant.inverse(c.center - Vec2(c.radius, 0));
  return vec(CircleArc(rhs, q), CircleArc(lhs, q));
}

template<Pb PS> Nested<CircleArc> VertexSet<PS>::unquantize_circle_arcs(const Quantizer<real,2>& quant, const ArcContours& contours) const {
  Nested<CircleArc, false> result;
  for(const auto contour : contours) {
    assert(contour.is_closed());
    const int n = contour.n_arcs();
    if(n == 1) {
      const auto a = *(contour.begin());
      result.append(unquantize_circle(quant, this->circle(reference_cid(a.head())), a.direction()));
      continue;
    }
    result.append_empty();
    bool can_cull = (n >= 3);

    for(const SignedArcInfo sa : contour) {
      assert(!sa.is_full_circle());
      if(can_cull && reference(sa.head()).radius == helper_circle_radius()) {
        can_cull = false;
      }
      else {
        can_cull = true;
        const auto unsigned_arc = this->arc(ccw_arc(sa));
        const auto x0 = quant.inverse(incident(sa.head()).approx.guess());
        result.append_to_back(CircleArc(x0, sign(sa.direction()) * unsigned_arc.q()));
      }
    }

    // Quantization can introduce tiny self intersecting loops that remain after splitting so we attempt to cull any miniscule slivers.
    // This doesn't feel like a robust solution, but I don't know a better alternative. Future versions of this function might leave slivers for caller to handle.
    if(result.back().size() <= 3) {
      // If the entire arc is thin, area will be less than thickness * perimeter / 2
      // We can use diagonal of bounding box as an estimate for perimeter / 2
      const real d = approximate_bounding_box(result.back()).sizes().magnitude();
      const real a = circle_arc_area(result.back());
      if(abs(a) < d * quant.inverse.unquantize_length(2*constructed_arc_endpoint_error_bound())) {
        result.pop_back();
      }
    }
  }
  return result.freeze();
}

// For an ArcContour we always keep an extra head even for closed contours so that end of arc can always be found by looking at next address
// Since we should never need to access direction flag on the last head, we use that to mark if contour is closed
// This lets us differentiate between a closed contour and an open contour that happens to end at the same vertex where it started
bool RawArcContour::is_closed() const {
  const bool result = (heads.back().direction == heads.front().direction);
  assert(!result || is_same_vid(heads.front().iid,heads.back().iid)); // Check that marking of direction flag is consistent with end vertices
  return result;
}

void ArcContours::end_closed_contour() {
  assert(!store.empty() && !store.back().empty());
  store.append_to_back(store.back().front());
  assert(back().is_closed());
}

void ArcContours::end_open_contour(const VertexId end_vertex) {
  const auto final_dir = store.back().empty() ? ArcDirection() : -store.back().front().direction;
  store.append_to_back({iid_cl(end_vertex), final_dir});
  assert(!back().is_closed());
}

template<Pb PS> Nested<CircleArc> PlanarArcGraph<PS>::unquantize_circle_arcs(const Quantizer<real,2>& quant, const Nested<const HalfedgeId> contours) const {
  const auto arc_contours = combine_concentric_arcs(edges_to_closed_contours(contours));
  return vertices.unquantize_circle_arcs(quant, arc_contours);
}

////////////////////////////////////////////////////////////////////////////////

template<Pb PS> static void sort_circle_incidents(NestedField<IncidentId, CircleId>& circle_incidents, const VertexSet<PS>& verts) {
  for(const auto& incidents : circle_incidents.raw) {
    if(incidents.empty())
      continue;
    // We radix sort quadrants first to make things easy for final sort
    const auto start_of_q2 = std::partition(incidents.begin(), incidents.end(), [&verts](const IncidentId iid) { return verts.incident(iid).q < 2; });
    const auto start_of_q1 = std::partition(incidents.begin(), start_of_q2,     [&verts](const IncidentId iid) { return verts.incident(iid).q == 0; });
    const auto start_of_q3 = std::partition(start_of_q2,       incidents.end(), [&verts](const IncidentId iid) { return verts.incident(iid).q == 2; });
    const ExactCircle<PS> c = verts.reference(incidents.front());
    const auto cmp = [&verts, c](const IncidentId iid0, const IncidentId iid1) { return (iid0 != iid1) && c.intersections_ccw_same_q(verts.incident(iid0),verts.incident(iid1)); };
    std::sort(incidents.begin(), start_of_q1,     cmp);
    std::sort(start_of_q1,       start_of_q2,     cmp);
    std::sort(start_of_q2,       start_of_q3,     cmp);
    std::sort(start_of_q3,       incidents.end(), cmp);
  }
}

template<Pb PS> static NestedField<IncidentId, CircleId> get_circle_orders(const VertexSet<PS>& vertices) {
  auto circle_counts = Field<int,CircleId>(vertices.n_circles());
  for(const IncidentId iid : vertices.incident_ids()) {
    ++circle_counts[vertices.reference_cid(iid)];
  }
  auto result = NestedField<IncidentId, CircleId>(circle_counts, uninit);
  for(const IncidentId iid : vertices.incident_ids()) {
    const CircleId cid = vertices.reference_cid(iid);
    result[cid][--circle_counts[cid]] = iid;
  }
  assert(circle_counts.flat.contains_only(0));
  sort_circle_incidents(result, vertices);
  return result;
}

static Field<int, IncidentId> inverted_permutation(const RawArray<const IncidentId> sorted_iids) {
  auto result = Field<int, IncidentId>(sorted_iids.size(), uninit);
  for(const int offset : range(sorted_iids.size())) {
    const auto iid = sorted_iids[offset];
    result[iid] = offset;
  }
  return result;
}

template<Pb PS> VertexSort<PS>::VertexSort(const VertexSet<PS>& vertices)
 : circle_permutation(get_circle_orders(vertices))
 , incident_permutation(inverted_permutation(circle_permutation.raw.flat))
{}
template<Pb PS> VertexSort<PS>::VertexSort(Uninit)
{ }

template<Pb PS> bool VertexSort<PS>::arc_interior_contains(const UnsignedArcInfo a, const IncidentId i) const {
  const int a_src = psudo_angle(a.src);
  const int a_dst = psudo_angle(a.dst);
  const int a_i = psudo_angle(i);

  // We need to check if psudo_angle wrapped around
  if(a_src < a_dst) {
    // If we don't wrap just check that i is in the range
    // We use a strict less to exclude i==src or i==dst
    return (a_src < a_i) && (a_i < a_dst);
  }
  else if(a_dst < a_src) {
    // If we wrap around we include range above src (up to origin where it wraps around) and below dst
    // As above we use a strict less to exclude i==src or i==dst
    return (a_src < a_i) || (a_i < a_dst);
  }
  else {
    assert(a_src == a_dst);
    assert(a.src == a.dst);
    return true; // Repeated endpoints are a full circle which contains everything
  }
}

template<Pb PS> void ArcAccumulator<PS>::add_full_circle(const ExactCircle<PS>& c, const ArcDirection dir) {
  const auto helper_c = ExactCircle<PS>(c.center + exact::Vec2(c.radius, 0), 1); // Construct a circle that we know will have intersections
  const auto helper_i = c.intersection_min(helper_c); // Build arc to go from that intersection

  const CircleId ref_cid = vertices.get_or_insert(c);
  const CircleId inc_cid = vertices.get_or_insert(helper_c);
  const IncidentId iid = vertices.get_or_insert(helper_i, ref_cid, inc_cid);
  contours.append_and_close(vec(SignedArcHead({iid, dir})));
};

template<Pb PS> void ArcAccumulator<PS>::append_to_back(const SignedArcHead h) {
  assert(contours.back().heads.empty() || vertices.circle_ids(to_vid(h.iid)).contains(vertices.reference_cid(contours.back().heads.back().iid)));
  contours.append_to_back(h);
}

template<Pb PS> void ArcAccumulator<PS>::copy_contours(const ArcContours& src_contours, const VertexSet<PS>& src_vertices) {
  const int base_n = contours.size();
  contours.store.extend(src_contours.store);
  for(const auto i : range(base_n, contours.size())) {
    for(auto& h : contours.store[i]) {
      const auto ref_cid = vertices.get_or_insert(src_vertices.reference(h.iid));
      const auto inc_cid = vertices.get_or_insert(src_vertices.incident(h.iid).as_circle());
      h.iid = vertices.get_or_insert(src_vertices.incident(h.iid), ref_cid, inc_cid);
    }
  }
}
template<Pb PS> Ref<PlanarArcGraph<PS>> ArcAccumulator<PS>::compute_embedding() const {
  return new_<PlanarArcGraph<PS>>(vertices, contours);
}

template<Pb PS> Tuple<Ref<PlanarArcGraph<PS>>,Nested<HalfedgeId>> ArcAccumulator<PS>::split_and_union() const {
  auto result = tuple(new_<PlanarArcGraph<PS>>(vertices, contours), Nested<HalfedgeId>());
  result.y = extract_region(result.x->topology, faces_greater_than(*(result.x), 0));
  return result;
}

template<Pb PS, class Pred> Field<bool, FaceId> find_faces(const PlanarArcGraph<PS>& g, const Pred& p) {
  const auto& t = *(g.topology);
  auto result = Field<bool, FaceId>(t.n_faces(), uninit);
  if(t.n_faces() > 0) {
    const auto depths = compute_winding_numbers(t, g.boundary_face(), g.edge_windings);
    for(const FaceId fid : t.faces())
      result[fid] = p(depths[fid]);
  }
  return result;
}

template<Pb PS> Field<bool, FaceId> faces_greater_than(const PlanarArcGraph<PS>& g, const int depth) {
  return find_faces(g, [depth](const int face_depth) { return face_depth > depth; });
}
template<Pb PS> Field<bool, FaceId> odd_faces(const PlanarArcGraph<PS>& g) {
  return find_faces(g, [](const int face_depth) { return (bool)(face_depth & 1); });
}

template<Pb PS> Ref<PlanarArcGraph<PS>> quantize_circle_arcs(const Quantizer<real,2>& quant, const Nested<const CircleArc> arcs) {
  IntervalScope scope;
  auto result = new_<PlanarArcGraph<PS>>(uninit);
  auto new_arcs = result->vertices.quantize_circle_arcs(quant, arcs);
  result->embed_arcs(new_arcs);
  return result;
}

template<Pb PS> Tuple<Quantizer<real,2>,Ref<PlanarArcGraph<PS>>> quantize_circle_arcs(const Nested<const CircleArc> arcs) {
  auto bounds = approximate_bounding_box(arcs);
  if(bounds.empty()) bounds = Box<Vec2>::unit_box(); // We generate a non-degenerate box in case input was empty
  const Quantizer<real,2> quant = make_arc_quantizer(bounds);
  return tuple(quant, quantize_circle_arcs<PS>(quant, arcs));
}

#define INSTANTIATE(PS) \
  template class VertexField<PS>; \
  template class CircleSet<PS>; \
  template class VertexSet<PS>; \
  template class CircleTree<PS>; \
  template class VertexSort<PS>; \
  template class PlanarArcGraph<PS>; \
  template struct ArcAccumulator<PS>; \
  template Ref<PlanarArcGraph<PS>> quantize_circle_arcs(const Quantizer<real,2>& quant, const Nested<const CircleArc> arcs); \
  template Tuple<Quantizer<real,2>,Ref<PlanarArcGraph<PS>>> quantize_circle_arcs(const Nested<const CircleArc> arcs); \
  template Field<bool, FaceId> faces_greater_than(const PlanarArcGraph<PS>& g, const int depth); \
  template Field<bool, FaceId> odd_faces(const PlanarArcGraph<PS>& g); \
//INSTANTIATE(Pb::Explicit)
INSTANTIATE(Pb::Implicit)
#undef INSTANTIATE
template<> GEODE_DEFINE_TYPE(PlanarArcGraph<Pb::Implicit>);



} // namespace geode
