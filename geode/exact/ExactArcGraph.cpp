#include <geode/array/NestedField.h>
#include <geode/array/sort.h>
#include <geode/exact/circle_quantization.h>
#include <geode/exact/ExactArcGraph.h>
#include <geode/exact/scope.h>
#include <geode/geometry/BoxTree.h>
#include <geode/geometry/traverse.h>
#include <geode/mesh/ComponentData.h>
#include <geode/structure/Hashtable.h>
#include <cmath>

namespace geode {

namespace {
// An arc with endpoints at intersections of a horizontal and a circle
// With some carefully designed templates and overloads this could share most of its implementation with ExactArc, but for now it doesn't seem worth the added complexity
template<Pb PS> struct ExactHorizontalArc {
  ExactCircle<PS> circle;
  IncidentCircle<PS> i;
  IncidentHorizontal<PS> h;
  bool h_is_src;

  // Construct ccw arc from i to h
  ExactHorizontalArc(const ExactCircle<PS>& _circle, const IncidentCircle<PS>& _i, const IncidentHorizontal<PS>& _h)
   : circle(_circle), i(_i), h(_h)
   , h_is_src(false) { }

  // Construct ccw arc from h to i
  ExactHorizontalArc(const ExactCircle<PS>& _circle, const IncidentHorizontal<PS>& _h, const IncidentCircle<PS>& _i)
   : circle(_circle), i(_i), h(_h)
   , h_is_src(true) { }

  bool contains(const IncidentCircle<PS>& i) const;

  bool contains(const CircleIntersection<PS>& i) const { return contains(i.as_incident_to(circle)); }

  SmallArray<IncidentCircle<PS>,2> intersections_if_any(const ExactCircle<PS>& incident) const;
  SmallArray<IncidentCircle<PS>,2> intersections_if_any(const ExactArc<PS>& a) const;
};

template<Pb PS> bool ExactHorizontalArc<PS>::contains(const IncidentCircle<PS>& i) const {
  const IncidentCircle<PS>& src = this->i;
  const IncidentHorizontal<PS>& dst = h;
  const bool flipped = h_is_src;

  if (src.q != dst.q) { // arc starts and ends in different quadrants
    if (src.q == i.q)
      return flipped ^ circle.intersections_ccw_same_q(src, i);
    else if (dst.q == i.q)
      return flipped ^ circle.intersections_ccw_same_q(i, dst);
    else
      return flipped ^ (((i.q-src.q)&3)<((dst.q-src.q)&3));
  } else { // arc starts and ends in the same quadrant
    const bool small = circle.intersections_ccw_same_q(src, dst);
    return flipped ^ small ^ (   src.q != i.q
                               || (small ^ circle.intersections_ccw_same_q(src,i))
                               || (small ^ circle.intersections_ccw_same_q(i,dst)));
  }
}

template<Pb PS> Box<exact::Vec2> bounding_box(const ExactHorizontalArc<PS>& a) {
  const IncidentCircle<PS>& src = a.i;
  const IncidentHorizontal<PS>& dst = a.h;
  const bool flipped = a.h_is_src;

  // We start with the bounding box of the endpoints
  auto box = Box<exact::Vec2>::combine(src.box(),dst.box());

  if(src.q == dst.q) {
    // If src and dst are in same quadrant, arc will either hit all 4 axis or none
    if(flipped ^ !a.circle.intersections_ccw_same_q(src, dst))
      return bounding_box(a.circle);
    else
      return box;
  }
  else {
    // If src and dst are in different quadrants we update each crossed axis
          auto q     = a.h_is_src ? a.h.q : a.i.q;
    const auto dst_q = a.h_is_src ? a.i.q : a.h.q;
    do {
      q = (q+1)&3; // Step to next quadrant
      switch(q) {
        // Add start of new quadrant
        // Arc bounding box must be a subset of the circle bounding box so we can directly update each axis as we cross into the quadrant
        case 0: box.max.x = a.circle.center.x + a.circle.radius; break;
        case 1: box.max.y = a.circle.center.y + a.circle.radius; break;
        case 2: box.min.x = a.circle.center.x - a.circle.radius; break;
        case 3: box.min.y = a.circle.center.y - a.circle.radius; break;
        GEODE_UNREACHABLE();
      }
    } while(q != dst_q); // Go until we end up at dst
    return box;
  }
}

} // anonymous namespace

static Array<int> overlapping_leaves(const BoxTree<Vec2>& tree, const Box<Vec2>& b) {
  assert(tree.leaf_size == 1);
  struct Visitor {
    const BoxTree<Vec2>& tree;
    const Box<Vec2> b;
    Array<int> found;
    bool cull(const int n) {
      return !b.intersects(tree.boxes[n]);
    }
    void leaf(const int n) {
      assert(b.intersects(tree.boxes[n])); // Cull should catch this
      found.extend(tree.prims(n));
    }
  };
  auto v = Visitor({tree, b});
  single_traverse(tree, v);
  return v.found;
}

static Array<int> make_next_i(const int n) {
  auto result = Array<int>(n, uninit);
  for(int i_prev = n-1, i_curr = 0; i_curr < n; i_prev = i_curr++) {
    result[i_prev] = i_curr;
  }
  return result;
}

////////////////////////////////////////////////////////////////////////////////

template<Pb PS> static ExactCircle<PS> helper_circle(const exact::Vec2 center) {
  return ExactCircle<PS>(center,  constructed_arc_endpoint_error_bound());
}

////////////////////////////////////////////////////////////////////////////////

template<Pb PS> static inline bool left_of_center(const IncidentCircle<PS>& i) { return i.q == 1 || i.q == 2; }

namespace {

// These both track directions of halfedges, but have seperate enums to make usage clearer

enum class ArcDirection : bool { CCW = false, CW = true }; // directed_edge assumes these values are matched to behavior in HalfedgeGraph
enum class CircleFace : bool { interior = false, exterior = true };

// For an ExactArcGraph we use forward halfedges as CCW arcs and reversed halfedges as CW arcs
inline HalfedgeId directed_edge(const EdgeId eid, const ArcDirection direction) { assert(eid.valid()); return HalfedgeId(eid.idx()<<1 | static_cast<bool>(direction)); }
inline ArcDirection arc_direction(const HalfedgeId hid) { assert(hid.valid()); return static_cast<ArcDirection>(hid.idx() & 1); }
// inline HalfedgeId ccw_edge(const EdgeId eid) { return directed_edge(eid, ArcDirection::CCW); }
inline HalfedgeId cw_edge (const EdgeId eid) { return directed_edge(eid, ArcDirection::CW ); }
inline bool is_ccw(const HalfedgeId hid) { return arc_direction(hid) == ArcDirection::CCW; }

// The interior face of a circle is adjacent to the ccw/forward halfedges
constexpr ArcDirection interior_edge_dir = ArcDirection::CCW;
// The exterior face of a circle is adjacent to the cw/reversed halfedges
constexpr ArcDirection exterior_edge_dir = ArcDirection::CW;

// For a ray traveling left to right along i, this returns the direction for halfedge that gets hit at i before crossing e
template<Pb PS> inline ArcDirection front_edge_dir(const IncidentHorizontal<PS>& i) {
  return i.left ? interior_edge_dir  // Entering circle from left so exterior edge comes first
                : exterior_edge_dir; // Leaving circle to the right so interior edge comes first
}

// What direction from i along reference crosses into i.as_circle()
template<Pb PS> inline ArcDirection dir_into_incident(const IncidentCircle<PS>& i) { return cl_is_reference(i.side) ? ArcDirection::CCW : ArcDirection::CW; }

template<Pb PS> inline ArcDirection front_edge_dir(const ArcDirection reference_dir, const IncidentCircle<PS>& i) {
  return (cl_is_reference(i.side) ^ (reference_dir != ArcDirection::CCW)) ? ArcDirection::CW
                                                                          : ArcDirection::CCW;
}

inline CircleFace circle_face(const HalfedgeId hid) { return is_ccw(hid) ? CircleFace::interior : CircleFace::exterior; }

template<Pb PS> inline CircleFace face_at_src(const ExactArc<PS>& a) {
  return cl_is_reference(a.src.side) ? CircleFace::interior : CircleFace::exterior;
}
template<Pb PS> inline CircleFace face_at_dst(const ExactArc<PS>& a) {
  return cl_is_reference(a.dst.side) ? CircleFace::exterior : CircleFace::interior;
}

// For a path that travels either just inside or just outside of a circle, find any crossing of a given arc
// This will include endpoints of the arc as long as the arc extends into circle across the path
template<Pb PS> static inline SmallArray<IncidentCircle<PS>,2> path_adjacent_intersections(const ExactCircle<PS>& path_circle, const CircleFace path_face, const ExactArc<PS>& a) {
  SmallArray<IncidentCircle<PS>,2> result;
  // We get all circle intersections and filter out ones on this arc
  for(const auto& i : a.circle.intersections_if_any(path_circle)) {
    // We check if i is an endpoint of arc so that we handle coincident geometry specially
    if(a.circle.is_same_intersection(i, a.src)) {
      if(path_face == face_at_src(a)) // Check if arc extends off of circle to cross over the path
        result.append(i.reference_as_incident(a.circle));
    }
    else if(a.circle.is_same_intersection(i, a.dst)) {
      if(path_face == face_at_dst(a))
        result.append(i.reference_as_incident(a.circle));
    }
    else if(a.unsafe_contains(i)) {
      result.append(i.reference_as_incident(a.circle));
    }
  }
  return result;
}

} // anonymous namespace

namespace {
template<Pb PS> struct VerticalSort {
  ExactCircle<PS> reference;
  template<class T> bool operator()(const Tuple<IncidentCircle<PS>,T>& i0, const Tuple<IncidentCircle<PS>,T>& i1) const {
    return !reference.is_same_intersection(i0.x, i1.x) && reference.intersections_upwards(i0.x, i1.x);
  }
};
} // anonymous namespace

namespace {
template<Pb PS> struct RightwardRaycast {
  const ExactArcGraph<PS>& g;
  const HorizontalIntersection<PS> src;
  const CircleFace starting_face;
  const BoxTree<exact::Vec2>& edge_tree;
  const real min_x;
  RightwardRaycast(const ExactArcGraph<PS>& _g, const HorizontalIntersection<PS> _src, const CircleFace _starting_face, const BoxTree<exact::Vec2>& _edge_tree)
   : g(_g)
   , src(_src)
   , starting_face(_starting_face)
   , edge_tree(_edge_tree)
   , min_x(src.x.box().min)
  { }
  Array<Tuple<HorizontalIntersection<PS>,HalfedgeId>> found;

  bool cull(const int n) const {
    const auto box = edge_tree.boxes(n);
    return box.max.y<src.line.y || box.min.y>src.line.y || box.max.x<min_x;
  }

  void leaf(const int n) {
    assert(edge_tree.prims(n).size()==1);
    const EdgeId eid = EdgeId(edge_tree.prims(n)[0]);
    const auto a = g.arc(eid);
    for(const auto& hit : intersections_if_any(a, src.line)) {
      if(is_same_intersection(hit, src)) {
        const CircleFace face_before_i = hit.left ? CircleFace::exterior : CircleFace::interior;
        if(starting_face != face_before_i)
          continue;
      }
      else if(!intersections_rightwards(src, hit)) {
        continue; // Skip any intersections not to the right
      }
      // If entering circle from the left, we hit reversed halfedge first
      // If leaving circle to the right, we hit forward halfedge first
      const bool which = hit.left;
      found.append(tuple(HorizontalIntersection<PS>(hit, a.circle), HalfedgeGraph::halfedge(eid, which)));
    }
  }
};
} // anonymous namespace

template<Pb PS> static Array<HalfedgeId> rightwards_raycast(const ExactArcGraph<PS>& g, const HorizontalIntersection<PS>& start, const CircleFace starting_face, const BoxTree<exact::Vec2>& edge_tree) {
  auto raycast = RightwardRaycast<PS>(g, start, starting_face, edge_tree);
  single_traverse(edge_tree,raycast);
  sort(raycast.found);
  return raycast.found.template project<HalfedgeId, &Tuple<HorizontalIntersection<PS>,HalfedgeId>::y>().copy();
}

// Construct a path from a point immediately adjacent to seed_he to infinity tracking all edge crossings
// In most cases a horizontal raycast that intersects the seeding edge's arc will be sufficient
// For nearly degenerate cases other constructs will be used to track crossings
template<Pb PS> Array<HalfedgeId> ExactArcGraph<PS>::path_to_infinity(const HalfedgeId seed_he, const BoxTree<exact::Vec2>& edge_tree) const {

  // First we attempt a simple horizontal raycast
  const auto seed_a = arc(graph->edge(seed_he));
  const auto& seed_c = seed_a.circle;
  const auto arc_bounds = bounding_box(seed_a);
  const auto seed_face = circle_face(seed_he); // Use edge direction to see if we are inside or outside

  // Any Horizontal with a y value in this range will be sure to intersect the seed circle
  const auto safe_y = Box<Quantized>(seed_c.center.y).thickened(seed_c.radius - 1);
  const auto h = ExactHorizontal<PS>(safe_y.clamp(Quantized(std::round(arc_bounds.vertical_box().center().x))));

  assert(has_intersections(seed_c, h));

  const auto seed_circle_hits = seed_c.get_intersections(h); // Our choice of h should ensure intersections exist

  for(const auto& i : seed_circle_hits) {
    if(!seed_a.contains_horizontal(i))
      continue; // Skip through intersections to try and find one on the arc
    // We found intersection on the target edge so we use a rightwards raycast directly
    return rightwards_raycast(*this, HorizontalIntersection<PS>(i, seed_c), seed_face, edge_tree);
  }

  // If we didn't return somewhere above, our arc must be too small so we need to extend it until it touches the horizontal
  // Since arc doesn't intersect the horizontal, we should have either both endpoints below, or both above
  assert(seed_c.intersections_upwards(seed_a.src, seed_circle_hits[0]) == seed_c.intersections_upwards(seed_a.dst, seed_circle_hits[0]));

  const bool arc_below = seed_c.intersections_upwards(seed_a.src, seed_circle_hits[0]); // We can check one of the two endpoints to find direction to horizontal
  const bool dst_closer = seed_c.intersections_upwards(seed_a.src, seed_a.dst) == arc_below; // We can compare the arc endpoints to determine which seeds closer to the horizontal
  const auto path_dir = dst_closer ? ArcDirection::CCW // Path will continue CCW around circle from seed_a.dst to h
                                   : ArcDirection::CW; // Path will go back CW around circle from seed_a.src to h

  const auto& path_start = dst_closer ? seed_a.dst : seed_a.src;   // Path will start at closest arc endpoint
  const auto& path_end = seed_circle_hits[left_of_center(path_start)]; // Path will end on the closest horizontal intersection to start

  assert(path_end.left == left_of_center(path_start)); // Check that order of hits is as expected

  const auto path_arc = (path_dir == ArcDirection::CCW) ? ExactHorizontalArc<PS>(seed_c, path_start, path_end)  // Continue ccw from seed_a.dst to h
                                                        : ExactHorizontalArc<PS>(seed_c, path_end, path_start); // Flip endpoint to get canonical (ccw) order

  // We will gather all edges crossed by our circular path
  // This is not quite the edges that cross the path's circle since edges that end at the circle are might also be included depending on their face
  auto path_crossings = Array< Tuple< IncidentCircle<PS>, HalfedgeId> >();
  for(const int leaf : overlapping_leaves(edge_tree, bounding_box(path_arc))) {
    const auto leaf_eid = EdgeId(leaf);
    const auto leaf_a = arc(leaf_eid);
    for(const auto& i : path_adjacent_intersections(seed_c, seed_face, leaf_a)) {
      assert(leaf_eid != HalfedgeGraph::edge(seed_he)); // Shouldn't have intersections on seed arc since it is coincident to path
      if(seed_c.is_same_intersection(path_start, i) ? true // We include endpoints of path arc
                                                    : path_arc.contains(i)) { // and interior of path arc
        const auto front_he = directed_edge(leaf_eid, front_edge_dir(path_dir, i));
        path_crossings.append(tuple(i, front_he));
      }
    }
  }

  // All intersections should be on same left/right half of the circle so we can sort vertically
  sort(path_crossings, VerticalSort<PS>({seed_c})); // from bottom to top
  if(!arc_below) // If arc started above horizontal, our sort is backwards so we need to reverse order
    path_crossings.reverse();

  // Extract out halfedges
  Array<HalfedgeId> result;

  // We add all of the intersections following path down to the horizontal
  result.extend(path_crossings.template project<HalfedgeId, &Tuple<IncidentCircle<PS>,HalfedgeId>::y>());

  // After tracing to the horizontal intersection we can use a horizontal raycast to get the rest of the way to infinity
  result.extend(rightwards_raycast(*this, HorizontalIntersection<PS>(path_end, seed_c), seed_face, edge_tree));

  return result;
}

namespace {
template<Pb PS> struct NewArc {
  EdgeId src_id;
  HalfedgeId x0_helper;
  real q;
  exact::Vec2 x0;
  ExactCircle<PS> circle;
  IncidentCircle<PS> x0_inc;
  IncidentCircle<PS> x1_inc;
  ArcDirection direction;

  NewArc(const EdgeId _src_id, const exact::Vec2 _x0, const real _q)
   : src_id(_src_id)
   , q(_q)
   , x0(_x0)
  { }

  void set_circle_and_direction(const exact::Vec2 x1) {
    const auto c_and_r = construct_circle_center_and_radius(x0, x1, q);
    circle.center = c_and_r.x;
    circle.radius = c_and_r.y;
    static_assert(PS != Pb::Explicit, "Error: Must set circle perturbation seed");
    direction = (q >= 0.) ? ArcDirection::CCW : ArcDirection::CW;
    assert(!is_same_circle(circle, helper_circle<PS>(x0)));
  }

  // These find the first/last intersection relative to direction of arc
  // first_intersection is where arc enters incident and last_intersection is where arc leaves incident
  // Assuming start of arc isn't inside incident, using first_intersection as end will have a shorter arc than last_intersection
  IncidentCircle<PS> first_intersection(const ExactCircle<PS>& incident) const {
    return (direction == ArcDirection::CCW) ? circle.intersection_min(incident) : circle.intersection_max(incident);
  }
  IncidentCircle<PS> last_intersection(const ExactCircle<PS>& incident) const {
    return (direction != ArcDirection::CCW) ? circle.intersection_min(incident) : circle.intersection_max(incident);
  }

  // Choose a vertex for the start of the arc
  void set_x0_inc(const ExactCircle<PS>& prev_circle) {
    for(const auto& i : intersections_if_any(prev_circle, circle)) {
      if(i.is_inside(helper_circle<PS>(x0))) {
        // If we find an intersection inside of the helper circle use it
        x0_inc = i.as_incident_to(circle);
        return;
      }
    }
    // By default we just use the intersection furthest along this arc
    x0_inc = last_intersection(helper_circle<PS>(x0));
  }

  void set_x1_inc(const NewArc& next) {
    if(is_same_circle(circle, next.x0_inc)) { // If next arc starts at intersection with this circle we use that vertex
      x1_inc = next.x0_inc.reference_as_incident(next.circle); // We have to reverse it to get intersection relative to current circle
      return;
    }

    if(is_same_circle(circle, next.circle)) { // If the next arcs continues along the current circle
      // Merging these two arcs is hard since we need to track cases where we wrap all the way around or reverse directions
      // Instead of merging we just reuse intersection with a helper circle on next arc
      assert(is_same_circle(next.x0_inc, helper_circle<PS>(next.x0))); // We should already have intersection from when we computed next.x0_inc
      // WARNING: It's important that we reuse same intersection point (otherwise we would have a gap between this and next)
      x1_inc = next.x0_inc;
      return;
    }

    // If the above cases fall through we compute a new intersection at the helper circle
    x1_inc = first_intersection(helper_circle<PS>(next.x0));
  }
};
} // anonymous namespace

template<Pb PS> ExactArcGraph<PS>::ExactArcGraph()
 : graph(new_<HalfedgeGraph>())
{
}
template<Pb PS> ExactArcGraph<PS>::~ExactArcGraph() {
}


template<Pb PS> Nested<HalfedgeId> ExactArcGraph<PS>::quantize_and_add_arcs(const Quantizer<real,2>& quant, const Nested<const CircleArc>& src_arcs) {
  IntervalScope scope;
  Nested<HalfedgeId, false> result;
  for(const int i : range(src_arcs.size())) {
    result.append(quantize_and_add_arcs(quant, src_arcs[i], EdgeId(src_arcs.offsets[i])));
  }
  return result.freeze();
}

template<Pb PS> Array<HalfedgeId> ExactArcGraph<PS>::quantize_and_add_arcs(const Quantizer<real,2>& quant, const RawArray<const CircleArc> src_arcs, const EdgeId base_ref_id) {
  Array<HalfedgeId> new_contour;
  assert(base_ref_id.valid());
  const int src_n = src_arcs.size();
  const auto src_next = make_next_i(src_n);

  // Filter out degenerate source arcs and seed initial values
  auto new_arcs = Array<NewArc<PS>>();
  {
    if(src_n == 0) // Make sure we have a front value we an access
      return new_contour;
    exact::Vec2 x0 = quant(src_arcs.front().x);
    for(const auto src_i : range(src_n)) {
      exact::Vec2 x1 = quant(src_arcs[src_next[src_i]].x);
      if(x0 == x1 || has_intersections(helper_circle<PS>(x0), helper_circle<PS>(x1)))
        continue; // Skip over small degenerate arcs
      // If we skipped arcs, x0 won't be same as src_arcs[src_i].x, but it should have changed by at most the helper circle diameter + 1 so it should be safe to use
      new_arcs.append(NewArc<PS>(EdgeId(base_ref_id.idx() + src_i), x0, src_arcs[src_i].q));
      x0 = x1;
    }
  }

  // If after filtering we have a single point, discard it
  if(new_arcs.size() <= 1)
    return new_contour;

  const int new_n = new_arcs.size();
  for(int curr_i = new_n-1, next_i = 0; next_i < new_n; curr_i = next_i++) {
    // Construct all of the circles
    new_arcs[curr_i].set_circle_and_direction(new_arcs[next_i].x0);
  }
  for(int prev_i = new_n-1, curr_i = 0; curr_i < new_n; prev_i = curr_i++) {
    // Construct starting vertices
    new_arcs[curr_i].set_x0_inc(new_arcs[prev_i].circle);
  }
  for(int curr_i = new_n-1, next_i = 0; next_i < new_n; curr_i = next_i++) {
    // Either reuse starting vertices or construct new end ones
    new_arcs[curr_i].set_x1_inc(new_arcs[next_i]);
  }

  for(int prev_i = new_n-1, curr_i = 0; curr_i < new_n; prev_i = curr_i++) {
    // Although we have a series of arcs and their endpoints, some neighbors might need helper arcs inserted between them
    auto& curr = new_arcs[curr_i];
    auto& prev = new_arcs[prev_i];

    // Check if end of previous arc is start of current arc
    if(!is_same_intersection(prev.circle, prev.x1_inc, curr.circle, curr.x0_inc)) {
      // If endpoints aren't the same, we will need to construct a helper arc
      const auto hc = helper_circle<PS>(curr.x0); // Helper arc will use this circle

      // Both sides should be using intersections with a helper circle at curr.x0
      assert(is_same_circle(hc, prev.x1_inc));
      assert(is_same_circle(hc, curr.x0_inc));

      // Inputs to quantization will usually need to undergo a union (to remove self intersections) before further operations are safe
      // In this case quantization will usually be on positive area polyarcs. Under the assumption that eroding small features is better
      // than adding them we want add helper arcs with a negative winding

      // Canonical arcs are ccw so we connect prev to dst and curr to src if we are going to use a negative winding
      // We compute all options and choose the best one
      const auto src_options = hc.get_intersections(curr.circle);
      const auto dst_option_0 = prev.x1_inc.reference_as_incident(prev.circle);
      const auto dst_option_1 = hc.other_intersection(dst_option_0);

      assert(!hc.is_same_intersection(dst_option_0, dst_option_1));

      // Guess first option on each side
      auto helper_arc = ExactArc<PS>({hc, src_options[0], dst_option_0});

      // If curr and prev had an intersection inside we wouldn't need the helper arc
      // This implies that the ccw order or intersections will be a rotation of src_option[a], src_option[!a], dst_option[b], dst_option[!b] for some a and b
      if(helper_arc.unsafe_contains(src_options[1]))
        helper_arc.src = src_options[1]; // If src_option[0] was first, switch to it
      if(helper_arc.unsafe_contains(dst_option_1)) {
        helper_arc.dst = dst_option_1; // If dst_option[1] was first, switch to it
        prev.x1_inc = dst_option_1.reference_as_incident(hc);
      }

      // We need to make sure curr starts at end of helper_arc
      curr.x0_inc = helper_arc.src.reference_as_incident(helper_arc.circle);

      assert(is_same_intersection(prev.circle, prev.x1_inc, helper_arc.circle, helper_arc.dst));
      assert(is_same_intersection(curr.circle, curr.x0_inc, helper_arc.circle, helper_arc.src));

      const int weight = 1;
      const int winding = -1; // As explained above, we use negative winding to try to minimize features added by quantization
      const EdgeId new_edge = add_arc(helper_arc, EdgeValue(weight,winding));
      curr.x0_helper = cw_edge(new_edge);
    }
  }

  for(const auto& curr : new_arcs) {
      // Convert 'curr' into an exact arc with the correct direction/winding
    const auto arc = (curr.direction == ArcDirection::CCW) ? ExactArc<PS>({curr.circle, curr.x0_inc, curr.x1_inc})
                                                           : ExactArc<PS>({curr.circle, curr.x1_inc, curr.x0_inc});
    const int weight = 1;
    const int winding = (curr.direction == ArcDirection::CCW) ? 1 : -1;
    const EdgeId new_edge = add_arc(arc, EdgeValue(weight, winding));
    if(curr.x0_helper.valid()) {
      new_contour.append(curr.x0_helper);
    }
    new_contour.append(directed_edge(new_edge, curr.direction));
  }
  return new_contour;
}

template<Pb PS> bool is_same_intersection(const ExactCircle<PS>& ref0, const IncidentCircle<PS>& inc0, const ExactCircle<PS>& ref1, const IncidentCircle<PS>& inc1) {
  if(inc0.side == inc1.side)
    return is_same_circle(ref0, ref1) && is_same_circle(inc0, inc1);
  else
    return is_same_circle(ref0, inc1) && is_same_circle(inc0, ref1);
}

namespace {
template<Pb PS> struct CompareInc {
  const ExactArcGraph<PS>& g;
  const ExactCircle<PS> c;
  bool operator()(const IncidentId i0, const IncidentId i1) const {
    assert(is_same_circle(c, g.reference(i0)) && is_same_circle(c, g.reference(i1)));
    return (i0 != i1) && c.intersections_sorted(g.incident(i0), g.incident(i1));
  }
};
} // anonymous namespace

template<Pb PS> Field<IncidentId, IncidentId> ExactArcGraph<PS>::ccw_next() const {
  IdSet<ExactCircle<PS>, CircleId> circles; // Track duplicate references to the same circle

  auto i_to_c = Field<CircleId, IncidentId>(vertices.size() * 2, uninit);
  const auto incident_ids = i_to_c.id_range();

  Field<int, CircleId> counts;

  // Get unique circles and count incident vertices
  for(const IncidentId iid : incident_ids) {
    const CircleId cid = circles.get_or_insert(reference(iid));
    i_to_c[iid] = cid;
    counts.grow_until_valid(cid);
    ++counts[cid];
  }
  const auto circle_ids = circles.id_range();

  // Gather incident vertices for each circle
  auto circle_inc = NestedField<IncidentId, CircleId>(counts);
  for(const IncidentId iid : incident_ids) {
    const CircleId cid = i_to_c[iid];
    circle_inc[cid][--counts[cid]] = iid;
  }

  assert(counts.flat.contains_only(0)); // Check that we got back to starting state

  // Sort incident vertices on a per-circle basis
  for(const CircleId cid : circle_ids) {
    const auto curr_inc = circle_inc[cid];
    sort(circle_inc[cid], CompareInc<PS>({*this, circles[cid]}));
  }

  auto result = Field<IncidentId, IncidentId>(incident_ids.size(), uninit);

  for(const auto cid : circle_ids) {
    const auto cv = circle_inc[cid];
    if(cv.empty())
      continue;
    IncidentId prev_id = cv.back();
    for(const IncidentId curr_id : cv) {
      result[prev_id] = curr_id;
      prev_id = curr_id;
    }
  }
  return result;
}

// Replace all coincident edges with non-overlapping edges
template<Pb PS> void ExactArcGraph<PS>::new_noncoincident_edges() {
  const auto incident_ids = this->incident_ids();
  const auto n_incidents = this->n_incidents();
  const auto ccw_next = this->ccw_next();
  // Get value at each incident edge
  auto incident_values = Field<EdgeValue, IncidentId>(n_incidents);

  for(const EdgeId eid : edge_ids()) {
    const IncidentId edge_src = src(eid);
    const IncidentId edge_dst = dst(eid);
    const EdgeValue ev = edges[eid].value;

    // Walk over all vertices in the edge and update them
    bool first = true;
    for(IncidentId c = edge_src; first || c != edge_dst; c = ccw_next[c]) {
      incident_values[c] += ev;
      first = false;
    }
  }

  // We will create a completely new graph
  auto new_graph = new_<HalfedgeGraph>();
  auto new_edges = Field<EdgeInfo, EdgeId>();
  // We will use the same vertices
  new_graph->add_vertices(n_vertices());

  for(const IncidentId src : incident_ids) {
    const auto& v = incident_values[src];
    assert(v.weight >= 0 && v.weight <= abs(v.winding));
    if(v.weight == 0)
      continue;
    const IncidentId dst = ccw_next[src];

    GEODE_UNUSED const EdgeId new_eid = new_graph->unsafe_add_edge(to_vid(src), to_vid(dst));
    new_edges.append(EdgeInfo({v, side(src), side(dst) }));
    assert(new_edges.id_range().back() == new_eid);
  }

  edges = new_edges;
  graph = new_graph;
}

template<Pb PS> VertexId ExactArcGraph<PS>::get_or_insert(const CircleIntersectionKey<PS>& i) {
  const VertexId result = vertices.get_or_insert(i);
  if(!graph->valid(result)) {
    graph->add_vertex();
  }
  assert(graph->valid(result));
  return result;
}

template<Pb PS> VertexId ExactArcGraph<PS>::get_or_insert_intersection(const CircleIntersection<PS>& i) {
  const VertexId result = vertices.get_or_insert_intersection(i);
  if(!graph->valid(result)) {
    graph->add_vertex();
  }
  assert(graph->valid(result));
  return result;
}

template<Pb PS> EdgeId ExactArcGraph<PS>::add_arc(const ExactArc<PS>& arc, const EdgeValue value) {
  const VertexId e_src = get_or_insert(CircleIntersectionKey<PS>(arc.circle, arc.src));
  const VertexId e_dst = get_or_insert(CircleIntersectionKey<PS>(arc.circle, arc.dst));
  EdgeId new_e = graph->unsafe_add_edge(e_src, e_dst);
  edges.append(EdgeInfo({value, arc.src.side, arc.dst.side}));
  assert(new_e == edges.id_range().back());
  // Check that we have all the right circles
  assert(is_same_circle(reference(src(new_e)), arc.circle));
  assert(is_same_circle(reference(dst(new_e)), arc.circle));
  assert(is_same_circle(incident(src(new_e)), arc.src));
  assert(is_same_circle(incident(dst(new_e)), arc.dst));
  return new_e;
}

template<Pb PS> EdgeId ExactArcGraph<PS>::add_full_circle(const ExactCircle<PS>& c, const EdgeValue value) {
  // A previous implementation used special logic to track a dummy intersection, but this seems simpler
  const auto helper_c = ExactCircle<PS>(c.center + exact::Vec2(c.radius, 0), 1); // Construct a circle that we know will have intersections
  const auto helper_i = c.intersection_min(helper_c); // Build arc to go from that intersection
  return add_arc(ExactArc<PS>({c, helper_i, helper_i}), value);
}

template<Pb PS> Nested<CircleArc> ExactArcGraph<PS>::unquantize_circle_arcs(const Quantizer<real,2>& quant, const Nested<const HalfedgeId> contours) const {
  Nested<CircleArc, false> result;
  for(const auto contour : contours) {
    assert(contour.size() != 0);
    assert(graph->src(contour.front()) == graph->dst(contour.back())); // Each contour should form a closed loop
    // Full circles need special handling since q value will be undefined for repeated endpoints
    // More generally, any arc larger than a half circle will amplify worst case error (by a factor of abs(arc.q))
    // TODO: To ensure we don't exceed error bounds, perhaps we should subdivide all arcs with large q values?
    if(contour.size() == 1) {
      const HalfedgeId he = contour.front();
      const auto& c = circle(graph->edge(he));
      const int s = graph->is_forward(he) ? 1 : -1;
      const Vec2 rhs = quant.inverse(c.center + Vec2(c.radius, 0));
      const Vec2 lhs = quant.inverse(c.center - Vec2(c.radius, 0));
      result.append_empty();
      result.append_to_back(CircleArc(rhs, real(s)));
      result.append_to_back(CircleArc(lhs, real(s)));
    }
    else {
      result.append_empty();
      bool culled_prev = false;
      for(const int i : range(contour.size())) {
        const HalfedgeId he = contour[i];
        assert(graph->dst(he) == graph->src(contour[(i + 1) % contour.size()])); // Should have continuous contours
        const auto a = arc(graph->edge(he));
        // TODO: We could track total deviation from last used endpoint so that we could cull multiple arcs in a row
        if(!culled_prev && a.circle.radius == constructed_arc_endpoint_error_bound()) {
          culled_prev = true;
          continue;
        }
        else {
          culled_prev = false;
        }
        const auto x = quant.inverse(vertices[graph->src(he)].approx.guess());
        const real q = graph->is_forward(he) ? a.q() : -a.q();
        result.append_to_back(CircleArc(x, q));
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
  }
  return result.freeze();
}

namespace {
template<Pb PS> struct NewEdgeIntersection {
  CircleIntersection<PS> i;
  EdgeId e0, e1;
};

template<Pb PS> struct IntersectionHelper {
  const ExactArcGraph<PS>& g;
  const BoxTree<exact::Vec2>& tree;
  Array<NewEdgeIntersection<PS>> new_intersections;
  bool cull(const int n) const { return false; }
  bool cull(const int n0, const int n1) const { return false; }
  void leaf(const int n) const { assert(tree.prims(n).size()==1); }

  void leaf(const int n0, const int n1) {
    if(n0 == n1) // Only check
      return;
    assert(tree.prims(n0).size()==1 && tree.prims(n1).size()==1);
    const EdgeId e0 = EdgeId(tree.prims(n0)[0]),
                 e1 = EdgeId(tree.prims(n1)[0]);
    const auto a0 = g.arc(e0),
               a1 = g.arc(e1);

    for(const auto& hit : intersections_if_any(a0, a1)) {
      new_intersections.append({hit, e0, e1});
    }
  }
};

template<Pb PS> Array<NewEdgeIntersection<PS>> edge_intersections(const ExactArcGraph<PS>& g, const BoxTree<exact::Vec2>& tree) {
  IntersectionHelper<PS> helper({g, tree});
  double_traverse(tree, helper);
  return helper.new_intersections;
}

template<Pb PS> Ref<BoxTree<exact::Vec2>> make_edge_tree(const ExactArcGraph<PS>& g) {
  Field<Box<exact::Vec2>, EdgeId> boxes(g.n_edges(),uninit);
  for(const EdgeId eid : g.edge_ids()) {
    boxes[eid] = bounding_box(g.arc(eid));
  }
  return new_<BoxTree<exact::Vec2>>(boxes.flat, 1);
}

struct EdgeSplit {
  EdgeId eid;
  IncidentId iid;
  bool operator<(const EdgeSplit& rhs) const { return eid < rhs.eid; }
};

} // anonymous namespace

template<Pb PS> void ExactArcGraph<PS>::split_edges() {
  IntervalScope scope;
  new_noncoincident_edges();

  const auto edge_tree = make_edge_tree(*this);
  const auto hits = edge_intersections(*this, *edge_tree);

  Array<EdgeSplit> splits;
  splits.preallocate(2*hits.size() + 1);

  for(const auto& h : hits) {
    const VertexId new_vid = get_or_insert_intersection(h.i);
    const IncidentId iid0 = incident_id(new_vid, h.i.find(circle(h.e0)));
    const IncidentId iid1 = incident_id(new_vid, h.i.find(circle(h.e1)));
    splits.append_assuming_enough_space({h.e0, iid0});
    splits.append_assuming_enough_space({h.e1, iid1});
  }

  sort(splits); // Will sort by edge to group all splits for the same edge
  splits.append_assuming_enough_space(EdgeSplit()); // Add split with invalid ids to trigger handling on last iteration of loop
  EdgeId current_edge;
  Array<IncidentId> edge_intersections;
  for(const auto& s : splits) {
    if(s.eid != current_edge) { // Are we on a new edge?
      if(!edge_intersections.empty()) { // If we have accumulated vertices we need to split
        assert(current_edge.valid());
        const IncidentId e_dst = dst(current_edge);
        edge_intersections.append(e_dst); // Add end of edge so we can find actual order
        const int n = edge_intersections.size();
        assert(n >= 2); // Should have at least 1 new vertex and dst

        const ExactCircle<PS> ec = circle(current_edge);

        sort(edge_intersections, CompareInc<PS>({*this, circle(current_edge)})); // Sort by angle
        const int dst_i = edge_intersections.find(e_dst);
        assert(edge_intersections.valid(dst_i)); // Must have found the dst vertex

        // Since we sorted by absolute angle, we need to rotate dst_i around to the end
        for(const auto& r : vec(range(dst_i + 1, n), // Start just after dst_i
                                range(0, dst_i))) { // Wrap back to start and go to just before dst_i
          for(const int i : r) {
            current_edge = _unsafe_split_edge(current_edge, edge_intersections[i]);
          }
        }
        edge_intersections.clear(); // Reset for next edge
      }
      current_edge = s.eid;
    }
    edge_intersections.append(s.iid);
  }
  assert(!current_edge.valid()); // Should have hit the invalid edge trigger at the end

  compute_embedding();
}

// Splits an edge at an intersection
// Doesn't ensure correct orientation of halfedges around intersection
template<Pb PS> EdgeId ExactArcGraph<PS>::_unsafe_split_edge(const EdgeId eid0, const IncidentId iid) {
  assert(is_same_circle(reference(iid), circle(eid0))); // Check that incident is on the correct circle
  assert(arc(eid0).unsafe_contains(incident(iid))); // Check that intersection is actually on the edge

  const ReferenceSide s = side(iid);
  const EdgeId eid1 = graph->split_edge(eid0, to_vid(iid));
  GEODE_UNUSED const EdgeId appended_id = edges.append(uninit);
  assert(appended_id == eid1);
  auto& e0 = edges[eid0];
  auto& e1 = edges[eid1];
  e1.value = e0.value; // Copy value
  e1.src_side = s; // New src
  e1.dst_side = e0.dst_side; // Copy dst
  e0.dst_side = s; // dst needs to be updated

  assert(dst(eid0) == iid);
  assert(src(eid1) == iid);
  assert(is_same_circle(circle(eid0), circle(eid1)));

  return eid1;
}

namespace {
template<Pb PS> struct RayCast {
  const ExactArcGraph<PS>& g;
  const ExactHorizontal<PS> y_ray;
  const BoxTree<exact::Vec2>& edge_tree;
  Array<Tuple<HorizontalIntersection<PS>,HalfedgeId>> found;

  bool cull(const int n) const {
    const auto box = edge_tree.boxes(n);
    return box.max.y<y_ray.y || box.min.y>y_ray.y;
  }

  void leaf(const int n) {
    assert(edge_tree.prims(n).size()==1);
    const EdgeId eid = EdgeId(edge_tree.prims(n)[0]);
    const auto a = g.arc(eid);
    for(const auto& hit : intersections_if_any(a, y_ray)) {
      // If entering circle from the left, we hit reversed halfedge first
      // If leaving circle to the right, we hit forward halfedge first
      const bool which = hit.left;
      found.append(tuple(HorizontalIntersection<PS>(hit, a.circle), HalfedgeGraph::halfedge(eid, which)));
    }
  }
};
} // anonymous namespace

// Find all edge crossings for a horizontal
// Halfedges indicate sign of crossing
template<Pb PS> Array<HalfedgeId> ExactArcGraph<PS>::horizontal_raycast(const ExactHorizontal<PS>& y_ray, const BoxTree<exact::Vec2>& edge_tree) const {
  auto ray_cast = RayCast<PS>({*this, y_ray, edge_tree});
  single_traverse(edge_tree,ray_cast);
  sort(ray_cast.found);
  return ray_cast.found.template project<HalfedgeId, &Tuple<HorizontalIntersection<PS>,HalfedgeId>::y>().copy();
}

// Compute halfedge links around a vertex to get a planar embedding
template<Pb PS> void ExactArcGraph<PS>::embed_vertex(const VertexId vid) {
  // Since each edge is positively oriented around its circle, we can determine relative order of incident halfedges just by looking at flags
  // There can be up to 4 outgoing halfedges on a vertex which can be identified by checking which circle they go along and if halfedge is forward
  // We use slots for the 4 possible edges in ccw order: pos_cr, pos_cl, neg_cr, neg_cl

  assert(graph->degree(vid) <= 4); // We require coincident edges to be merged beforehand so should have at most 4 edges

  auto outgoing = Vector<HalfedgeId, 4>();
  int count = 0;
  for(const HalfedgeId hid : graph->outgoing(vid)) {
    const bool fwd = graph->is_forward(hid);
    const ReferenceSide s = fwd ? edges[graph->edge(hid)].src_side : edges[graph->edge(hid)].dst_side;
    const int slot = (!fwd << 1) | (s == ReferenceSide::cr);
    assert(!outgoing[slot].valid());
    outgoing[slot] = hid;
    ++count;
  }


  if(count <= 2)
    return; // If we have 2 or fewer edges they can't be in a bad order and we don't even need to update links

  if(count < 4) {
    assert(count == 3);
    // Scoot values after empty slot in
    for(int dst_i : range(outgoing.find(HalfedgeId()), count))
      outgoing[dst_i] = outgoing[dst_i + 1];
  }

  // Link consecutive ids
  for(int prev_i = count - 1, next_i = 0; next_i < count; prev_i = next_i++) {
    const HalfedgeId prev = graph->reverse(outgoing[prev_i]);
    const HalfedgeId next = outgoing[next_i];
    assert(graph->edge(prev) != graph->edge(next));
    assert(graph->dst(prev) == vid);
    assert(graph->src(next) == vid);
    graph->unsafe_link(prev, next);
  }
}

template<Pb PS> FaceId ExactArcGraph<PS>::boundary_face() const {
  return FaceId(0);
}


template<Pb PS> Box<Vec2> bounding_box(const ExactArcGraph<PS>& g) {
  Box<Vec2> result;
  for(const EdgeId eid : g.edge_ids()) {
    result.enlarge(bounding_box(g.arc(eid)));
  }
  return result;
}

template<Pb PS> void ExactArcGraph<PS>::compute_embedding() {
  for(const VertexId vid : vertex_ids()) {
    embed_vertex(vid);
  }
  auto& g = *graph;
  const auto _edge_tree = make_edge_tree(*this);
  const auto& edge_tree = *_edge_tree;

  g.initialize_borders();
  auto cd = ComponentData(g);
  FaceId infinity_face;

  for(const ComponentId seed_c : cd.components()) {

    // For each component we perform a raycast to check if it is inside some other component
    if(cd[seed_c].exterior_face.valid())
      continue; // If we found the component exterior in a previous test we don't need to keep looking

    const auto seed_he = g.halfedge(cd.border(seed_c));
    Array<HalfedgeId> path_from_infinity = path_to_infinity(seed_he, edge_tree);
    path_from_infinity.reverse(); // Make path walk backwards from infinity
    for(auto& hid : path_from_infinity) hid = g.reverse(hid); // Get opposite edges for each crossing

    // Force seed_he to be updated by adding it as the end of the path
    path_from_infinity.append(seed_he);

    FaceId current_face = infinity_face;
    for(const HalfedgeId hit : path_from_infinity) {
      const BorderId hit_border = g.border(hit);

      // If we haven't created the first face, do so now
      if(!current_face.valid()) {
        infinity_face = g.new_face_for_border(hit_border);
        current_face = infinity_face;
      }

      auto& hit_component = cd[hit_border];
      if(!hit_component.exterior_face.valid())
        hit_component.exterior_face = current_face; // First hit on a component is always its exterior face

      if(!g.face(hit_border).valid())
        g.add_to_face(current_face, hit_border); // Add the border that we hit to the current face if it isn't already part

      assert(g.face(hit_border) == current_face); // Border should already be part of face or have just been added

      // Stepping across edge will switch to a new border...
      const BorderId opp_border = g.border(g.reverse(hit));
      current_face = g.face(opp_border); // ...and also a new face

      assert(cd.component(opp_border) == cd.component(hit_border)); // We should have stayed on the same component when crossing the edge

      if(!current_face.valid()) // If the new border didn't already have a face, we need to add one
        current_face = g.new_face_for_border(opp_border);
    }

    assert(cd[seed_c].exterior_face.valid()); // path should have initialized at least seed

  }

  // Now that we have all possible relative info about each component, we need to generate interior faces that weren't hit by any rays
  // We need to be careful since components that are too small will not have been hit by any rays
  for(const BorderId bid : g.borders()) {
    assert(cd[bid].exterior_face.valid());
    if(!g.face(bid).valid() // Find borders that didn't get assigned a face yet
     && cd[bid].exterior_face.valid()) { // Make sure this wasn't because of a degenerate component
      g.new_face_for_border(bid); // Add a new face
    }
  }

  assert(!infinity_face.valid() || infinity_face == boundary_face());

  // After this function all borders should have a valid face id
}

template<Pb PS> Field<int, FaceId> ExactArcGraph<PS>::face_winding_depths() const {
  auto edge_windings = Field<int, EdgeId>(n_edges(), uninit);
  for(const EdgeId eid : edge_ids()) {
    edge_windings[eid] = edges[eid].value.winding;
  }
  return compute_winding_numbers(graph, boundary_face(), edge_windings);
}

template<Pb PS> Field<bool, FaceId> faces_greater_than(const ExactArcGraph<PS>& g, const int depth) {
  const auto depths = g.face_winding_depths();
  auto result = Field<bool, FaceId>(g.graph->n_faces(), uninit);
  for(const FaceId fid : g.graph->faces())
    result[fid] = (depths[fid] > depth);
  return result;
}

template<Pb PS> Field<bool, FaceId> odd_faces(const ExactArcGraph<PS>& g) {
  const auto depths = g.face_winding_depths();
  auto result = Field<bool, FaceId>(g.graph->n_faces(), uninit);
  for(const FaceId fid : g.graph->faces())
    result[fid] = ((depths[fid] & 1) != 0);
  return result;
}

template<Pb PS> Tuple<Quantizer<real,2>,ExactArcGraph<PS>> quantize_circle_arcs(const Nested<const CircleArc> arcs, const Box<Vec2> min_bounds) {
  auto bounds = approximate_bounding_box(arcs);
  bounds.enlarge(min_bounds);
  if(bounds.empty())
    bounds = Box<Vec2>::unit_box(); // We generate a non-degenerate box in case input was empty
  const auto quant = make_arc_quantizer(bounds);
  return tuple(quant, quantize_circle_arcs<PS>(arcs, quant));
}

template<Pb PS> ExactArcGraph<PS> quantize_circle_arcs(const Nested<const CircleArc> arcs, const Quantizer<real,2> quant) {
  ExactArcGraph<PS> result;
  result.quantize_and_add_arcs(quant, arcs);
  return result;
}


#define INSTANTIATE(PS) \
  template class ExactArcGraph<PS>; \
  template Field<bool, FaceId> faces_greater_than(const ExactArcGraph<PS>& g, const int depth); \
  template Field<bool, FaceId> odd_faces(const ExactArcGraph<PS>& g); \
  template Box<Vec2> bounding_box(const ExactArcGraph<PS>& g); \
  template Tuple<Quantizer<real,2>,ExactArcGraph<PS>> quantize_circle_arcs(const Nested<const CircleArc> arcs, const Box<Vec2> min_bounds); \
  template ExactArcGraph<PS> quantize_circle_arcs(const Nested<const CircleArc> arcs, const Quantizer<real,2> quant);

// Need to set perturbation indicies in quantization Explicit version to work: INSTANTIATE(Pb::Explicit)
INSTANTIATE(Pb::Implicit)

#undef INSTANTIATE

} // namespace geode