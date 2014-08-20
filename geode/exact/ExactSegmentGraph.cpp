#include <geode/array/sort.h>
#include <geode/exact/constructions.h>
#include <geode/exact/ExactSegmentGraph.h>
#include <geode/exact/predicates.h>
#include <geode/exact/scope.h>
#include <geode/geometry/polygon.h>
#include <geode/geometry/traverse.h>
#include <geode/mesh/ComponentData.h>
#include <geode/utility/curry.h>

namespace geode {

static inline bool is_same_point(const exact::Perturbed2 p0, const exact::Perturbed2 p1) {
  assert(p0.seed() != p1.seed() || p0.value() == p1.value());
  return p0.seed() == p1.seed();
}

namespace {
template<class T> struct RRange {
  typedef decltype(declval<T>().begin()) FwdIter;
  typedef std::reverse_iterator<FwdIter> RevIter;
  RevIter rbegin, rend;
  RRange(T& t)
   : rbegin(t.end())
   , rend(t.begin())
  { }
  RevIter begin() const { return rbegin; }
  RevIter end() const { return rend; }
};
template<class T> RRange<T> reversed_range(T& t) { return RRange<T>(t); }

template<class T, class U> static Array<T> asarray_of(const Array<U>& a) {
  static_assert(sizeof(U) == sizeof(T), "At least sizes must match");
  return Array<T>(a.size(),reinterpret_cast<T*>(a.data()),a.owner());
}

static Array<Box<exact::Vec2>> segment_boxes(RawArray<const int> next, RawArray<const exact::Vec2> X) {
  Array<Box<exact::Vec2>> boxes(X.size(),uninit);
  for (int i=0;i<X.size();i++)
    boxes[i] = bounding_box(X[i],X[next[i]]);
  return boxes;
}

// Compute all nontrivial intersections between segments
struct SegmentIntersections {
  const ExactSegmentSet& segs;
  Array<Vector<SegmentId,2>> pairs;

  SegmentIntersections(const ExactSegmentSet& _segs)
    : segs(_segs) {}

  bool cull(const int n) const { return false; }
  bool cull(const int n0, const int box1) const { return false; }
  void leaf(const int n) const { assert(segs.tree->prims(n).size()==1); }

  void leaf(const int n0, const int n1) {
    assert(segs.tree->prims(n0).size()==1 && segs.tree->prims(n1).size()==1);
    const auto s0 = SegmentId(segs.tree->prims(n0)[0]),
               s1 = SegmentId(segs.tree->prims(n1)[0]);
    if (segs.segments_intersect(s0,s1))
        pairs.append(vec(s0,s1));
  }
};

struct PairOrder {
  const ExactSegmentSet& segs;
  const Vector<exact::Perturbed2,2> segment;

  PairOrder(const ExactSegmentSet& _segs, SegmentId seg_i)
    : segs(_segs), segment(segs.src(seg_i), segs.dst(seg_i)) {}

  bool operator()(const Tuple<SegmentId,VertexId> j_and_v, const Tuple<SegmentId,VertexId> k_and_v) const {
    const auto j = j_and_v.x;
    const auto k = k_and_v.x;
    if (j==k)
      return false;
    return segment_intersections_ordered(segment.x,segment.y,
                                         segs.src(j), segs.dst(j),
                                         segs.src(k), segs.dst(k));
  }
};

struct RightwardRaycast {
  const ExactSegmentSet& segs;
  const exact::Perturbed2 start;
  Array<SegmentId> hits;

  bool cull(const int n) const {
    const auto box = segs.tree->boxes(n);
    return box.max.x<start.value().x || box.max.y<start.value().y || box.min.y>start.value().y;
  }

  void leaf(const int n) {
    assert(segs.tree->prims(n).size()==1);
    const auto s = SegmentId(segs.tree->prims(n)[0]);
    const auto src = segs.src(s);
    const auto dst = segs.dst(s);

    if(is_same_point(start, src) || is_same_point(start, dst))
      return;
    const bool src_above = upwards(start,src),
               dst_above = upwards(start,dst);
    if(src_above == dst_above || dst_above != triangle_oriented(src,dst,start))
      return; // Check that segment crosses ray and that intersection is to the right or start
    hits.append(s);
  }

  bool operator()(const SegmentId s0, const SegmentId s1) const {
    return ray_intersections_rightwards(segs.src(s0),segs.dst(s0),segs.src(s1),segs.dst(s1),start);
  }

  RightwardRaycast(const ExactSegmentSet& segs, const SegmentId src_seg)
   : segs(segs), start(segs.src(src_seg))
  {
    single_traverse(*segs.tree,*this);
    sort(hits, *this);
  }
};

} // anonymous namespace

ExactSegmentSet::ExactSegmentSet(const Nested<const exact::Vec2> polys)
 : src_pts(polys.flat)
 , next(asarray_of<const SegmentId>(closed_contours_next(polys)))
 , tree(new_<BoxTree<exact::Vec2>>(segment_boxes(asarray_of<const int>(next.flat),src_pts.flat),1))
{ }

bool ExactSegmentSet::segments_intersect(const SegmentId s0, const SegmentId s1) const {
  const auto d0 = next[s0], d1 = next[s1];
  if((s0==s1) || (s0==d1) || (d0==s1) || (d0==d1))
    return false; // Ignore intersections at endpoint
  return geode::segments_intersect(src(s0),src(d0),src(s1),src(d1));
}

exact::Vec2 ExactSegmentSet::approx_intersection(const SegmentId s0, const SegmentId s1) const {
  return segment_segment_intersection(src(s0),dst(s0),src(s1),dst(s1));
}

bool ExactSegmentSet::directions_oriented(const SegmentId s0, const SegmentId s1) const {
  return segment_directions_oriented(src(s0),dst(s0),src(s1),dst(s1));
}

uint8_t ExactSegmentSet::quadrant(const SegmentId s) const {
  const auto s_src = src(s), s_dst = dst(s);
  const bool d = upwards(s_dst, s_src);
  const bool l = rightwards(s_dst, s_src);
  return (d << 1) | (l ^ d);
}

Array<Vector<SegmentId, 2>> ExactSegmentSet::intersection_pairs() const {
  SegmentIntersections pairs(*this);
  double_traverse(*tree,pairs);
  return pairs.pairs;
}

static VertexId src_id(const SegmentId s) { return VertexId(s.idx()); }

static NestedField<Tuple<SegmentId,VertexId>, SegmentId> setup_segment_verts(const ExactSegmentSet& segs, Field<ExactSegmentGraph::VertexInfo, VertexId>& vertices) {
  IntervalScope scope;
  const auto intersections = segs.intersection_pairs();

  const int n_verts = segs.size() + intersections.size();

  vertices.preallocate(n_verts);

  // Create info for input points
  vertices.flat.extend_assuming_enough_space(segs.size(), uninit); // Update size for segment verts so we can initalize them out of order
  for(const SegmentId prev_seg : id_range<SegmentId>(segs.size())) { 
    const SegmentId curr_seg = segs.next[prev_seg];
    vertices[src_id(curr_seg)] = {vec(prev_seg, curr_seg), segs.src_pts[curr_seg]};
  }

  auto counts = Field<int, SegmentId>(segs.size());
  for(const auto i : intersections) {
    ++counts[i.x];
    ++counts[i.y];
  }

  auto result = NestedField<Tuple<SegmentId,VertexId>, SegmentId>(counts, uninit);
  for(const auto p : intersections) {
    const auto new_vid = vertices.append_assuming_enough_space({vec(p.x,p.y),segs.approx_intersection(p.x,p.y)}); // Create info for vertex
    result(p.x,--counts[p.x]) = tuple(p.y, new_vid); 
    result(p.y,--counts[p.y]) = tuple(p.x, new_vid);
  }
  assert(counts.flat.contains_only(0));

  // Sort each set of intersections along their segment
  for(const SegmentId s : result.id_range()) {
    auto other = result[s];
    sort(other, PairOrder(segs,s));
  }

  return result;
}

EdgeId ExactSegmentGraph::first_edge(const SegmentId s) const {
  // Since each segment's edges are added in order they will be in a contigious range 
  // Since each segment gets one edge + one edge per intersection we can use segment_verts to find start of that range
  const auto result = EdgeId(s.idx() + segment_verts.front_offset(s));
  assert(topology->src(result) == src_id(s));
  assert(edges[result].segment == s);
  return result;
}

  
void ExactSegmentGraph::add_edge(const VertexId src, const VertexId dst, const SegmentId segment) {
  assert(topology->n_edges() == edges.size());
  topology->unsafe_add_edge(src, dst);
  edges.append({segment});
  assert(topology->n_edges() == edges.size());
}

ExactSegmentGraph::ExactSegmentGraph(const Nested<const exact::Vec2> polys)
 : segs(polys)
 , vertices()
 , segment_verts(setup_segment_verts(segs, vertices))
 , topology(new_<HalfedgeGraph>())
{
  IntervalScope scope;
  const int n_segs = segs.size();
  const int n_verts = vertices.size();
  const int n_edges = n_segs + segment_verts.total_size(); // Each segment gets and edge, and each segment_vert splits that edge into one more edge
  topology->add_vertices(n_verts);
  edges.preallocate(n_edges);
  // We walk input segments and construct graph edges for them
  for(const SegmentId s : id_range<SegmentId>(n_segs)) {
    auto prev = src_id(s); // Walk from start of a segment
    for(const auto o : segment_verts[s]) {
      add_edge(prev, o.y, s); // Add edge at intermediate intersections
      prev = o.y;
    }
    add_edge(prev, dst_id(s), s); // Continue to end of segment
  }
  // Make sure we at least ended up with the right number of things
  assert(edges.size() == n_edges);
  assert(topology->n_edges() == n_edges);

  // Input vertices are degree 2 so whatever linkage they have will work
  // Intersections vertices need to be linked in ccw order
  for(const VertexId vid : id_range<VertexId>(n_verts - n_segs)+n_segs) {
    assert(topology->degree(vid) == 4); // All new verts should be degree 4
    const auto& v_info = vertices[vid];
    const SegmentId seg0 = v_info.segs[!segs.directions_oriented(v_info.segs.x,v_info.segs.y)];

    Vector<HalfedgeId,4> sorted_he; // CCW order should be fwd on seg0, fwd on !seg0, rev on seg0, rev on !seg0
    for(const HalfedgeId he : topology->outgoing(vid)) {
      const auto s = edges[HalfedgeGraph::edge(he)].segment;
      sorted_he[HalfedgeGraph::is_forward(he)<<1 | (s == seg0)] = he;
    }
    for(const int i : range(4)) { // Link edges in a loop
      assert(sorted_he[i].valid());
      topology->unsafe_link(HalfedgeGraph::reverse(sorted_he[i]), sorted_he[(i + 1) & 3]);
    }
  }

  // At this point we should have correctly oriented halfedges for the entire graph
  topology->initialize_borders();
  auto cd = ComponentData(*topology);
  FaceId infinity_face;

  // For each component we perform a raycast if needed to check if it is inside some other component
  for(const ComponentId seed_c : cd.components()) {
    if(cd[seed_c].exterior_face.valid())
      continue; // If we found the component exterior in a previous test or we don't need to keep looking
    const SegmentId seed_seg = edges[HalfedgeGraph::edge(topology->halfedge(cd.border(seed_c)))].segment; // Grab some segment on component
    assert(cd.component(topology->border(topology->halfedge(src_id(seed_seg)))) == seed_c); // Check that segment belongs to component
    initialize_path_faces(path_from_infinity(seed_seg), infinity_face, *topology, cd);
    assert(cd[seed_c].exterior_face.valid()); // Make sure that exterior face got set
  }

  assert(!infinity_face.valid() || infinity_face == boundary_face());
  topology->initialize_remaining_faces();
}

HalfedgeId ExactSegmentGraph::right_face_halfedge(const EdgeId eid) const {
  const SegmentId s = edges[eid].segment;
  const bool seg_up = upwards(segs.src(s), segs.dst(s));
  return HalfedgeGraph::halfedge(eid, seg_up);
}

Array<HalfedgeId> ExactSegmentGraph::path_from_infinity(const SegmentId seed_segment) const {
  const auto seed_vert = src_id(seed_segment);
  const Array<SegmentId> ray_segments_hits = RightwardRaycast(segs, seed_segment).hits;
  const auto ray_start = segs.src(seed_segment);

  Array<HalfedgeId> result;
  result.preallocate(ray_segments_hits.size() + 1);

  for(const auto seg_i : reversed_range(ray_segments_hits)) { // Iterate backwards to get intersections ordered left to right from infinity back to start
    const EdgeId hit_edge = find_ray_hit(ray_start, seg_i); // Find specific edge from segment that was hit
    result.append_assuming_enough_space(right_face_halfedge(hit_edge));
  }

  // We will search for first edge ccw from +x axis at seed_vert
  HalfedgeId min_he;
  SegmentId min_s;
  bool min_fwd;
  uint8_t min_q = -1;
  for(const HalfedgeId he : topology->outgoing(seed_vert)) {
    const SegmentId s = edges[HalfedgeGraph::edge(he)].segment;
    const bool fwd = HalfedgeGraph::is_forward(he);
    const auto q = fwd ? segs.quadrant(s) : ((segs.quadrant(s) + 2)&3);
    if(!min_he.valid()
      || q < min_q
      || (q == min_q && segs.directions_oriented(s, min_s) ^ min_fwd ^ fwd)) {
      min_he = he;
      min_s = s;
      min_fwd = fwd;
      min_q = q;
    }
  }
  assert(min_he.valid());
  // min_he is on the correct edge, but the incoming halfedges will be first so we reverse it
  result.append_assuming_enough_space(HalfedgeGraph::reverse(min_he));
  return result;
}

static bool ray_below_hit(const ExactSegmentSet& segs, const exact::Perturbed2 seg_src, const exact::Perturbed2 seg_dst, const exact::Perturbed2 ray_start, const Tuple<SegmentId, VertexId>& hit) {
  return segment_intersection_above_point(seg_src, seg_dst, segs.src(hit.x), segs.dst(hit.x), ray_start);
}
static bool ray_above_hit(const ExactSegmentSet& segs, const exact::Perturbed2 seg_src, const exact::Perturbed2 seg_dst, const exact::Perturbed2 ray_start, const Tuple<SegmentId, VertexId>& hit) {
  return !segment_intersection_above_point(seg_src, seg_dst, segs.src(hit.x), segs.dst(hit.x), ray_start);
}

bool ExactSegmentGraph::vertex_pt_upwards(const VertexId vid, const exact::Perturbed2 pt) const {
  const bool is_input_point = (vid.idx() < segs.size());
  if(is_input_point) {
    return upwards(segs.src(SegmentId(vid.idx())), pt);
  }
  else {
    const auto& v = vertices[vid];
    return !segment_intersection_above_point(segs.src(v.segs[0]),segs.dst(v.segs[0]),
                                             segs.src(v.segs[1]),segs.dst(v.segs[1]), pt);
  }
}

EdgeId ExactSegmentGraph::find_ray_hit(const exact::Perturbed2 ray_start, const SegmentId hit_segment) const {
  const auto o = segment_verts[hit_segment];
  EdgeId hit_edge;

  const auto first_on_seg = first_edge(hit_segment);

  if(!o.empty()) {
    const auto hit_src = segs.src(hit_segment);
    const auto hit_dst = segs.dst(hit_segment);
    const bool seg_up = upwards(hit_src, hit_dst);

    const auto ray_before_hit = curry(seg_up ? ray_below_hit
                                             : ray_above_hit,
                                      segs, hit_src, hit_dst);
    const auto dst_iter = std::upper_bound(o.begin(), o.end(), ray_start, ray_before_hit); // This is the endpoint right after ray intersection
    const int delta = int(dst_iter - o.begin()); // first_on_seg will end at o.begin(), each point after goes to the next edge
    hit_edge = EdgeId(first_on_seg.idx() + delta);
    assert(dst_iter != o.end() || topology->dst(hit_edge) == dst_id(hit_segment));
    assert(topology->dst(hit_edge) == (dst_iter != o.end() ? dst_iter->y : dst_id(hit_segment))); 
  }
  else {
    hit_edge = first_on_seg;
  }

  assert(topology->valid(hit_edge)); // Should be a valid edge
  assert(edges[hit_edge].segment == hit_segment); // Edge needs to belong to segment
  assert(vertex_pt_upwards(topology->src(hit_edge), ray_start) != vertex_pt_upwards(topology->dst(hit_edge), ray_start)); // Edge needs to cross y value of ray_start
  return hit_edge;
}

FaceId ExactSegmentGraph::boundary_face() const {
  return FaceId(0); // Boundary face is always the first created face
}


} // namespace geode
