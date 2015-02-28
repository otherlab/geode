#include <geode/mesh/HalfedgeGraph.h>
#include <geode/python/Class.h>
#include <geode/structure/Tuple.h>
#include <geode/structure/UnionFind.h>
#include <geode/array/RawField.h>
#include <geode/array/Nested.h>

namespace geode {

GEODE_DEFINE_TYPE(HalfedgeGraph)

HalfedgeGraph::HalfedgeGraph() {}

HalfedgeGraph::HalfedgeGraph(const HalfedgeGraph& g)
  : halfedges_(g.halfedges_.copy())
  , vertices_(g.vertices_.copy())
  , borders_(g.borders_.copy())
  , faces_(g.faces_.copy())
{}

HalfedgeGraph::~HalfedgeGraph() {}

GEODE_CORE_EXPORT Ref<HalfedgeGraph> HalfedgeGraph::copy() const { return new_<HalfedgeGraph>(*this); }

bool HalfedgeGraph::valid(const VertexId v)   const {
  return vertices_.valid(v);
}
bool HalfedgeGraph::valid(const HalfedgeId e) const {
  return halfedges_.valid(e);
}
bool HalfedgeGraph::valid(const EdgeId e)     const {
  return e.valid() && halfedges_.valid(halfedge(e,false));
}
bool HalfedgeGraph::valid(const BorderId b)   const {
  assert(has_all_border_data());
  return borders_.valid(b);
}
bool HalfedgeGraph::valid(const FaceId f)     const {
  return faces_.valid(f);
}

VertexId HalfedgeGraph::add_vertex() {
  return vertices_.append(VertexInfo({HalfedgeId()}));
}

void HalfedgeGraph::add_vertices(const int n) {
  assert(n >= 0);
  vertices_.flat.resize(vertices_.size() + n);
}

int HalfedgeGraph::degree(const VertexId v) const {
  int result = 0;
  for(GEODE_UNUSED const HalfedgeId e : outgoing(v)) {
    assert(src(e) == v);
    ++result;
  }
  return result;
}

Array<VertexId> HalfedgeGraph::one_ring(const VertexId v) const {
  Array<VertexId> result;
  for(const HalfedgeId e : outgoing(v)) {
    assert(src(e) == v);
    result.append(dst(e));
  }
  return result;
}

GEODE_CORE_EXPORT EdgeId HalfedgeGraph::split_edge(const EdgeId e, const VertexId use_vertex) {
  assert(safe_to_modify_edges());

  const int base = halfedges_.size();
  assert(base % 2 == 0); // Should have even number of halfedges
  const HalfedgeId src_to_dst = halfedge(e, false);
  const HalfedgeId dst_to_src = halfedge(e, true);
  const HalfedgeId v_to_dst = HalfedgeId(base + 0);
  const HalfedgeId dst_to_v = HalfedgeId(base + 1);
  const HalfedgeId src_to_v = src_to_dst; // Origional edge will now go to v
  const HalfedgeId v_to_src = dst_to_src; // And it's reverse will go back from v

  const VertexId orig_dst = dst(e);

  assert(valid(use_vertex)); // Must be valid vertex

  const bool v_isolated = isolated(use_vertex);

  HalfedgeInfo v_to_dst_info;
  v_to_dst_info.prev = v_isolated ? src_to_v : prev(halfedge(use_vertex));
  v_to_dst_info.next = next(src_to_dst);
  v_to_dst_info.src = use_vertex;

  HalfedgeInfo dst_to_v_info;
  dst_to_v_info.prev = prev(dst_to_src);
  dst_to_v_info.next = v_to_src;
  dst_to_v_info.src = orig_dst;

  if(v_isolated) {
    *halfedge_ptr(use_vertex) = v_to_dst;
  }

  GEODE_ASSERT(v_to_dst == halfedges_.append(v_to_dst_info));
  GEODE_ASSERT(dst_to_v == halfedges_.append(dst_to_v_info));

  // Connect links back to added edge
  halfedges_[v_to_dst_info.next].prev = v_to_dst;
  halfedges_[v_to_dst_info.prev].next = v_to_dst;
  halfedges_[dst_to_v_info.next].prev = dst_to_v;
  halfedges_[dst_to_v_info.prev].next = dst_to_v;
  halfedges_[v_to_src].src = use_vertex;

  const HalfedgeId v_out = halfedge(use_vertex);
  unsafe_link(src_to_v, v_out);

  // Check and fix edge lookup for orig_dst
  HalfedgeId& dst_first_edge = *halfedge_ptr(orig_dst);
  assert(dst_first_edge.valid());
  if(dst_first_edge == dst_to_src)
    dst_first_edge = dst_to_v;

  return edge(v_to_dst);
}

// Given e0 from src0 to dst0 and split from srcs to dsts (with prev(split) == sp and next(reverse(split)) == sn)
// Make e0 go from src0 to srcs, and add e0b from srcs to dsts while connecting halfedges as follows:
//   prev(e0) == sp, next(e0) == split
//   prev(e0b) == reverse(split), next(e0b) == sn
GEODE_CORE_EXPORT EdgeId HalfedgeGraph::split_edge_across(const EdgeId e0, const HalfedgeId split) {
  assert(safe_to_modify_edges());
  assert(dst(e0) != src(split));
  const VertexId srcs = src(split);
  const VertexId dst0 = dst(e0);
  const HalfedgeId sp_to_srcs = prev(split);
  const HalfedgeId srcs_to_sn = next(reverse(split));
  const HalfedgeId srcs_to_dsts = split;
  const HalfedgeId dsts_to_srcs = reverse(split);

  const HalfedgeId src0_to_dst0 = halfedge(e0, false);
  const HalfedgeId dst0_to_src0 = halfedge(e0, true);

  const HalfedgeId src0_to_srcs = src0_to_dst0;
  const HalfedgeId srcs_to_src0 = dst0_to_src0;

  const int base = halfedges_.size();
  assert(base % 2 == 0); // Should have even number of halfedges
  const HalfedgeId srcs_to_dst0 = HalfedgeId(base + 0);
  const HalfedgeId dst0_to_srcs = HalfedgeId(base + 1);

  // Fix edge lookup for dst0
  HalfedgeId& dst0_edge = vertices_[dst0].halfedge;
  if(dst0_edge == dst0_to_src0)
    dst0_edge = dst0_to_srcs;

  // Add new halfedges
  GEODE_ASSERT(srcs_to_dst0 == halfedges_.append(HalfedgeInfo({dsts_to_srcs, next(src0_to_dst0), srcs})));
  GEODE_ASSERT(dst0_to_srcs == halfedges_.append(HalfedgeInfo({prev(dst0_to_src0), srcs_to_sn, dst0 })));

  // Update neighbors of new halfedges
  halfedges_[prev(srcs_to_dst0)].next = srcs_to_dst0;
  halfedges_[next(srcs_to_dst0)].prev = srcs_to_dst0;
  halfedges_[prev(dst0_to_srcs)].next = dst0_to_srcs;
  halfedges_[next(dst0_to_srcs)].prev = dst0_to_srcs;

  // Fix dangling edges not adjacent to new halfedges
  halfedges_[srcs_to_src0].src = srcs;
  unsafe_link(src0_to_srcs, srcs_to_dsts);
  unsafe_link(sp_to_srcs, srcs_to_src0);

  //assert(links_consistent());
  return edge(srcs_to_dst0);
}

Vector<HalfedgeId*, 4> HalfedgeGraph::halfedge_refs(const HalfedgeId h) {
  const VertexId src_h = src(h);
  const BorderId border_h = border(h);
  return vec(next_ptr(prev(h)),
             prev_ptr(next(h)),
             src_h.valid() && halfedge(src_h) == h ? halfedge_ptr(src_h) : nullptr,
             border_h.valid() && halfedge(border_h) == h ? halfedge_ptr(border_h) : nullptr);
}

bool HalfedgeGraph::check_invariants() const {
  for(const HalfedgeId he : halfedges()) {
    if(prev(next(he)) != he) {
      return false;
    }
    if(src(next(he)) != dst(he)) {
      return false;
    }
    if(!halfedge(src(he)).valid()) {
      return false;
    }
  }
  for(const VertexId v : vertices()) {
    const HalfedgeId he = halfedge(v);
    if(he.valid() && src(he) != v) {
      return false;
    }
  }
  if(!borders_.empty() || !faces_.empty()) {
    for(const HalfedgeId he : halfedges()) {
      if(!border(he).valid()) {
        return false;
      }
      if(border(he) != border(next(he))) {
        return false;
      }
    }

    for(const BorderId b : borders()) {
      if(prev(next(b)) != b) {
        return false;
      }
      if(border(halfedge(b)) != b) {
        return false;
      }
      if(face(b) != face(next(b))) {
        return false;
      }
    }

    for(const FaceId f : faces()) {
      if(face(border(f)) != f) {
        return false;
      }
    }
  }
  return true;
}

void HalfedgeGraph::swap_ids(const EdgeId e0, const EdgeId e1) {
  // We need to update both halfedges for each edge
  const auto h = vec(halfedge(e0,0),halfedge(e1,0),halfedge(e0,1),halfedge(e1,1));

  // First collect all of the references beforehand so that we can't break any along the way
  const auto refs = vec(halfedge_refs(h[0]),halfedge_refs(h[1]),halfedge_refs(h[2]),halfedge_refs(h[3]));

  for(const int i_old : range(4)) {
    const auto i_new = i_old^1;
    for(const auto& r : refs[i_old]) {
      if(!r) continue; // Skip references that weren't filled
      assert(*r == h[i_old]); // Check that we have a references that used to point to old id
       *r = h[i_new]; // Make each old reference point to new id
    }
  }

  const auto edge_view = this->edge_view(); // Cast halfedge data to Vector<HalfedgeData,2>
  std::swap(edge_view[e0], edge_view[e1]);
}

void HalfedgeGraph::unsafe_flip_edge(const EdgeId e) {
  const auto h = vec(halfedge(e,0),halfedge(e,1));
  const auto refs = vec(halfedge_refs(h[0]),halfedge_refs(h[1]));
  for(int i_old : range(2)) {
    const auto i_new = i_old^1;
    for(const auto& r : refs[i_old]) {
      if(!r) continue;
      assert(*r == h[i_old]);
      *r = h[i_new];
    }
  }
  std::swap(halfedges_[h[0]], halfedges_[h[1]]);
}

void HalfedgeGraph::erase_last_edge() {
  assert(safe_to_modify_edges());

  const auto e = edges().back();

  // Collect info on edge
  const auto h = halfedges(e);
  const auto p = vec(prev(h[0]),prev(h[1]));
  const auto n = vec(next(h[0]),next(h[1]));
  const auto v_src = vec(src(h[0]),src(h[1]));
  const auto v_src_ptr = vec(
    (                        v_src[0].valid() && halfedge(v_src[0]) == h[0]) ? halfedge_ptr(v_src[0]) : nullptr,
    (v_src[0] != v_src[1] && v_src[1].valid() && halfedge(v_src[1]) == h[1]) ? halfedge_ptr(v_src[1]) : nullptr);

  for(const int i : range(2)) {
    if(v_src_ptr[i]) {
      const bool is_degree_one = (p[i] == h[i^1]); // Fast check to see if vertex will be isolated
      assert((degree(v_src[i]) == 1) == is_degree_one); // Verify that fast check is correct
      if(is_degree_one) {
        *(v_src_ptr[i]) = HalfedgeId();
      }
      else {
        const HalfedgeId next_outgoing = reverse(p[i]);
        assert(src(next_outgoing) == v_src[i]);
        *(v_src_ptr[i]) = next_outgoing;
      }
    }
  }

  // Update halfedge links
  unsafe_link(p[0],n[0]);
  unsafe_link(p[1],n[1]);

  halfedges_.flat.pop();
  halfedges_.flat.pop();
}

void HalfedgeGraph::swap_ids(const VertexId v0, const VertexId v1) {
  const auto v0_out = outgoing(v0);
  const auto v1_out = outgoing(v1);
  for(const HalfedgeId h : v0_out) {
    auto& src = *(src_ptr(h));
    assert(src == v0);
    src = v1;
  }
  for(const HalfedgeId h : v1_out) {
    auto& src = *(src_ptr(h));
    assert(src == v1);
    src = v0;
  }
  std::swap(vertices_[v0], vertices_[v1]);
}

void HalfedgeGraph::erase_last_vertex() {
  assert(isolated(vertices().back()));
  vertices_.flat.pop();
}

EdgeId HalfedgeGraph::unsafe_add_edge(const VertexId src, const VertexId dst) {
  assert(safe_to_modify_edges());

  const HalfedgeId src_to_dst = halfedges_.append(HalfedgeInfo({HalfedgeId(), HalfedgeId(), src}));
  const HalfedgeId dst_to_src = halfedges_.append(HalfedgeInfo({HalfedgeId(), HalfedgeId(), dst}));
  assert(edge(src_to_dst) == edge(dst_to_src));

  // For each vertex we look at a pair of linked halfedges through it:
  //   For src these are: ...sp_to_src <- src -> src_to_sn... (eventually back to sp_to_src)
  //   For dst these are: ...dp_to_dst <- dst -> dst_to_dn... (eventually back to dp_to_dst)

  // New links will splice chain of halfedge:
  //   ...sp_to_src <- src -> src_to_dst <- dst -> dst_to_dn... (eventually to dp_to_dst)
  //   ...dp_to_dst <- dst -> dst_to_src <- src -> src_to_sn... (eventually back to sp_to_src)

  if(vertices_[src].halfedge.valid()) {
    const HalfedgeId src_to_sn = vertices_[src].halfedge;
    const HalfedgeId sp_to_src = prev(src_to_sn);
    // ...sp_to_src <--> src_to_sn... becomes: ...sp_to_src <--> src_to_dst...dst_to_src <--> src_to_sn...
    unsafe_link(sp_to_src, src_to_dst);
    unsafe_link(dst_to_src, src_to_sn);
  }
  else { // If src_to_sn isn't valid, src is an isolated vertex
    vertices_[src].halfedge = src_to_dst; // Use added edge for vertex edge
    unsafe_link(dst_to_src, src_to_dst); // Connect ...dst_to_src <--> src_to_dst...
  }

  if(vertices_[dst].halfedge.valid()) {
    const HalfedgeId dst_to_dn = vertices_[dst].halfedge;
    const HalfedgeId dp_to_dst = prev(dst_to_dn);
    // ...dp_to_dst <--> dst_to_dn... becomes: ...dp_to_dst <--> dst_to_src...src_to_dst <--> dst_to_dn...
    unsafe_link(dp_to_dst, dst_to_src);
    unsafe_link(src_to_dst, dst_to_dn);
  }
  else { // If dst_to_dn isn't valid, dst is an isolated vertex
    vertices_[dst].halfedge = dst_to_src; // Use added edge for dst_to_dn
    unsafe_link(src_to_dst, dst_to_src); // Connect ...src_to_dst <--> dst_to_src...
  }

  return edge(src_to_dst);
}

void HalfedgeGraph::unsafe_disconnect_src(const HalfedgeId he) {
  assert(safe_to_modify_edges());

  const bool is_degree_one = (reverse(he) == prev(he));
  assert(is_degree_one == (degree(src(he)) == 1));

  // This is the vertex->halfedge link we need to update
  auto& src_vertex_he = vertices_[src(he)].halfedge;

  if(is_degree_one) {
    assert(reverse(he) == prev(he)); // Shouldn't need to update halfedge linkage since this end of the edge is already isolated
    assert(src_vertex_he == he); // Since this is the only edge it should be the marked one
    src_vertex_he = HalfedgeId();
    halfedges_[he].src = VertexId();
  }
  else {
    // Check if vertex->halfedge points to this halfedge
    if(src_vertex_he == he) {
      // If so we need to point it to another
      src_vertex_he = right(he);
      assert(src(src_vertex_he) == src(he));
      assert(src_vertex_he != he);
    }

    unsafe_link(prev(he), right(he)); // Connect halfedges to and from disconnected he to each other

    // User will need to reconnect edge manually before things are back to happy so this shouldn't be neccessary
    // We fix dangling links anyway just to be safe.
    unsafe_link(reverse(he), he);
    halfedges_[he].src = VertexId();
  }
}

void HalfedgeGraph::unsafe_reconnect_src(const HalfedgeId he, const VertexId new_src) {
  assert(safe_to_modify_edges());
  // This should only be called on dangling edges generated by unsafe_disconnect_src
  assert(!src(he).valid()); // For now those should have src to to an invalid id
  assert(prev(he) == reverse(he)); // This should also connect back to themselves

  const bool was_isolated = isolated(new_src); // Check this before we start to modify things

  // Get vertex->halfedge link for new_src
  auto& src_halfedge = vertices_[new_src].halfedge;

  halfedges_[he].src = new_src; // Connect the halfedge to its new src

  if(was_isolated) {
    src_halfedge = he; // Previously isolated vertex must link to he
    unsafe_link(reverse(he), he); // Vertex is now degree one so edge links back to itself at src
  }
  else {
    // Get neighbors before we start mucking with them
    const HalfedgeId old_src_out = src_halfedge;
    const HalfedgeId old_src_in = prev(src_halfedge);
    // Insert he into existing one-ring
    unsafe_link(reverse(he), old_src_out);
    unsafe_link(old_src_in, he);
  }
}

void HalfedgeGraph::initialize_borders() {
  assert(borders_.empty());

  for(const HalfedgeId seed_hid : halfedges()) {
    // Skip over halfedges until we find one that doesn't have a border set
    if(halfedges_[seed_hid].border.valid())
      continue;
    // Create a new border for this halfedge
    const BorderId bid = borders_.append(BorderInfo());
    BorderInfo& b = borders_[bid];
    b.halfedge = seed_hid;
    b.prev = bid;
    b.next = bid;

    // Walk along adjacent halfedges to mark them as belonging to the new border
    for(const HalfedgeId hid : border_edges(seed_hid)) {
      assert(!halfedges_[hid].border.valid()); // We should traverse each edge exactly once
      halfedges_[hid].border = bid;
    }
  }

  assert(has_all_border_data());
}

FaceId HalfedgeGraph::new_face_for_border(const BorderId b) {
  assert(has_all_border_data());
  assert(!borders_[b].face.valid());
  const FaceId new_face = faces_.append(FaceInfo({b}));
  borders_[b].face = new_face;
  return new_face;
}

void HalfedgeGraph::add_to_face(const FaceId f, const BorderId child) {
  assert(f.valid());
  assert(!face(child).valid());

  borders_[child].face = f;

  const BorderId parent = border(f);
  assert(parent.valid());

  BorderId& next_at_p = borders_[parent ].next;
  BorderId& prev_to_p = borders_[next_at_p].prev;

  BorderId& prev_at_c = borders_[child    ].prev;
  BorderId& next_to_c = borders_[prev_at_c].next;

  // All of these should have valid ids
  assert(valid(next_at_p) && valid(prev_to_p) && valid(prev_at_c) && valid(next_to_c));

  // Swap next links between 'a' and whatever used to point forward to 'b'
  swap(next_at_p, next_to_c);
  // Swap prev links between 'b' and whatever used to point back to 'a'
  swap(prev_at_c, prev_to_p);

  // Make sure that a.next == b and b.prev == a
  assert(borders_[parent].next == child);
  assert(borders_[child].prev == parent);
  assert(face(child) == f);
}

void HalfedgeGraph::initialize_remaining_faces() {
  for(const BorderId bid : borders()) {
    if(!face(bid).valid()) // Find borders that didn't get assigned a face yet
      new_face_for_border(bid); // Add a new face
  }
}

std::ostream& operator<<(std::ostream& os, const HalfedgeGraph& g) {
  os << "{ HalfedgeGraph with " << g.n_vertices() << " vertices and " << g.n_edges() << " edges:\n"
     << "\tEdges:";
  for(const EdgeId eid : g.edges()) {
    os << " " <<  eid << ": " << g.vertices(eid);
  }
  os << "\n\tVertex rings:";
  for(const VertexId vid : g.vertices()) {
    os << "[ ";
    for(const HalfedgeId hid : g.outgoing(vid)) {
      os << g.edge(hid) << (g.is_forward(hid) ? "+" : "-") << g.vertices(hid) << " ";
    }
    os << "]";
  }
  if(g.n_faces() != 0) {
    os << "\n\tFaces:\n";
    for(const FaceId fid : g.faces()) {
      const HalfedgeId hid = g.halfedge(fid);
      os << fid << "[" << g.src(hid) << "-->" << g.dst(hid) << "]\n";
    }
  }
  os << "}";
  return os;
}

Field<CrossingInfo, FaceId> get_crossing_depths(const HalfedgeGraph& g, const FaceId boundary_face) {
  assert(g.has_all_border_data());
  auto info = Field<CrossingInfo, FaceId>(g.n_faces());

  typedef Tuple<FaceId, int> Update;
  Array<Update> queue;

  queue.append(Update(boundary_face, 0));
  info[boundary_face].depth = 0;

  while(!queue.empty()) {
    const Update top = queue.pop();
    const FaceId curr_face = top.x;
    const int depth = top.y;

    // Check if we already updated a face and can skip it
    if(info[curr_face].depth != depth) {
      assert(info[curr_face].depth < depth);
      continue;
    }

    // Loop over all borders of the face
    for(const BorderId bid : g.face_borders(curr_face)) {
      // And over all edges of each border
      for(const HalfedgeId eid : g.border_edges(bid)) {
        // Get the opposite face
        const HalfedgeId opp_edge = g.reverse(eid);
        const FaceId opp_face = g.face(opp_edge);
        assert(opp_face.valid());
        auto& opp_info = info[opp_face];
        if(opp_info.depth > (depth+1)) { // If we found a shorter path, propagate an update
          opp_info.depth = depth+1; // Mark smaller value so we don't try and propagate multiple updates
          opp_info.next = opp_edge;
          queue.append(Update(opp_face, opp_info.depth));
        }
      }
    }
  }
  return info;
}

bool has_manifold_edge_weights(const HalfedgeGraph& g, const RawField<const int, EdgeId> edge_weights) {
  for(const VertexId vid : g.vertices()) {
    int net = 0;
    for(const HalfedgeId hid : g.outgoing(vid)) {
      const EdgeId eid = HalfedgeGraph::edge(hid);
      if(HalfedgeGraph::is_forward(hid))
        net += edge_weights[eid];
      else
        net -= edge_weights[eid];
    }
    if(net)
      return false;
  }
  return true;
}

// compute_winding_numbers has a number of fragile and complicated internal interactions that proved difficult to debug
// compute_winding_numbers_oracle is a slower but more straightforward implementation that can be used as a reference
#define USE_WINDING_NUMBER_ORACLE 0

// We compile this function even if USE_WINDING_NUMBER_ORACLE is set to 0 to help catch refactoring bugs
GEODE_UNUSED static Field<int, FaceId> compute_winding_numbers_oracle(const HalfedgeGraph& g, const FaceId boundary_face, const RawField<const int, EdgeId> edge_weights) {
  GEODE_WARNING("Slow, redundant winding number check is enabled!");
  assert(g.has_all_border_data());
  assert(g.check_invariants());
  assert(has_manifold_edge_weights(g, edge_weights));
  GEODE_CONSTEXPR_UNLESS_MSVC int unset_winding_number = std::numeric_limits<int>::max();
  auto winding_numbers = Field<int, FaceId>(g.n_faces());
  winding_numbers.flat.fill(unset_winding_number);

  // Handle graph with no faces
  if(winding_numbers.empty())
    return winding_numbers;

  assert(g.valid(boundary_face));

  Array<FaceId> queue;
  queue.append(boundary_face);
  winding_numbers[boundary_face] = 0;
  while(!queue.empty()) {
    const FaceId curr_face = queue.pop();
    const int n = winding_numbers[curr_face];
    assert(n != unset_winding_number);
    // Loop over all borders of the face
    for(const BorderId bid : g.face_borders(curr_face)) {
      // And over all edges of each border
      for(const HalfedgeId eid : g.border_edges(bid)) {
        // Get the opposite face
        const FaceId opp_face = g.opp_face(eid);
        auto& opp_n = winding_numbers[opp_face];
        if(opp_n == unset_winding_number) {
          // If we haven't set the winding number, update it
          opp_n = n - edge_weights[HalfedgeGraph::edge(eid)] * (HalfedgeGraph::is_forward(eid) ? 1 : -1);
          queue.append(opp_face);
        }
        else {
          // If we have set the winding number, check that value is consistant
          assert(opp_n == (n - edge_weights[HalfedgeGraph::edge(eid)] * (HalfedgeGraph::is_forward(eid) ? 1 : -1)));
        }
      }
    }
  }
  assert(!winding_numbers.flat.contains(unset_winding_number));
  return winding_numbers;
}

namespace {
class WindingDepthHelper {
  static GEODE_CONSTEXPR_UNLESS_MSVC int invalid_depth = INT_MAX;
  struct DepthNode {
    int parent_ = -1;
    int depth_to_parent_;
    bool is_root() const { return parent_ < 0; }
    FaceId parent() const { assert(parent_ >= 0); return FaceId(parent_); }
    int rank() const { assert(parent_ < 0); return parent_; }
    int depth_to_parent() const { assert(!is_root()); return depth_to_parent_; }
  };
  Field<DepthNode, FaceId> nodes;
  Field<int, FaceId> depths;

#if USE_WINDING_NUMBER_ORACLE
 public:
  Field<int, FaceId> oracle;
 private:
  bool depth_to_parent_correct(const FaceId f) const {
    if(nodes[f].is_root()
     || oracle[f] + nodes[f].depth_to_parent() == oracle[nodes[f].parent()])
      return true;
    return false;
  }
  bool absolute_depth_correct(const FaceId f, const int depth) {
    return oracle[f] == depth;
  }
  int oracle_depth(const FaceId f) { return oracle[f]; }
#else
 bool depth_to_parent_correct(const FaceId a) const { return true; }
 bool absolute_depth_correct(const FaceId f, const int depth) { return true; }
 int oracle_depth(const FaceId f) { return 0; }
#endif

  void set_parent(const FaceId child, const FaceId new_parent, const int new_depth_to_parent) {
    auto& c = nodes[child];
    assert(c.is_root());
    c.parent_ = new_parent.idx();
    c.depth_to_parent_ = new_depth_to_parent;
    assert(depth_to_parent_correct(child));
  }

  void share_grandparent(const FaceId child) {
    assert(depth_to_parent_correct(child));
    auto& c = nodes[child];
    assert(!c.is_root());
    auto& p = nodes[c.parent()];
    assert(!p.is_root());
    c.parent_ = p.parent_;
    c.depth_to_parent_ += p.depth_to_parent();
    assert(depth_to_parent_correct(child));
  }

  FaceId find_root(FaceId i, int& depth_to_root) {
    depth_to_root = 0;
    for(;;) {
      if(nodes[i].is_root())
        return i;
      depth_to_root += nodes[i].depth_to_parent();
      const FaceId pi = nodes[i].parent();
      if(nodes[pi].is_root())
        return pi;
      depth_to_root += nodes[pi].depth_to_parent();
      share_grandparent(i);
      i = nodes[pi].parent();
    }
  }

  FaceId merge_roots(FaceId a, int a_to_b, FaceId b) {
    assert(depth_to_parent_correct(a));
    assert(depth_to_parent_correct(b));
    assert(nodes[a].is_root() && nodes[b].is_root());
    if(a != b) {
      if(nodes[a].rank() > nodes[b].rank()) {
        swap(a,b);
        a_to_b = -a_to_b;
      }
      else if(nodes[a].rank() == nodes[b].rank()) {
        nodes[a].parent_--;
      }
      nodes[b].parent_ = a.idx();
      nodes[b].depth_to_parent_ = -a_to_b;
      assert(depth_to_parent_correct(b));
    }
    else {
      assert(a_to_b == 0);
    }
    return a;
  }

  void seed_depth(FaceId seed, int seed_depth) {
    assert(depths.valid(seed));
    assert(depths[seed] == invalid_depth);
    do {
      assert(depths.valid(seed));
      assert(depth_to_parent_correct(seed));
      assert(absolute_depth_correct(seed, seed_depth));
      depths[seed] = seed_depth;
      if(nodes[seed].is_root())
        return;
      seed_depth += nodes[seed].depth_to_parent();
      seed = nodes[seed].parent();
      assert(depths.valid(seed));
      assert(absolute_depth_correct(seed, seed_depth));
    } while(depths[seed] == invalid_depth);
  }

 public:
  WindingDepthHelper(const int n_faces)
   : nodes(n_faces)
   , depths(n_faces, uninit)
  {
    depths.flat.fill(invalid_depth);
  }

  FaceId connect(const FaceId a, const int a_to_b, const FaceId b) {
    int a_to_root_a = 0;
    int b_to_root_b = 0;
    const FaceId root_a = find_root(a, a_to_root_a);
    const FaceId root_b = find_root(b, b_to_root_b);
    const int root_a_to_root_b = -a_to_root_a + a_to_b + b_to_root_b;
    return merge_roots(root_a, root_a_to_root_b, root_b);
  }

  Field<int, FaceId> get_depths(const FaceId seed, const int seed_d) {
    assert(depths.valid(seed));
    assert(absolute_depth_correct(seed, seed_d));
    seed_depth(seed, seed_d);
    for(const FaceId start : depths.id_range()) {
      if(depths[start] != invalid_depth)
        continue;
      FaceId i = start;
      int start_to_i = 0;
      do {
        start_to_i += nodes[i].depth_to_parent();
        i = nodes[i].parent();
        assert(absolute_depth_correct(i, oracle_depth(start) + start_to_i));
        assert(depth_to_parent_correct(i));
        assert(i.valid());
      } while(depths[i] == invalid_depth);
      const int start_depth = depths[i] - start_to_i;
      assert(depths.valid(seed));
      seed_depth(start, start_depth); // Save data back ensuring that no nodes need to be traversed multiple times
      assert(depths[i] != invalid_depth);
    }
    assert(!depths.flat.contains(invalid_depth));
    return depths;
  }
}; }

Field<int, FaceId> compute_winding_numbers(const HalfedgeGraph& g, const FaceId boundary_face, const RawField<const int, EdgeId> edge_weights) {
  assert(g.check_invariants());
  assert(has_manifold_edge_weights(g, edge_weights));
  auto helper = WindingDepthHelper(g.n_faces());
#if USE_WINDING_NUMBER_ORACLE
  helper.oracle = compute_winding_numbers_oracle(g, boundary_face, edge_weights);
#endif
  for(const EdgeId eid : g.edges()) {
    const Vector<FaceId,2> faces = g.faces(eid);
    if(faces[0] != faces[1])
      helper.connect(faces[0],-edge_weights[eid],faces[1]);
  }
  const auto result = helper.get_depths(boundary_face, 0);
#if USE_WINDING_NUMBER_ORACLE
  assert(result.flat == helper.oracle.flat);
#endif
  return result;
}


Nested<HalfedgeId> extract_region(const HalfedgeGraph& g, const RawField<const bool, FaceId> interior_faces) {
  assert(g.n_faces() == interior_faces.size());
  struct BoundaryPredicate {
    const HalfedgeGraph& g;
    const RawField<const bool, FaceId>& interior_faces;
    bool operator()(const HalfedgeId hid) const { return interior_faces[g.face(hid)] && !interior_faces[g.opp_face(hid)]; }
  };
  const auto is_boundary = BoundaryPredicate({g,interior_faces});

  Nested<HalfedgeId, false> contours;

  auto seen = Field<bool,EdgeId>(g.n_edges(),uninit);
  seen.flat.fill(true); // Assume any face without valid edges doesn't need to be processed
  for(const FaceId fid : g.faces()) { // Iterate over faces to ensure we only process edges with a valid face
    for(const BorderId bid : g.face_borders(fid)) {
      for(const HalfedgeId hid : g.border_edges(bid)) {
        if(g.is_forward(hid)) // Only check at the forward halfedge
          seen[g.edge(hid)] = (interior_faces[g.face(hid)] == interior_faces[g.opp_face(hid)]);
      }
    }
  }

  for(const EdgeId seed_eid : g.edges()) {
    if(seen[seed_eid])
      continue;
    const HalfedgeId fwd = g.halfedge(seed_eid, false);
    const HalfedgeId opp = g.halfedge(seed_eid, true);
    assert(g.valid(g.face(fwd)) && g.valid(g.face(opp))); // Should have marked any edges with invalid faces as seen
    assert(is_boundary(fwd) != is_boundary(opp));
    HalfedgeId curr = is_boundary(fwd) ? fwd : opp;
    contours.append_empty();
    do {
      assert(is_boundary(curr));
      contours.append_to_back(curr);
      seen[g.edge(curr)] = true;

      HalfedgeId next = g.next(curr);
      while(!is_boundary(next)) {
        next = g.right(next);
        assert(g.src(next) == g.dst(curr));
        assert(next != g.next(curr)); // Check that we didn't loop back to first edge
      }
      assert(g.src(next) == g.dst(curr));
      curr = next;
    } while(!seen[g.edge(curr)]);
    assert(!contours.back().empty());
    assert(g.src(contours.back().front()) == g.dst(contours.back().back())); // Check that contour connects back to itself
  }
  assert(seen.flat.contains_only(true));
  return contours.freeze();
}

} // namespace geode

