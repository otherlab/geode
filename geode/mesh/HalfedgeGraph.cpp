#include <geode/mesh/HalfedgeGraph.h>
#include <geode/python/Class.h>
#include <geode/structure/Tuple.h>
#include <geode/array/RawField.h>
#include <geode/array/Nested.h>

namespace geode {

GEODE_DEFINE_TYPE(HalfedgeGraph)

HalfedgeGraph::HalfedgeGraph() {}

HalfedgeGraph::HalfedgeGraph(const HalfedgeGraph& g)
  : halfedges_(g.halfedges_.copy())
  , vertex_to_edge_(g.vertex_to_edge_.copy())
{}

HalfedgeGraph::~HalfedgeGraph() {}

GEODE_CORE_EXPORT Ref<HalfedgeGraph> HalfedgeGraph::copy() const { return new_<HalfedgeGraph>(*this); }

bool HalfedgeGraph::valid(VertexId v)   const { 
  return vertex_to_edge_.valid(v);
}
bool HalfedgeGraph::valid(HalfedgeId e) const { 
  return halfedges_.valid(e);
}
bool HalfedgeGraph::valid(EdgeId e)     const { 
  return e.valid() && halfedges_.valid(halfedge(e,false));
}
bool HalfedgeGraph::valid(BorderId b)   const {
  assert(has_all_border_data());
  return borders_.valid(b);
}
bool HalfedgeGraph::valid(FaceId f)     const {
  return faces_.valid(f);
}

VertexId HalfedgeGraph::add_vertex() {
  return vertex_to_edge_.append(HalfedgeId());
}

void HalfedgeGraph::add_vertices(int n) {
  assert(n >= 0);
  vertex_to_edge_.flat.resize(vertex_to_edge_.size() + n);
  assert(n == 0 || vertex_to_edge_.flat.back().valid() == false); // Check that new elements were correctly initialized
}

Array<VertexId> HalfedgeGraph::one_ring(VertexId v) const {
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

  assert(vertex_to_edge_.valid(use_vertex) && !vertex_to_edge_[use_vertex].valid()); // Must be valid vertex with no edges
  vertex_to_edge_[use_vertex] = v_to_dst;
  
  GEODE_ASSERT(v_to_dst == halfedges_.append(HalfedgeInfo({src_to_v, next(src_to_dst), use_vertex})));
  GEODE_ASSERT(dst_to_v == halfedges_.append(HalfedgeInfo({prev(dst_to_src), v_to_src, orig_dst})));
  
  halfedges_[next(v_to_dst)].prev = v_to_dst;
  halfedges_[prev(dst_to_v)].next = dst_to_v;

  halfedges_[src_to_v].next = v_to_dst;
  halfedges_[v_to_src].prev = dst_to_v;
  halfedges_[v_to_src].src = use_vertex;

  // Check and fix edge lookup for orig_dst
  HalfedgeId& dst_first_edge = vertex_to_edge_[orig_dst];
  assert(dst_first_edge.valid());
  if(dst_first_edge == dst_to_src)
    dst_first_edge = dst_to_v;

  //assert(links_consistent());
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
  HalfedgeId& dst0_edge = vertex_to_edge_[dst0];
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



EdgeId HalfedgeGraph::unsafe_add_edge(VertexId src, VertexId dst) {
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

  if(vertex_to_edge_[src].valid()) {
    const HalfedgeId src_to_sn = vertex_to_edge_[src];
    const HalfedgeId sp_to_src = prev(src_to_sn);
    // ...sp_to_src <--> src_to_sn... becomes: ...sp_to_src <--> src_to_dst...dst_to_src <--> src_to_sn...
    unsafe_link(sp_to_src, src_to_dst);
    unsafe_link(dst_to_src, src_to_sn);
  }
  else { // If src_to_sn isn't valid, src is an isolated vertex
    vertex_to_edge_[src] = src_to_dst; // Use added edge for vertex edge
    unsafe_link(dst_to_src, src_to_dst); // Connect ...dst_to_src <--> src_to_dst... 
  }

  if(vertex_to_edge_[dst].valid()) {
    const HalfedgeId dst_to_dn = vertex_to_edge_[dst];
    const HalfedgeId dp_to_dst = prev(dst_to_dn);
    // ...dp_to_dst <--> dst_to_dn... becomes: ...dp_to_dst <--> dst_to_src...src_to_dst <--> dst_to_dn...
    unsafe_link(dp_to_dst, dst_to_src);
    unsafe_link(src_to_dst, dst_to_dn);
  }
  else { // If dst_to_dn isn't valid, dst is an isolated vertex
    vertex_to_edge_[dst] = dst_to_src; // Use added edge for dst_to_dn
    unsafe_link(src_to_dst, dst_to_src); // Connect ...src_to_dst <--> dst_to_src...
  }

  return edge(src_to_dst);
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

Field<int, FaceId> compute_winding_numbers(const HalfedgeGraph& g, const FaceId boundary_face, const RawField<const int, EdgeId> edge_weights) {
  assert(g.has_all_border_data());
  const int unset_winding_number = std::numeric_limits<int>::max();
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
          opp_n = n - edge_weights[g.edge(eid)] * (g.is_forward(eid) ? 1 : -1);
          queue.append(opp_face);
        }
        else {
          // If we have set the winding number, check that value is consistant
          assert(opp_n == (n - edge_weights[g.edge(eid)] * (g.is_forward(eid) ? 1 : -1)));
        }
      }
    }
  }
  assert(!winding_numbers.flat.contains(unset_winding_number));
  return winding_numbers;
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

