#include <geode/mesh/HalfedgeGraph.h>
#include <geode/python/Class.h>

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
  os << "}";
  return os;
}

} // namespace geode

