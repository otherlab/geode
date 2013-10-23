// A halfedge data structure representing oriented planar graphs.
// This class represents topology only, not geometry.

#include <other/core/array/Field.h>
#include <other/core/mesh/ids.h>

namespace other {

// Important invariants:
// 1. e = next(e) will eventually form a cycle
// 2. prev and next invert each other (i.e. prev(next(e)) == e)
// 3. src(next(e)) == dst(e)
// There can be self loops: src(e) == dst(e) and even next(e) == e
// Not enforced programmatically, but should be true for ArcGraphs: Given vertices v0,v1, there are at most four halfedges from v0 to v1.

class HalfedgeGraph : public Object {
public:
  OTHER_DECLARE_TYPE(OTHER_CORE_EXPORT)
  typedef Object Base;

  struct HalfedgeInfo {
    HalfedgeId prev, next;
    VertexId src;
  };
  struct OutgoingCirculator;

  Field<HalfedgeInfo,HalfedgeId> halfedges_;
  Field<HalfedgeId,VertexId> vertex_to_edge_; // src(vertex_to_edge_[v]) == v if valid

protected:
  OTHER_CORE_EXPORT HalfedgeGraph();
  OTHER_CORE_EXPORT HalfedgeGraph(const HalfedgeGraph& g);
public:
  ~HalfedgeGraph();
  // Copy the graph
  OTHER_CORE_EXPORT Ref<HalfedgeGraph> copy() const;

  // Count various features
  int n_vertices()  const { return vertex_to_edge_.size(); }
  int n_halfedges() const { return halfedges_.size(); }
  int n_edges()     const { return halfedges_.size()>>1; }

  // Walk around the halfedge structure.  These always succeed given valid ids,
  // but may return invalid ids as a result
  HalfedgeId halfedge(VertexId v)           const { assert(!valid(vertex_to_edge_[v]) || src(vertex_to_edge_[v]) == v); return vertex_to_edge_[v]; }
  HalfedgeId prev(HalfedgeId e)             const { return halfedges_[e].prev; }
  HalfedgeId next(HalfedgeId e)             const { return halfedges_[e].next; }
  HalfedgeId reverse(HalfedgeId e)          const { assert(e.valid()); return HalfedgeId(e.id^1); }
  EdgeId     edge(HalfedgeId e)             const { assert(e.valid()); return EdgeId(e.id>>1); }
  EdgeId     edge(VertexId v)               const { return edge(halfedge(v)); } // Warning, this will fail if no edge connected to vertex exists
  HalfedgeId halfedge(EdgeId e, bool which) const { assert(e.valid()); return HalfedgeId((e.id<<1)|which); }
  VertexId   src(HalfedgeId e)              const { return halfedges_[e].src; }
  VertexId   dst(HalfedgeId e)              const { return src(reverse(e)); }
  VertexId   src(EdgeId e)                  const { return src(halfedge(e,false)); }
  VertexId   dst(EdgeId e)                  const { return src(halfedge(e,true)); }
  HalfedgeId left(HalfedgeId e)             const { return reverse(prev(e)); }
  HalfedgeId right(HalfedgeId e)            const { return next(reverse(e)); }

  // Check id validity
  bool valid(VertexId v)   const;
  bool valid(HalfedgeId e) const;
  bool valid(EdgeId e)     const;

  bool is_forward(HalfedgeId e) const { return (e.id&1) == 0; }

  // Iterate over vertices, edges, or faces
  Range<IdIter<VertexId>>   vertices()  const { return Range<IdIter<VertexId>>  (VertexId(0),  VertexId(  n_vertices())); }
  Range<IdIter<HalfedgeId>> halfedges() const { return Range<IdIter<HalfedgeId>>(HalfedgeId(0),HalfedgeId(n_halfedges())); }
  Range<IdIter<EdgeId>>     edges()     const { return Range<IdIter<EdgeId>>    (EdgeId(0),    EdgeId(    n_edges())); }

  Vector<VertexId,2> vertices(HalfedgeId e) const {
    return vec(src(e),dst(e));
  }
  Vector<VertexId,2> vertices(EdgeId e) const { return vertices(halfedge(e, false)); }

  inline Range<OutgoingCirculator> outgoing(VertexId v) const;

  // Find a halfedge between two vertices, or return an invalid id if none exists.
  OTHER_CORE_EXPORT HalfedgeId halfedge(VertexId v0, VertexId v1) const;

  // Compute the edge degree of a vertex in O(degree) time.
  OTHER_CORE_EXPORT int degree(VertexId v) const;

  // All connected edges to vertex
  OTHER_CORE_EXPORT Array<VertexId> one_ring(VertexId v) const;

  // Add a new isolated vertex and return its id
  OTHER_CORE_EXPORT VertexId add_vertex();

  // Add n isolated vertices
  OTHER_CORE_EXPORT void add_vertices(int n);

  OTHER_CORE_EXPORT EdgeId add_edge(const VertexId v0, const VertexId v1);

  // Split e0 and e1 at a new point
  // src of e0 and e1 will be unchanged
  // dst of e0 and e1 will be result.x
  // result.y will go from result.x to old value of dst(e0)
  // result.z will go from result.x to old value of dst(e1)
  // OTHER_CORE_EXPORT Tuple<VertexId, EdgeId, EdgeId> add_intersection(const EdgeId e0, const EdgeId e1);

  OTHER_CORE_EXPORT EdgeId split_edge(const EdgeId e, const VertexId use_vertex); // Vertex must be valid, but with no edges
  OTHER_CORE_EXPORT EdgeId split_edge_across(const EdgeId e0, const HalfedgeId split);

  // Use only through HalfedgeMesh::outgoing()
  struct OutgoingCirculator {
    const HalfedgeGraph& mesh;
    HalfedgeId e;
    bool first;
    OutgoingCirculator(const HalfedgeGraph& mesh, HalfedgeId e, bool first) : mesh(mesh), e(e), first(first) {}
    void operator++() { e = mesh.left(e); first = false; }
    bool operator!=(OutgoingCirculator o) { return first || e!=o.e; } // For use only inside range-based for loops
    HalfedgeId operator*() const { return e; }
  };

  // The remaining functions are mainly for internal use, or for external routines that perform direct surgery
  // on the halfedge structure.  Use with caution!

  // Adds a new edge. New halfedges will be linked with arbitrarily choosen neighbor at src and dst
  EdgeId unsafe_add_edge(VertexId src, VertexId dst);

  // Link two edges together
  void unsafe_link(HalfedgeId p, HalfedgeId n) {
    halfedges_[p].next = n;
    halfedges_[n].prev = p;
    assert(dst(p) == src(n));
  }

  void unsafe_change_end(const EdgeId e, const VertexId at_vertex);

};

std::ostream& operator<<(std::ostream& os, const HalfedgeGraph& g);

inline Range<HalfedgeGraph::OutgoingCirculator> HalfedgeGraph::outgoing(VertexId v) const {
  const auto e = halfedge(v);
  const HalfedgeGraph::OutgoingCirculator c(*this,e,e.valid());
  return Range<HalfedgeGraph::OutgoingCirculator>(c,c);
}

}