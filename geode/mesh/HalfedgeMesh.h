// A halfedge data structure representing oriented triangle meshes.
// This class represents topology only, not geometry.
#pragma once

#include <geode/mesh/ids.h>
#include <geode/array/Field.h>
namespace geode {

struct OutgoingCirculator;

// Important invariants:
// 1. The topology is always locally manifold, with the proviso that a vertex may
//    have multiple disjoint boundary curves (this is common when triangles are
//    added from triangle soup.
// 2. If a vertex is a boundary vertex, its outgoing halfedge is a boundary halfedge.
// 3. If e is a halfedge, either e or reverse(e) is not a boundary.
// 4. There are no self loops: src(e) != dst(e).
// 5. Given vertices v0,v1, there is at most one halfedge from v0 to v1.

class HalfedgeMesh : public Object {
public:
  GEODE_DECLARE_TYPE(GEODE_CORE_EXPORT)
  typedef Object Base;

  // We deliberately expose the internals publically, so that users with special requirements
  // may perform surgery on the raw halfedge structure.  Normal use should go through the
  // shielded interface below.
  struct HalfedgeInfo {
    HalfedgeId prev,next;
    VertexId src;
    FaceId face;
  };
  Field<HalfedgeInfo,HalfedgeId> halfedges_;
  Field<HalfedgeId,VertexId> vertex_to_edge_; // outgoing halfedge, and boundary if possible
  Field<HalfedgeId,FaceId> face_to_edge_; // one of this face's halfedges

protected:
  GEODE_CORE_EXPORT HalfedgeMesh();
  GEODE_CORE_EXPORT HalfedgeMesh(const HalfedgeMesh& mesh);
public:
  ~HalfedgeMesh();

  // Copy the mesh
  GEODE_CORE_EXPORT Ref<HalfedgeMesh> copy() const;

  // Count various features
  int n_vertices()  const { return vertex_to_edge_.size(); }
  int n_halfedges() const { return halfedges_.size(); }
  int n_edges()     const { return halfedges_.size()>>1; }
  int n_faces()     const { return face_to_edge_.size(); }

  // Walk around the halfedge structure.  These always succeed given valid ids,
  // but may return invalid ids as a result (e.g., the face of a boundary halfedge).
  HalfedgeId halfedge(VertexId v)           const { return vertex_to_edge_[v]; }
  HalfedgeId prev(HalfedgeId e)             const { return halfedges_[e].prev; }
  HalfedgeId next(HalfedgeId e)             const { return halfedges_[e].next; }
  HalfedgeId reverse(HalfedgeId e)          const { assert(valid(e)); return HalfedgeId(e.id^1); }
  EdgeId     edge(HalfedgeId e)             const { assert(valid(e)); return EdgeId(e.id>>1); }
  HalfedgeId halfedge(EdgeId e, bool which) const { assert(valid(e)); return HalfedgeId((e.id<<1)|which); }
  VertexId   src(HalfedgeId e)              const { return halfedges_[e].src; }
  VertexId   dst(HalfedgeId e)              const { return src(reverse(e)); }
  FaceId     face(HalfedgeId e)             const { return halfedges_[e].face; }
  HalfedgeId halfedge(FaceId f)             const { return face_to_edge_[f]; }
  HalfedgeId left(HalfedgeId e)             const { return reverse(prev(e)); }
  HalfedgeId right(HalfedgeId e)            const { return next(reverse(e)); }

  // Check id validity
  bool valid(VertexId v)   const { return vertex_to_edge_.valid(v); }
  bool valid(HalfedgeId e) const { return halfedges_.valid(e); }
  bool valid(EdgeId e)     const { return unsigned(e.id<<1)<unsigned(halfedges_.size()); }
  bool valid(FaceId f)     const { return face_to_edge_.valid(f); }

  // Check for boundaries
  bool is_boundary(HalfedgeId e) const { return !face(e).valid(); }
  bool is_boundary(EdgeId e)     const { return is_boundary(halfedge(e,0)) || is_boundary(halfedge(e,1)); }
  bool is_boundary(VertexId v)   const { const auto e = halfedge(v); return !e.valid() || !face(e).valid(); }
  bool isolated(VertexId v)      const { return !halfedge(v).valid(); }
  GEODE_CORE_EXPORT bool has_boundary() const;
  GEODE_CORE_EXPORT bool is_manifold() const;
  GEODE_CORE_EXPORT bool is_manifold_with_boundary() const;

  // Tuples or iterable ranges of neighbors
  Vector<HalfedgeId,3> halfedges(FaceId f) const {
    const auto e0 = halfedge(f);
    return vec(e0,next(e0),prev(e0));
  }
  Vector<VertexId,2> vertices(HalfedgeId e) const {
    return vec(src(e),dst(e));
  }
  Vector<VertexId,3> vertices(FaceId f) const {
    const auto e = halfedges(f);
    return vec(src(e.x),src(e.y),src(e.z));
  }
  inline Range<OutgoingCirculator> outgoing(VertexId v) const;

  // Iterate over vertices, edges, or faces
  Range<IdIter<VertexId>>   vertices()  const { return Range<IdIter<VertexId>>  (VertexId(0),  VertexId(  n_vertices())); }
  Range<IdIter<HalfedgeId>> halfedges() const { return Range<IdIter<HalfedgeId>>(HalfedgeId(0),HalfedgeId(n_halfedges())); }
  Range<IdIter<EdgeId>>     edges()     const { return Range<IdIter<EdgeId>>    (EdgeId(0),    EdgeId(    n_edges())); }
  Range<IdIter<FaceId>>     faces()     const { return Range<IdIter<FaceId>>    (FaceId(0),    FaceId(    n_faces())); }

  // Find a halfedge between two vertices, or return an invalid id if none exists.
  GEODE_CORE_EXPORT HalfedgeId halfedge(VertexId v0, VertexId v1) const;

  // Find the halfedge between two faces, or return an invalid id if none exists.  The halfedge belongs to the first face.
  // This function works correctly if the input faces are invalid.
  GEODE_CORE_EXPORT HalfedgeId common_halfedge(FaceId f0, FaceId f1) const;

  // Extract all triangles as a flat array
  Array<Vector<int,3>> elements() const;

  // Compute the edge degree of a vertex in O(degree) time.
  GEODE_CORE_EXPORT int degree(VertexId v) const;

  // Compute all boundary loops
  Nested<HalfedgeId> boundary_loops() const;

  // Compute the Euler characteristic
  int chi() const {
    return n_vertices()-n_edges()+n_faces();
  }

  // Add a new isolated vertex and return its id
  GEODE_CORE_EXPORT VertexId add_vertex();

  // Add n isolated vertices and return the first id
  GEODE_CORE_EXPORT void add_vertices(int n);

  // Add a new face.  If the result would not be manifold, no change is made and ValueError is thrown (TODO: throw a better exception).
  GEODE_CORE_EXPORT FaceId add_face(Vector<VertexId,3> v);

  // Add many new faces
  GEODE_CORE_EXPORT void add_faces(RawArray<const Vector<int,3>> vs);

  // Split a face into three by inserting a new vertex in the center
  GEODE_CORE_EXPORT VertexId split_face(FaceId f);

  // Split a face into three by inserting an existing isolated vertex in the center.  Afterwards, face(halfedge(c))==f.
  GEODE_CORE_EXPORT void split_face(FaceId f, VertexId c);

  // Check whether an edge flip would result in a manifold mesh
  GEODE_CORE_EXPORT bool is_flip_safe(HalfedgeId e) const;

  // Flip the two triangles adjacent to a given halfedge.  The routines throw an exception if is_flip_safe fails; call unsafe_flip_edge if you've already checked.
  GEODE_CORE_EXPORT void flip_edge(HalfedgeId e);

  // Permute vertices: vertices v becomes vertex permutation[v]
  GEODE_CORE_EXPORT void permute_vertices(RawArray<const int> permutation, bool check=false);

  // Run an expensive internal consistency check.  Safe to call even if the structure arrays are random noise.
  GEODE_CORE_EXPORT void assert_consistent(bool check_double_halfedges = true) const;

  // Print internal structure to Log::cout.  Safe to call even if the structure arrays are random noise.
  GEODE_CORE_EXPORT void dump_internals() const;

  // The remaining functions are mainly for internal use, or for external routines that perform direct surgery
  // on the halfedge structure.  Use with caution!

  // Link two edges together
  void unsafe_link(HalfedgeId p, HalfedgeId n) {
    halfedges_[p].next = n;
    halfedges_[n].prev = p;
  }

  // Link a face and edges together
  void unsafe_link_face(FaceId f, Vector<HalfedgeId,3> e) {
    face_to_edge_[f] = e.x;
    halfedges_[e.x].face = halfedges_[e.y].face = halfedges_[e.z].face = f;
    unsafe_link(e.x,e.y);
    unsafe_link(e.y,e.z);
    unsafe_link(e.z,e.x);
  }

  // Flip an edge assuming is_flip_safe(e)
  GEODE_CORE_EXPORT void unsafe_flip_edge(HalfedgeId e);

  // Remove a face from the mesh, shuffling face and halfedge ids in the process.
  // Vertex ids are untouched, and in particular isolated vertices are not erased.
  GEODE_CORE_EXPORT void unsafe_erase_face(FaceId f);

  // Remove the last vertex from the mesh, shuffling face and halfedge ids in the process.
  // This exists solely to erase sentinel vertices created by Delaunay.
  GEODE_CORE_EXPORT void unsafe_erase_last_vertex();
};

// Use only through HalfedgeMesh::outgoing()
struct OutgoingCirculator {
  const HalfedgeMesh& mesh;
  HalfedgeId e;
  bool first;
  OutgoingCirculator(const HalfedgeMesh& mesh, HalfedgeId e, bool first) : mesh(mesh), e(e), first(first) {}
  void operator++() { e = mesh.left(e); first = false; }
  bool operator!=(OutgoingCirculator o) { return first || e!=o.e; } // For use only inside range-based for loops
  HalfedgeId operator*() const { return e; }
};

inline Range<OutgoingCirculator> HalfedgeMesh::outgoing(VertexId v) const {
  const auto e = halfedge(v);
  const OutgoingCirculator c(*this,e,e.valid());
  return Range<OutgoingCirculator>(c,c);
}

}
