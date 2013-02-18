// A halfedge data structure representing oriented triangle meshes.
// This class represents topology only, not geometry.
#pragma once

#include <other/core/array/Field.h>
namespace other {

using std::ostream;
template<class Id> struct IdIter;
struct OutgoingCirculator;

// Important invariants:
// 1. The topology is always locally manifold, with the proviso that a vertex may
//    have multiple disjoint boundary curves (this is common when triangles are
//    added from triangle soup.
// 2. If a vertex is a boundary vertex, its outgoing halfedge is a boundary halfedge.
// 3. If e is a halfedge, either e or reverse(e) is not a boundary.
// 4. There are no self loops: src(e) != dst(e).
// 5. Given vertices v0,v1, there is at most one halfedge from v0 to v1.

#define OTHER_DEFINE_ID(Name) \
  struct Name { \
    int id; \
    Name() : id(-1) {} \
    explicit Name(int id) : id(id) {} \
    int idx() const { return id; } \
    bool valid() const { return id>=0; } \
    bool operator==(Name i) const { return id==i.id; } \
    bool operator!=(Name i) const { return id!=i.id; } \
    bool operator< (Name i) const { return id< i.id; } \
    bool operator<=(Name i) const { return id<=i.id; } \
    bool operator> (Name i) const { return id> i.id; } \
    bool operator>=(Name i) const { return id>=i.id; } \
    explicit operator int() const { return id; } \
  }; \
  template<> struct is_packed_pod<Name> : mpl::true_ {}; \
  static inline PyObject* to_python(Name i) { return to_python(i.id); } \
  template<> struct FromPython<Name>{static Name convert(PyObject* o) { return Name(FromPython<int>::convert(o)); }}; \
  static inline ostream& operator<<(ostream& output, Name i) { return output<<i.id; }
OTHER_DEFINE_ID(VertexId)
OTHER_DEFINE_ID(HalfedgeId)
OTHER_DEFINE_ID(FaceId)
OTHER_DECLARE_VECTOR_CONVERSIONS(OTHER_CORE_EXPORT,3,VertexId)

template<class Id> struct IdIter {
  Id i;
  IdIter(Id i) : i(i) {}
  void operator++() { i.id++; }
  bool operator!=(IdIter o) const { return i!=o.i; }
  Id operator*() const { return i; }
};

class HalfedgeMesh : public Object {
public:
  OTHER_DECLARE_TYPE(OTHER_CORE_EXPORT)
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
  OTHER_CORE_EXPORT HalfedgeMesh();
  OTHER_CORE_EXPORT HalfedgeMesh(const HalfedgeMesh& mesh);
public:
  ~HalfedgeMesh();

  // Copy the mesh
  OTHER_CORE_EXPORT Ref<HalfedgeMesh> copy() const;

  // Count various features
  int n_vertices()  const { return vertex_to_edge_.size(); }
  int n_halfedges() const { return halfedges_.size(); }
  int n_edges()     const { return halfedges_.size()>>1; }
  int n_faces()     const { return face_to_edge_.size(); }

  // Walk around the halfedge structure.  These always succeed given valid ids,
  // but may return invalid ids as a result (e.g., the face of a boundary halfedge).
  HalfedgeId halfedge(VertexId v)  const { return vertex_to_edge_[v]; }
  HalfedgeId prev(HalfedgeId e)    const { return halfedges_[e].prev; }
  HalfedgeId next(HalfedgeId e)    const { return halfedges_[e].next; }
  HalfedgeId reverse(HalfedgeId e) const { assert(valid(e)); return HalfedgeId(e.id^1); }
  VertexId   src(HalfedgeId e)     const { return halfedges_[e].src; }
  VertexId   dst(HalfedgeId e)     const { return src(reverse(e)); }
  FaceId     face(HalfedgeId e)    const { return halfedges_[e].face; }
  HalfedgeId halfedge(FaceId f)    const { return face_to_edge_[f]; }
  HalfedgeId left(HalfedgeId e)    const { return reverse(prev(e)); }
  HalfedgeId right(HalfedgeId e)   const { return next(reverse(e)); }

  // Check id validity
  bool valid(VertexId v)   const { return vertex_to_edge_.valid(v); }
  bool valid(HalfedgeId e) const { return halfedges_.valid(e); }
  bool valid(FaceId f)     const { return face_to_edge_.valid(f); }

  // Check for boundaries
  bool is_boundary(HalfedgeId e) const { return !face(e).valid(); }
  bool is_boundary(VertexId v)   const { const auto e = halfedge(v); return !e.valid() || !face(e).valid(); }
  bool isolated(VertexId v)      const { return !halfedge(v).valid(); }
  OTHER_CORE_EXPORT bool has_boundary() const;
  OTHER_CORE_EXPORT bool is_manifold() const;
  OTHER_CORE_EXPORT bool is_manifold_with_boundary() const;

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
  Range<IdIter<FaceId>>     faces()     const { return Range<IdIter<FaceId>>    (FaceId(0),    FaceId(    n_faces())); }

  // Find a halfedge between two vertices, or return an invalid id if none exists.
  OTHER_CORE_EXPORT HalfedgeId halfedge(VertexId v0, VertexId v1) const;

  // Find the halfedge between two faces, or return an invalid id if none exists.  The halfedge belongs to the first face.
  // This function works correctly if the input faces are invalid.
  OTHER_CORE_EXPORT HalfedgeId common_halfedge(FaceId f0, FaceId f1) const;

  // Extract all triangles as a flat array
  Array<Vector<int,3>> elements() const;

  // Compute the edge degree of a vertex in O(degree) time.
  OTHER_CORE_EXPORT int degree(VertexId v) const;

  // Compute all boundary loops
  NestedArray<HalfedgeId> boundary_loops() const;

  // Compute the Euler characteristic
  int chi() const {
    return n_vertices()-n_edges()+n_faces();
  }

  // Add a new isolated vertex and return its id
  OTHER_CORE_EXPORT VertexId add_vertex();

  // Add n isolated vertices and return the first id
  OTHER_CORE_EXPORT void add_vertices(int n);

  // Add a new face.  If the result would not be manifold, no change is made and ValueError is thrown (TODO: throw a better exception).
  OTHER_CORE_EXPORT FaceId add_face(Vector<VertexId,3> v);

  // Add many new faces
  OTHER_CORE_EXPORT void add_faces(RawArray<const Vector<int,3>> vs);

  // Split a face into three by inserting a new vertex in the center
  OTHER_CORE_EXPORT VertexId split_face(FaceId f);

  // Split a face into three by inserting an existing isolated vertex in the center.  Afterwards, face(halfedge(c))==f.
  OTHER_CORE_EXPORT void split_face(FaceId f, VertexId c);

  // Check whether an edge flip would result in a manifold mesh
  OTHER_CORE_EXPORT bool is_flip_safe(HalfedgeId e) const;

  // Flip the two triangles adjacent to a given halfedge.  The routines throw an exception if is_flip_safe fails; call unsafe_flip_edge if you've already checked.
  OTHER_CORE_EXPORT void flip_edge(HalfedgeId e);

  // Run an expensive internal consistency check.  Safe to call even if the structure arrays are random noise.
  OTHER_CORE_EXPORT void assert_consistent() const;

  // Print internal structure to Log::cout.  Safe to call even if the structure arrays are random noise.
  OTHER_CORE_EXPORT void dump_internals() const;

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
  OTHER_CORE_EXPORT void unsafe_flip_edge(HalfedgeId e);

  // Remove a face from the mesh, shuffling face and halfedge ids in the process.
  // Vertex ids are untouched, and in particular isolated vertices are not deleted.
  OTHER_CORE_EXPORT void unsafe_delete_face(FaceId f);

  // Remove the last vertex from the mesh, shuffling face and halfedge ids in the process.
  // This exists solely to delete sentinel vertices created by Delaunay.
  OTHER_CORE_EXPORT void unsafe_delete_last_vertex();
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
