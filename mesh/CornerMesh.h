// A corner data structure representing oriented triangle meshes.
// This class represents topology only, not geometry.
#pragma once

#include <other/core/mesh/ids.h>
#include <other/core/array/Field.h>
namespace other {

struct CornerMeshOutgoing;
template<class Id> struct CornerMeshIter;

// A CornerMesh consists of vertices and faces connected into an oriented manifold,
// plus special boundary halfedges along each boundary curve.  Treating boundary
// halfedges specially removes a bit of the elegance of the corner mesh, but has the
// huge advantage that code written for halfedge meshes can be ported relatively easily.
//
// Internal halfedges are represented implicitly as HalfedgeId(3*f+i), where f is the face
// and i in [0,3) is the index of the source vertex in the face.  Boundary halfedges are
// represented by -1-b where b is the boundary id.
//
// Warning: Since boundary halfedges are treated specially, fields indexed by HalfedgeId
// do not contain entries for boundary halfedges.
//
// The data structure is as follows:
// 1. Each finite vertex v has a pointer to one of its outgoing halfedges, named halfedge(v).
//    If v is a boundary vertex, halfedge(v) is a boundary halfedge.
// 2. Each triangle f has three pointers to vertices and three pointers to neighbor halfedges
//    in neighbor faces, named vertices(f) and neighbors(f).
// 4. Each boundary edge stores its source vertex, prev and next boundary edges, and its reverse.
//
// Important properties and invariants:
// 1. The topology is always locally manifold, with the proviso that a vertex may
//    have multiple disjoint boundary curves (this is common when triangles are
//    added from triangle soup.
// 2. If a vertex is a boundary vertex, halfedge(v) is a boundary halfedge.
// 3. If e is a halfedge, either e or reverse(e) is not a boundary.
// 4. There are no self loops: src(e) != dst(e).
// 5. There is at most one edge between each pair of vertices.
//
// Performance considerations:
// 1. IMPORTANT: Since boundary halfedges are stored specially, removing a single face in the interior
//    of a mesh will result in three new halfedge structs, increasing the storage costs.  If you plan
//    to fill in the missing space with further triangles, use alternative modification primitives
//    such as split_face or flip_edge (or new routines written on request).
// 2. Similarly, adding triangles one by one using add_face is much more expensive than a single
//    bulk call to add_faces.
// 3. For a mesh with v vertices, f faces, and b boundary edges, the storage costs are roughly
//    1v + 6f + 4b = 13 ints/vertex assuming low genus and few boundary vertices.
//
// CornerMesh deliberately exposes the arrays containing the data structure details as public API,
// so that users with special requirements may perform surgery on the raw structure.  Normal use
// should go through the high level interface.
//
// TODO: Check in add_face/add_vertex whether we exceed the data structure limits.

class CornerMesh : public Object {
public:
  OTHER_DECLARE_TYPE(OTHER_CORE_EXPORT)
  typedef Object Base;

  // Various feature counts, exluding deleted entries
  int n_vertices_;
  int n_faces_;
  int n_boundary_edges_;

  // Flat arrays describing the mesh structure.  Do not use these directly unless you have a good reason.
  struct FaceInfo {
    Vector<VertexId,3> vertices; // vertices.x==deleted_id if the face is deleted
    Vector<HalfedgeId,3> neighbors;
  };
  struct BoundaryInfo {
    HalfedgeId prev, next; // These always point to boundaries.  Deleted boundary edges are connected into a linked list via next.
    HalfedgeId reverse; // Always points to an interior halfedge
    VertexId src; // If deleted, src = deleted_id
  };
  Field<FaceInfo,FaceId> faces_;
  Field<HalfedgeId,VertexId> vertex_to_edge_; // outgoing halfedge, invalid if isolated, deleted_id if deleted
  Array<BoundaryInfo> boundaries_; // If HalfedgeId(-1-b) is a boundary halfedge, boundaries_[b] is its info

  // The linked list of deleted boundary edges
  HalfedgeId deleted_boundaries_;

protected:
  OTHER_CORE_EXPORT CornerMesh();
  OTHER_CORE_EXPORT CornerMesh(const CornerMesh& mesh);
public:
  ~CornerMesh();

  // Copy the mesh
  OTHER_CORE_EXPORT Ref<CornerMesh> copy() const;

  // Count various features, excluding deleted ids.  If you want to include deletions, use faces_.size() and such.
  int n_vertices()       const { return n_vertices_; }
  int n_faces()          const { return n_faces_; }
  int n_edges()          const { return (3*n_faces_+n_boundary_edges_)>>1; }
  int n_boundary_edges() const { return n_boundary_edges_; }

  // Walk around the mesh.  These always succeed given valid ids, but may return invalid ids as a result (e.g., the face of a boundary halfedge).
  inline HalfedgeId halfedge(VertexId v)        const;
  inline HalfedgeId prev    (HalfedgeId e)      const;
  inline HalfedgeId next    (HalfedgeId e)      const;
  inline HalfedgeId reverse (HalfedgeId e)      const;
  inline VertexId   src     (HalfedgeId e)      const;
  inline VertexId   dst     (HalfedgeId e)      const;
  inline FaceId     face    (HalfedgeId e)      const;
  inline VertexId   vertex  (FaceId f, int i=0) const;
  inline HalfedgeId halfedge(FaceId f, int i=0) const;
  inline HalfedgeId left    (HalfedgeId e)      const;
  inline HalfedgeId right   (HalfedgeId e)      const;

  // Check id validity or deletion.  A deleted id is considered invalid.
  inline bool valid(VertexId v)   const;
  inline bool valid(HalfedgeId e) const;
  inline bool valid(FaceId f)     const;
  inline bool deleted(VertexId v)   const;
  inline bool deleted(HalfedgeId e) const;
  inline bool deleted(FaceId f)     const;

  // Check for boundaries
  inline bool is_boundary(HalfedgeId e) const;
  inline bool is_boundary(VertexId v)   const;
  inline bool isolated   (VertexId v)   const;
  OTHER_CORE_EXPORT bool has_boundary() const; // O(1) time
  OTHER_CORE_EXPORT bool is_manifold() const; // O(1) time
  OTHER_CORE_EXPORT bool is_manifold_with_boundary() const; // O(n) time
  OTHER_CORE_EXPORT bool has_isolated_vertices() const; // O(n) time

  // Tuples or iterable ranges of neighbors
  inline Vector<HalfedgeId,3> halfedges(FaceId f) const;
  inline Vector<VertexId,2> vertices(HalfedgeId e) const;
  inline Vector<VertexId,3> vertices(FaceId f) const;
  inline Range<CornerMeshOutgoing> outgoing(VertexId v) const;

  // Iterate over vertices, edges, or faces, skipping deleted entries.
  inline Range<CornerMeshIter<VertexId>>   vertices()           const;
  inline Range<CornerMeshIter<FaceId>>     faces()              const;
  inline Range<CornerMeshIter<HalfedgeId>> halfedges()          const;
  inline Range<CornerMeshIter<HalfedgeId>> boundary_edges()     const;
  inline Range<CornerMeshIter<HalfedgeId>> interior_halfedges() const;

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

  // Compute the Euler characteristic.  This is correct only if the mesh has been garbage collected.
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
  // WARNING: The all halfedge ids in the two adjacent faces are changed, and the new id of the argument edge is returned.
  OTHER_CORE_EXPORT HalfedgeId flip_edge(HalfedgeId e) OTHER_WARN_UNUSED_RESULT;

  // Run an expensive internal consistency check.  Safe to call even if the structure arrays are random noise.
  OTHER_CORE_EXPORT void assert_consistent() const;

  // Print internal structure to Log::cout.  Safe to call even if the structure arrays are random noise.
  OTHER_CORE_EXPORT void dump_internals() const;

  // The remaining functions are mainly for internal use, or for external routines that perform direct surgery
  // on the internal structure.  Use with caution!

  // Link two boundary edges together
  void unsafe_boundary_link(HalfedgeId p, HalfedgeId n) {
    assert(p.id<0 && n.id<0);
    boundaries_[-1-p.id].next = n;
    boundaries_[-1-n.id].prev = p;
  }

  // Link an interior halfedge with an arbitrary opposite halfedge
  void unsafe_set_reverse(FaceId f, int i, HalfedgeId r) {
    faces_[f].neighbors[i] = r;
    if (r.id>=0) { 
      const int f1 = r.id/3;
      faces_.flat[f1].neighbors[r.id-3*f1] = HalfedgeId(3*f.id+i);
    } else
      boundaries_[-1-r.id].reverse = HalfedgeId(3*f.id+i);
  }

  // Mark features as deleted
  inline void unsafe_set_deleted(VertexId v);
  inline void unsafe_set_deleted(FaceId f);
  inline void unsafe_set_deleted(HalfedgeId b); // Must be a boundary edge

  // Iterate over vertices, edges, or faces *without* skipping deleted entries.
  inline Range<IdIter<VertexId>>   unsafe_vertices()       const;
  inline Range<IdIter<FaceId>>     unsafe_faces()          const;
  inline Range<IdIter<HalfedgeId>> unsafe_boundary_edges() const;
  inline Range<IdIter<HalfedgeId>> unsafe_interior_edges() const;

  // Flip an edge assuming is_flip_safe(e).
  // WARNING: The all halfedge ids in the two adjacent faces are changed, and the new id of the argument edge is returned.
  OTHER_CORE_EXPORT HalfedgeId unsafe_flip_edge(HalfedgeId e) OTHER_WARN_UNUSED_RESULT;

  // Remove a face from the mesh, shuffling face and halfedge ids in the process.
  // Vertex ids are untouched, and in particular isolated vertices are not deleted.
  OTHER_CORE_EXPORT void unsafe_delete_face(FaceId f);

  // Remove the last vertex from the mesh, shuffling face and halfedge ids in the process.
  // This exists solely to delete sentinel vertices created by Delaunay.
  OTHER_CORE_EXPORT void unsafe_delete_last_vertex();
};

// Mesh walking routines
inline HalfedgeId CornerMesh::halfedge(VertexId v) const {
  assert(valid(v));
  return vertex_to_edge_[v];
}
inline VertexId CornerMesh::vertex(FaceId f, int i) const {
  assert(valid(f));
  return faces_[f].vertices[i];
}
inline HalfedgeId CornerMesh::halfedge(FaceId f, int i) const {
  assert(valid(f) && unsigned(i)<3);
  return HalfedgeId(3*f.id+i);
}
inline HalfedgeId CornerMesh::prev(HalfedgeId e) const {
  assert(valid(e));
  return e.id>=0 ? HalfedgeId(e.id+(e.id%3==0?2:-1)) : boundaries_[-1-e.id].prev;
}
inline HalfedgeId CornerMesh::next(HalfedgeId e) const {
  assert(valid(e));
  return e.id>=0 ? HalfedgeId(e.id+(e.id%3==2?-2:1)) : boundaries_[-1-e.id].next;
}
inline HalfedgeId CornerMesh::reverse(HalfedgeId e) const {
  assert(valid(e));
  if (e.id>=0) {
    const int f = e.id/3;
    return faces_.flat[f].neighbors[e.id-3*f];
  }
  return boundaries_[-1-e.id].reverse;
}
inline VertexId CornerMesh::src(HalfedgeId e) const {
  assert(valid(e));
  if (e.id>=0) {
    const int f = e.id/3;
    return faces_.flat[f].vertices[e.id-3*f];
  }
  return boundaries_[-1-e.id].src;
}
inline VertexId CornerMesh::dst(HalfedgeId e) const {
  assert(valid(e));
  if (e.id>=0) {
    const int f = e.id/3,
              i = e.id-3*f;
    return faces_.flat[f].vertices[i==2?0:i+1];
  }
  return boundaries_[-1-boundaries_[-1-e.id].next.id].src;
}
inline FaceId CornerMesh::face(HalfedgeId e) const {
  assert(valid(e));
  return e.id>=0 ? FaceId(e.id/3) : FaceId();
}
inline HalfedgeId CornerMesh::left(HalfedgeId e)  const { return reverse(prev(e)); }
inline HalfedgeId CornerMesh::right(HalfedgeId e) const { return next(reverse(e)); }

// Check id validity or deletion.  A deleted id is considered invalid.
inline bool CornerMesh::valid(VertexId v) const {
  return vertex_to_edge_.valid(v) && !deleted(v);
}
inline bool CornerMesh::valid(HalfedgeId e) const {
  return e.id>=0 ? valid(FaceId(e.id/3))
                 : boundaries_.valid(-1-e.id) && boundaries_[-1-e.id].src.id!=deleted_id;
}
inline bool CornerMesh::valid(FaceId f) const {
  return faces_.valid(f) && !deleted(f);
}
inline bool CornerMesh::deleted(VertexId v) const {
  return vertex_to_edge_[v].id==deleted_id;
}
inline bool CornerMesh::deleted(HalfedgeId e) const {
  return e.id>=0 ? faces_.flat[e.id/3].vertices.x.id==deleted_id
                 : boundaries_[-1-e.id].src.id==deleted_id;
}
inline bool CornerMesh::deleted(FaceId f) const {
  return faces_[f].vertices.x.id==deleted_id;
}

// Mark features as deleted
inline void CornerMesh::unsafe_set_deleted(VertexId v) {
  vertex_to_edge_[v].id = deleted_id;
  n_vertices_--;
}
inline void CornerMesh::unsafe_set_deleted(FaceId f) {
  faces_[f].vertices.x.id = deleted_id;
  n_faces_--;
}
inline void CornerMesh::unsafe_set_deleted(HalfedgeId b) {
  boundaries_[-1-b.id].src.id = deleted_id;
  boundaries_[-1-b.id].next = deleted_boundaries_;
  deleted_boundaries_ = b;
  n_boundary_edges_--;
}

// Check for boundaries
inline bool CornerMesh::is_boundary(HalfedgeId e) const { assert(valid(e)); return e.id<0; }
inline bool CornerMesh::is_boundary(VertexId v)   const { assert(valid(v)); return halfedge(v).id<0; }
inline bool CornerMesh::isolated   (VertexId v)   const { assert(valid(v)); return !halfedge(v).valid(); }

// Use only through CornerMesh::outgoing()
struct CornerMeshOutgoing {
  const CornerMesh& mesh;
  HalfedgeId e;
  bool first;
  CornerMeshOutgoing(const CornerMesh& mesh, HalfedgeId e, bool first) : mesh(mesh), e(e), first(first) {}
  void operator++() { e = mesh.left(e); first = false; }
  bool operator!=(CornerMeshOutgoing o) { return first || e!=o.e; } // For use only inside range-based for loops
  HalfedgeId operator*() const { return e; }
};

// Tuples or iterable ranges of neighbors
inline Vector<HalfedgeId,3> CornerMesh::halfedges(FaceId f) const {
  return vec(HalfedgeId(3*f.id+0),
             HalfedgeId(3*f.id+1),
             HalfedgeId(3*f.id+2));
}
inline Vector<VertexId,2> CornerMesh::vertices(HalfedgeId e) const {
  return vec(src(e),dst(e));
}
inline Vector<VertexId,3> CornerMesh::vertices(FaceId f) const {
  assert(valid(f));
  return faces_[f].vertices;
}
inline Range<CornerMeshOutgoing> CornerMesh::outgoing(VertexId v) const {
  const auto e = halfedge(v);
  const CornerMeshOutgoing c(*this,e,e.valid());
  return Range<CornerMeshOutgoing>(c,c);
}

// Use only throw vertices(), faces(), or boundary_edges()
template<class Id> struct CornerMeshIter {
  const CornerMesh& mesh;
  Id i;
  const Id end;

  CornerMeshIter(const CornerMesh& mesh, Id i_, Id end)
    : mesh(mesh), i(i_), end(end) {
    while (i!=end && mesh.deleted(i)) i.id++;
  }

  void operator++() {
    i.id++;
    while (i!=end && mesh.deleted(i)) i.id++;
  }

  bool operator!=(CornerMeshIter o) const { return i!=o.i; } // Assume &mesh==&o.mesh
  Id operator*() const { return i; }
};

// Iterate over vertices, edges, or faces, skipping deleted entries
inline Range<CornerMeshIter<VertexId>> CornerMesh::vertices() const {
  const VertexId end(vertex_to_edge_.size());
  return Range<CornerMeshIter<VertexId>>(CornerMeshIter<VertexId>(*this,VertexId(0),end),CornerMeshIter<VertexId>(*this,end,end));
}
inline Range<CornerMeshIter<FaceId>> CornerMesh::faces() const {
  const FaceId end(faces_.size());
  return Range<CornerMeshIter<FaceId>>(CornerMeshIter<FaceId>(*this,FaceId(0),end),CornerMeshIter<FaceId>(*this,end,end));
}
inline Range<CornerMeshIter<HalfedgeId>> CornerMesh::boundary_edges() const {
  const HalfedgeId end(0);
  return Range<CornerMeshIter<HalfedgeId>>(CornerMeshIter<HalfedgeId>(*this,HalfedgeId(-boundaries_.size()),end),CornerMeshIter<HalfedgeId>(*this,end,end));
}
inline Range<CornerMeshIter<HalfedgeId>> CornerMesh::interior_halfedges() const {
  const HalfedgeId end(3*faces_.size());
  return Range<CornerMeshIter<HalfedgeId>>(CornerMeshIter<HalfedgeId>(*this,HalfedgeId(0),end),CornerMeshIter<HalfedgeId>(*this,end,end));
}
inline Range<CornerMeshIter<HalfedgeId>> CornerMesh::halfedges() const {
  const HalfedgeId end(3*faces_.size());
  return Range<CornerMeshIter<HalfedgeId>>(CornerMeshIter<HalfedgeId>(*this,HalfedgeId(-boundaries_.size()),end),CornerMeshIter<HalfedgeId>(*this,end,end));
}

// Iterate over vertices, edges, or faces *without* skipping deleted entries
inline Range<IdIter<VertexId>> CornerMesh::unsafe_vertices() const {
  return Range<IdIter<VertexId>>(VertexId(0),VertexId(vertex_to_edge_.size()));
}
inline Range<IdIter<FaceId>> CornerMesh::unsafe_faces() const {
  return Range<IdIter<FaceId>>(FaceId(0),FaceId(faces_.size()));
}
inline Range<IdIter<HalfedgeId>> CornerMesh::unsafe_boundary_edges() const {
  return Range<IdIter<HalfedgeId>>(HalfedgeId(-boundaries_.size()),HalfedgeId(0));
}

}
