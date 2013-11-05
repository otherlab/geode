#pragma once

#include <geode/mesh/TriangleTopology.h>
#include <geode/geometry/Triangle3d.h>

namespace geode {

#if 0

// A manifold triangle mesh with positions.
// Keeps positions consistent when vertex ids change.
// Some TriangleTopology functions are hidden, some are overridden, but since they're not virtual,
// it is possible to break a TriangleMesh by casting a pointer to it to a TriangleTopology pointer.
class TriangleMesh: public Object {
  GEODE_DECLARE_TYPE(GEODE_CORE_EXPORT)
  typedef Object Base;

protected:
  GEODE_CORE_EXPORT TriangleMesh(); // this creates a new Topology and a new positions field
  GEODE_CORE_EXPORT TriangleMesh(RawArray<const Vector<int,3>> const &tris, RawArray<const Vector<real,3>> const &X);  // creates a new topology and X and copies X
  GEODE_CORE_EXPORT TriangleMesh(TriangleSoup const &soup, RawArray<const Vector<real,3>> const &X); // creates a new topology and X and copies X
  GEODE_CORE_EXPORT TriangleMesh(TriangleTopology const &topo, Field<Vector<real,3>, VertexId> const &X); // We share data with position field

private:
  // hide these methods (they add vertex ids without adding the position)
  VertexId add_vertex();
  VertexId add_vertices(int);
  VertexId split_face(FaceId);

  // hide these methods (they change ids without reporting in what way)
  void erase_face_with_reordering(FaceId f);
  void erase_last_vertex_with_reordering();

public:

  TriangleTopology const &topo();
  Field<Vector<real,3>, VertexId> X;

  // copy the mesh, with a new, independent X field
  GEODE_CORE_EXPORT Ref<TriangleMesh> copy() const;

  // add a vertex
  GEODE_CORE_EXPORT VertexId add_vertex(Vector<real,3> const &x);

  // add a bunch of vertices (returns the id of the first, ids are contiguous)
  GEODE_CORE_EXPORT VertexId add_vertices(RawArray<const Vector<real,3>> X);

  // add another TriangleMesh (add its vertices and faces)
  GEODE_CORE_EXPORT Tuple<Array<int>, Array<int>, Array<int>> add(TriangleMesh const &mesh);

  // Permute vertices: vertex v becomes vertex permutation[v]
  GEODE_CORE_EXPORT void permute_vertices(RawArray<const int> permutation, bool check=false);

  // Compact the data structure, removing all erased primitives. Returns a tuple of permutations for
  // vertices, faces, and boundary halfedges, such that the old primitive i now has index permutation[i].
  GEODE_CORE_EXPORT Tuple<Array<int>, Array<int>, Array<int>> collect_garbage();

  // Local geometry //////////////////////////////////////////////////////////////////////

  // get a triangle area
  inline real area(FaceId fh) const { return area(X, fh); }

  // get a triangle or edge centroid
  GEODE_CORE_EXPORT Vector<real,3> centroid(FaceId fh) const;
  GEODE_CORE_EXPORT Vector<real,3> centroid(HalfedgeId eh) const;

  // get the face as a Triangle
  GEODE_CORE_EXPORT Triangle<Vector<real, 3> > triangle(FaceId fh) const;

  // get an edge as a Segment
  GEODE_CORE_EXPORT Segment<Vector<real, 3> > segment(HalfedgeId heh) const;

  // compute the cotan weight for an edge
  GEODE_CORE_EXPORT real cotan_weight(HalfedgeId eh) const;

  // just for completeness, get a position
  GEODE_CORE_EXPORT Vector<real,3> point(VertexId) const;

  // get an interpolated position
  GEODE_CORE_EXPORT Vector<real,3> point(HalfedgeId eh, real t) const;
  GEODE_CORE_EXPORT Vector<real,3> point(HalfedgeId eh, Vector<real,2> const &bary) const;
  GEODE_CORE_EXPORT Vector<real,3> point(FaceId fh, Vector<real,3> const &bary) const;

  // compute the normal of a face, edge, or vertex
  GEODE_CORE_EXPORT Vector<real,3> normal(FaceId) const;
  GEODE_CORE_EXPORT Vector<real,3> normal(HalfedgeId) const; // simple average of incident faces
  GEODE_CORE_EXPORT Vector<real,3> normal(VertexId) const; // cotan-weighted average of incident faces

  // get an interpolated normal at any point on the mesh (interpolates the vertex normals)
  GEODE_CORE_EXPORT Vector<real,3> normal(HalfedgeId eh, real t) const;
  GEODE_CORE_EXPORT Vector<real,3> normal(HalfedgeId eh, Vector<real,2> const &bary) const;
  GEODE_CORE_EXPORT Vector<real,3> normal(FaceId fh, Vector<real,3> const &bary) const;

  // Global geometry /////////////////////////////////////////////////////////////////////

  // bounding box
  GEODE_CORE_EXPORT Box<Vector<real,3> > bounding_box() const;

  // bounding box of just some faces
  GEODE_CORE_EXPORT Box<Vector<real,3> > bounding_box(RawArray<const FaceId> const &faces) const;

  // area weighted centroid of the mesh
  GEODE_CORE_EXPORT Vector<real,3> centroid() const;

  // mean edge length of the mesh
  GEODE_CORE_EXPORT real mean_edge_length() const;
};

#endif

}
