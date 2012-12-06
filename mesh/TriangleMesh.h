//#####################################################################
// Class TriangleMesh
//#####################################################################
//
// TriangleMesh stores immutable topology for a triangle mesh.  The advantage
// of immutability is that we don't have to worry about acceleration structures
// becoming invalid, and we can check validity once at construction time.
//
//#####################################################################
#pragma once

#include <other/core/array/Array.h>
#include <other/core/array/NestedArray.h>
#include <other/core/mesh/forward.h>
#include <other/core/python/Object.h>
#include <other/core/python/Ptr.h>
#include <other/core/python/Ref.h>
#include <other/core/vector/Vector.h>
namespace other {

class TriangleMesh : public Object {
  typedef real T;
  typedef Vector<T,2> TV2;
  typedef Vector<T,3> TV3;
public:
  OTHER_DECLARE_TYPE
  typedef Object Base;
  static const int d = 2;

  Array<const int> vertices; // flattened version of triangles
  Array<const Vector<int,3> > elements;
private:
  int node_count;
  mutable Ptr<SegmentMesh> segment_mesh_;
  mutable bool bending_quadruples_valid;
  mutable Array<Vector<int,4> > bending_quadruples_; // i,j,k,l means triangles (i,j,k),(k,j,l)
  mutable NestedArray<int> incident_elements_;
  mutable Array<Vector<int,3> > adjacent_elements_;
  mutable Ptr<SegmentMesh> boundary_mesh_;
  mutable Array<int> nodes_touched_;
  mutable NestedArray<const int> sorted_neighbors_;

protected:
  TriangleMesh(Array<const Vector<int,3> > elements) OTHER_EXPORT;
public:
  ~TriangleMesh();

  int nodes() const {
    return node_count;
  }

  // to match PolygonMesh
  Ref<const TriangleMesh> triangle_mesh() const {
    return ref(*this);
  }

  Ref<const SegmentMesh> segment_mesh() const OTHER_EXPORT;
  NestedArray<const int> incident_elements() const OTHER_EXPORT; // vertices to triangles
  Array<const Vector<int,3> > adjacent_elements() const OTHER_EXPORT; // triangles to triangles
  Ref<SegmentMesh> boundary_mesh() const OTHER_EXPORT; 
  Array<const Vector<int,4> > bending_quadruples() const OTHER_EXPORT;
  Array<const int> nodes_touched() const OTHER_EXPORT;
  NestedArray<const int> sorted_neighbors() const OTHER_EXPORT; // vertices to sorted one-ring
  T area(RawArray<const TV2> X) const OTHER_EXPORT;
  T volume(RawArray<const TV3> X) const OTHER_EXPORT; // assumes a closed surface
  T surface_area(RawArray<const TV3> X) const OTHER_EXPORT;
  Array<T> vertex_areas(RawArray<const TV3> X) const OTHER_EXPORT;
  Array<TV3> vertex_normals(RawArray<const TV3> X) const OTHER_EXPORT;
  Array<TV3> element_normals(RawArray<const TV3> X) const OTHER_EXPORT;
  Array<int> nonmanifold_nodes(bool allow_boundary) const OTHER_EXPORT;
};

}
