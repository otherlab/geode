//#####################################################################
// Class TriangleSoup
//#####################################################################
//
// TriangleSoup stores immutable topology for a triangle mesh.  The advantage
// of immutability is that we don't have to worry about acceleration structures
// becoming invalid, and we can check validity once at construction time.
//
//#####################################################################
#pragma once

#include <geode/array/Array.h>
#include <geode/array/Nested.h>
#include <geode/mesh/forward.h>
#include <geode/python/Object.h>
#include <geode/python/Ptr.h>
#include <geode/python/Ref.h>
#include <geode/vector/Vector.h>
namespace geode {

class TriangleSoup : public Object {
  typedef real T;
  typedef Vector<T,2> TV2;
  typedef Vector<T,3> TV3;
public:
  GEODE_DECLARE_TYPE(GEODE_CORE_EXPORT)
  typedef Object Base;
  static const int d = 2;

  Array<const int> vertices; // flattened version of triangles
  Array<const Vector<int,3>> elements;
private:
  int node_count;
  mutable Ptr<SegmentSoup> segment_soup_;
  mutable bool bending_tuples_valid;
  mutable Array<Vector<int,4>> bending_tuples_; // i,j,k,l means triangles (i,j,k),(k,j,l)
  mutable Nested<int> incident_elements_;
  mutable Array<Vector<int,3>> triangle_edges_;
  mutable Array<Vector<int,3>> adjacent_elements_;
  mutable Ptr<SegmentSoup> boundary_mesh_;
  mutable Array<int> nodes_touched_;
  mutable Nested<const int> sorted_neighbors_;

protected:
  GEODE_CORE_EXPORT explicit TriangleSoup(Array<const Vector<int,3>> elements);
public:
  ~TriangleSoup();

  int nodes() const {
    return node_count;
  }

  // to match PolygonSoup
  Ref<const TriangleSoup> triangle_mesh() const {
    return ref(*this);
  }

  GEODE_CORE_EXPORT Ref<const SegmentSoup> segment_soup() const;
  GEODE_CORE_EXPORT Array<const Vector<int,3>> triangle_edges() const; // triangles to edges
  GEODE_CORE_EXPORT Nested<const int> incident_elements() const; // vertices to triangles
  GEODE_CORE_EXPORT Array<const Vector<int,3>> adjacent_elements() const; // triangles to triangles
  GEODE_CORE_EXPORT Ref<SegmentSoup> boundary_mesh() const;
  GEODE_CORE_EXPORT Array<const Vector<int,4>> bending_tuples() const;
  GEODE_CORE_EXPORT Array<const int> nodes_touched() const;
  GEODE_CORE_EXPORT Nested<const int> sorted_neighbors() const; // vertices to sorted one-ring
  GEODE_CORE_EXPORT T area(RawArray<const TV2> X) const;
  GEODE_CORE_EXPORT T volume(RawArray<const TV3> X) const; // assumes a closed surface
  GEODE_CORE_EXPORT T surface_area(RawArray<const TV3> X) const;
  GEODE_CORE_EXPORT Array<T> vertex_areas(RawArray<const TV3> X) const;
  GEODE_CORE_EXPORT Array<TV3> vertex_normals(RawArray<const TV3> X) const;
  GEODE_CORE_EXPORT Array<TV3> element_normals(RawArray<const TV3> X) const;
  GEODE_CORE_EXPORT Array<int> nonmanifold_nodes(bool allow_boundary) const;
};

}
