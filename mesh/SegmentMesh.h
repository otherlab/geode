//#####################################################################
// Class SegmentMesh
//#####################################################################
//
// SegmentMesh stores immutable topology for a segment mesh.  The advantage
// of immutability is that we don't have to worry about acceleration structures
// becoming invalid, and we can check validity once at construction time.
//
//#####################################################################
#pragma once

#include <other/core/array/Array.h>
#include <other/core/array/NestedArray.h>
#include <other/core/python/Object.h>
#include <other/core/python/Ptr.h>
#include <other/core/python/Ref.h>
#include <other/core/vector/Vector.h>
namespace other {

class SegmentMesh : public Object {
public:
  OTHER_DECLARE_TYPE
  typedef Object Base;
  typedef Vector<real,2> TV2;
  static const int d = 1;

  Array<const int> vertices; // = scalar_view(elements)
  Array<const Vector<int,2> > elements;
private:
  const int node_count;
  mutable NestedArray<int> neighbors_; 
  mutable NestedArray<int> incident_elements_;
  mutable Array<Vector<int,2> > adjacent_elements_;
  mutable Vector<NestedArray<const int>,2> polygons_;

protected:
  SegmentMesh(Array<const Vector<int,2> > elements) OTHER_EXPORT;
public:
  ~SegmentMesh();

  int nodes() const
  {return node_count;}

  Ref<const SegmentMesh> segment_mesh() const
  {return ref(*this);}

  // Decompose segment mesh into maximal manifold contours, returning closed-contours, open-contours.
  // Nonmanifold vertices will show up several times in different open contours.
  const Vector<NestedArray<const int>,2>& polygons() const OTHER_EXPORT;

  NestedArray<const int> neighbors() const OTHER_EXPORT; // vertices to vertices
  NestedArray<const int> incident_elements() const OTHER_EXPORT; // vertices to segments
  Array<const Vector<int,2> > adjacent_elements() const OTHER_EXPORT; // segment to segments
  Array<TV2> element_normals(RawArray<const TV2> X) const OTHER_EXPORT;
  Array<int> nonmanifold_nodes(bool allow_boundary) const OTHER_EXPORT;
};

}
