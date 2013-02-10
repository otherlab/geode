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
#include <other/core/structure/Tuple.h>
namespace other {

class SegmentMesh : public Object {
public:
  OTHER_DECLARE_TYPE(OTHER_CORE_EXPORT)
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
  mutable Tuple<NestedArray<const int>,NestedArray<const int>> polygons_;
  mutable bool bending_tuples_valid;
  mutable Array<Vector<int,3>> bending_tuples_;

protected:
  OTHER_CORE_EXPORT SegmentMesh(Array<const Vector<int,2> > elements);

  int compute_node_count() const;
public:
  ~SegmentMesh();

  int nodes() const
  {return node_count;}

  Ref<const SegmentMesh> segment_mesh() const
  {return ref(*this);}

  // Decompose segment mesh into maximal manifold contours, returning closed-contours, open-contours.
  // Nonmanifold vertices will show up several times in different open contours.
  OTHER_CORE_EXPORT const Tuple<NestedArray<const int>,NestedArray<const int>>& polygons() const;

  OTHER_CORE_EXPORT NestedArray<const int> neighbors() const; // vertices to vertices
  OTHER_CORE_EXPORT NestedArray<const int> incident_elements() const; // vertices to segments
  OTHER_CORE_EXPORT Array<const Vector<int,2> > adjacent_elements() const; // segment to segments
  OTHER_CORE_EXPORT Array<TV2> element_normals(RawArray<const TV2> X) const;
  OTHER_CORE_EXPORT Array<int> nonmanifold_nodes(bool allow_boundary) const;
  OTHER_CORE_EXPORT Array<const Vector<int,3>> bending_tuples() const;
};

}
