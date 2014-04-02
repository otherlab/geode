//#####################################################################
// Class SegmentSoup
//#####################################################################
//
// SegmentSoup stores immutable topology for a segment mesh.  The advantage
// of immutability is that we don't have to worry about acceleration structures
// becoming invalid, and we can check validity once at construction time.
//
//#####################################################################
#pragma once

#include <geode/array/Array.h>
#include <geode/array/Nested.h>
#include <geode/utility/Object.h>
#include <geode/utility/Ptr.h>
#include <geode/utility/Ref.h>
#include <geode/vector/Vector.h>
#include <geode/structure/Tuple.h>
namespace geode {

class SegmentSoup : public Object {
public:
  GEODE_NEW_FRIEND
  typedef Object Base;
  typedef Vector<real,2> TV2;
  static const int d = 1;

  Array<const int> vertices; // = scalar_view(elements)
  Array<const Vector<int,2>> elements;
private:
  const int node_count;
  mutable Nested<int> neighbors_;
  mutable Nested<int> incident_elements_;
  mutable Array<Vector<int,2>> adjacent_elements_;
  mutable Tuple<Nested<const int>,Nested<const int>> polygons_;
  mutable bool bending_tuples_valid;
  mutable Array<Vector<int,3>> bending_tuples_;

protected:
  GEODE_CORE_EXPORT explicit SegmentSoup(Array<const Vector<int,2>> elements, const int min_nodes=0);

  int compute_node_count() const;
public:
  ~SegmentSoup();

  int nodes() const {
    return node_count;
  }

  Ref<const SegmentSoup> segment_soup() const {
    return ref(*this);
  }

  // Decompose segment mesh into maximal manifold contours, returning closed-contours, open-contours.
  // Nonmanifold vertices will show up several times in different open contours.
  GEODE_CORE_EXPORT const Tuple<Nested<const int>,Nested<const int>>& polygons() const;

  GEODE_CORE_EXPORT Nested<const int> neighbors() const; // vertices to vertices
  GEODE_CORE_EXPORT Nested<const int> incident_elements() const; // vertices to segments
  GEODE_CORE_EXPORT Array<const Vector<int,2>> adjacent_elements() const; // segment to segments
  GEODE_CORE_EXPORT Array<TV2> element_normals(RawArray<const TV2> X) const;
  GEODE_CORE_EXPORT Array<int> nonmanifold_nodes(bool allow_boundary) const;
  GEODE_CORE_EXPORT Array<const Vector<int,3>> bending_tuples() const;
};

}
