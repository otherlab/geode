//#####################################################################
// Class PolygonSoup
//#####################################################################
//
// PolygonSoup stores immutable topology for a polygon mesh.  The advantage
// of immutability is that we don't have to worry about acceleration structures
// becoming invalid, and we can check validity once at construction time.
//
//#####################################################################
#pragma once

#include <other/core/array/Array.h>
#include <other/core/mesh/forward.h>
#include <other/core/python/Object.h>
#include <other/core/python/Ptr.h>
#include <other/core/python/Ref.h>
namespace other {

class PolygonSoup : public Object {
public:
  OTHER_DECLARE_TYPE(OTHER_CORE_EXPORT)
  typedef Object Base;

  const Array<const int> counts; // number of vertices in each polygon
  const Array<const int> vertices; // indices of each polygon flattened into a single array
private:
  const int node_count, half_edge_count;
  mutable Ptr<SegmentMesh> segment_mesh_;
  mutable Ptr<TriangleSoup> triangle_mesh_;

protected:
  PolygonSoup(Array<const int> counts, Array<const int> vertices);
public:
  ~PolygonSoup();

  int nodes() const {
    return node_count;
  }

  OTHER_CORE_EXPORT Ref<SegmentMesh> segment_mesh() const;
  OTHER_CORE_EXPORT Ref<TriangleSoup> triangle_mesh() const;
};
}
