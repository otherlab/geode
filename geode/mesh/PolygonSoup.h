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

#include <geode/array/Array.h>
#include <geode/mesh/forward.h>
#include <geode/python/Object.h>
#include <geode/python/Ptr.h>
#include <geode/python/Ref.h>
namespace geode {

class PolygonSoup : public Object {
public:
  GEODE_DECLARE_TYPE(GEODE_CORE_EXPORT)
  typedef Object Base;

  const Array<const int> counts; // number of vertices in each polygon
  const Array<const int> vertices; // indices of each polygon flattened into a single array
private:
  const int node_count, half_edge_count;
  mutable Ptr<SegmentSoup> segment_soup_;
  mutable Ptr<TriangleSoup> triangle_mesh_;

protected:
  explicit PolygonSoup(Array<const int> counts, Array<const int> vertices, const int min_nodes=0);
public:
  ~PolygonSoup();

  int nodes() const {
    return node_count;
  }

  GEODE_CORE_EXPORT Ref<SegmentSoup> segment_soup() const;
  GEODE_CORE_EXPORT Ref<TriangleSoup> triangle_mesh() const;
};
}
