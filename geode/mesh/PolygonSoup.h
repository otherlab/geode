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
#include <geode/utility/Object.h>
#include <geode/utility/Ptr.h>
#include <geode/utility/Ref.h>
namespace geode {

class PolygonSoup : public Object {
public:
  GEODE_NEW_FRIEND
  typedef Object Base;

  const Array<const int> counts; // number of vertices in each polygon
  const Array<const int> vertices; // indices of each polygon flattened into a single array
private:
  const int node_count, half_edge_count;
  mutable Ptr<const SegmentSoup> segment_soup_;
  mutable Ptr<const TriangleSoup> triangle_mesh_;

protected:
  explicit PolygonSoup(Array<const int> counts, Array<const int> vertices, const int min_nodes=0);
public:
  ~PolygonSoup();

  int nodes() const {
    return node_count;
  }

  GEODE_CORE_EXPORT Ref<const SegmentSoup> segment_soup() const;
  GEODE_CORE_EXPORT Ref<const TriangleSoup> triangle_mesh() const;
};
}
