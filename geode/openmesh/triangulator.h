#pragma once

#include <geode/openmesh/TriMesh.h>
#include <geode/utility/unordered.h>
#ifdef GEODE_OPENMESH
namespace geode {

// base class for edge priority classes used in triangulators
class EdgePriority {
public:
  EdgePriority();
  virtual ~EdgePriority();

  // mark an edge as used and from now on forbidden
  virtual void used_edge(VertexHandle v1, VertexHandle v2) = 0;

  virtual double operator()(VertexHandle v1, VertexHandle v2) = 0;
};

GEODE_CORE_EXPORT int triangulate_face(TriMesh &mesh, std::vector<VertexHandle> const &face, std::vector<FaceHandle> &faces,
                     EdgePriority &ep, bool debug = false, int depth = 0);

GEODE_CORE_EXPORT int triangulate_cylinder(TriMesh &mesh, std::vector<VertexHandle> const &ring1, std::vector<VertexHandle> const &ring2,
                                           std::vector<FaceHandle> &faces, EdgePriority &ep, bool debug = false);

// a caching version of the edge priority
class GEODE_CORE_CLASS_EXPORT CachedEdgePriority : public EdgePriority {
protected:
  unordered_map<Vector<VertexHandle, 2>, double, Hasher> cache;
  virtual double computePriority(VertexHandle v1, VertexHandle v2) = 0;
public:
  GEODE_CORE_EXPORT CachedEdgePriority();
  GEODE_CORE_EXPORT ~CachedEdgePriority();
  GEODE_CORE_EXPORT virtual void used_edge(VertexHandle v1, VertexHandle v2);
  GEODE_CORE_EXPORT virtual double operator()(VertexHandle v1, VertexHandle v2);
};

// An edge priority class giving precedence to short edges.
class GEODE_CORE_CLASS_EXPORT ShortEdgePriority : public CachedEdgePriority {
public:
  GEODE_CORE_EXPORT ShortEdgePriority(TriMesh const &mesh);
  GEODE_CORE_EXPORT ~ShortEdgePriority();
protected:
  TriMesh const &mesh;
  virtual double computePriority(VertexHandle v1, VertexHandle v2);
};

}
#endif // GEODE_OPENMESH
