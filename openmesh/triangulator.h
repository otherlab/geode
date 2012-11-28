#pragma once

#include <other/core/openmesh/TriMesh.h>
#include <other/core/utility/tr1.h>
namespace other {
  
// base class for edge priority classes used in triangulators
class EdgePriority {
public:
  EdgePriority();
  virtual ~EdgePriority();

  // mark an edge as used and from now on forbidden
  virtual void used_edge(VertexHandle v1, VertexHandle v2) = 0;
  
  virtual double operator()(VertexHandle v1, VertexHandle v2) = 0;
};

int triangulate_face(TriMesh &mesh, std::vector<VertexHandle> const &face, std::vector<FaceHandle> &faces,
                     EdgePriority &ep, bool debug = false, int depth = 0);

int triangulate_cylinder(TriMesh &mesh, std::vector<VertexHandle> const &ring1, std::vector<VertexHandle> const &ring2, 
                         std::vector<FaceHandle> &faces, EdgePriority &ep, bool debug = false);

// a caching version of the edge priority
class CachedEdgePriority: public EdgePriority {
protected:
  std::tr1::unordered_map<Vector<VertexHandle, 2>, double, Hasher> cache;
  virtual double computePriority(VertexHandle v1, VertexHandle v2) = 0;
public:
  CachedEdgePriority();
  ~CachedEdgePriority();
  virtual void used_edge(VertexHandle v1, VertexHandle v2);
  virtual double operator()(VertexHandle v1, VertexHandle v2);
};

// An edge priority class giving precedence to short edges. 
class ShortEdgePriority: public CachedEdgePriority {
public:
  ShortEdgePriority(TriMesh const &mesh);
  ~ShortEdgePriority();
protected:
  TriMesh const &mesh;
  virtual double computePriority(VertexHandle v1, VertexHandle v2);
};

}
