//#####################################################################
// Evaluate signed distances between a point cloud and a mesh
//#####################################################################
#pragma once

#include <geode/array/Array.h>
#include <geode/geometry/forward.h>
#include <geode/structure/Tuple.h>
#include <geode/vector/Vector3d.h>
namespace geode {

struct CloseTriangleInfo {
  real phi;
  Vector<real,3> normal;
  int triangle;
  Vector<real,3> weights;
};

GEODE_CORE_EXPORT void surface_levelset(const ParticleTree<Vector<real,3>>& particles,
                                        const SimplexTree<Vector<real,3>,2>& surface,
                                        RawArray<CloseTriangleInfo> info,
                                        const real max_distance, const bool compute_signs);

// Functional-style version: returns distance, normals, closest triangle, and barycentric weights per point.
GEODE_CORE_EXPORT Tuple<Array<real>,Array<Vector<real,3>>,Array<int>,Array<Vector<real,3>>>
surface_levelset(const ParticleTree<Vector<real,3>>& particles, const SimplexTree<Vector<real,3>,2>& surface,
                 const real max_distance, const bool compute_signs);

}
