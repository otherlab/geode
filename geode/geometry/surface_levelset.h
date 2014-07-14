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

GEODE_CORE_EXPORT void evaluate_surface_levelset(const ParticleTree<Vector<real,3>>& particles,
                                                 const SimplexTree<Vector<real,3>,2>& surface,
                                                 RawArray<CloseTriangleInfo> info,
                                                 real max_distance, bool compute_signs);

// Functional-style version
GEODE_CORE_EXPORT Tuple<Array<real>,Array<Vector<real,3>>,Array<int>,Array<Vector<real,3>>>
evaluate_surface_levelset(const ParticleTree<Vector<real,3>>& particles,
                          const SimplexTree<Vector<real,3>,2>& surface,
                          real max_distance, bool compute_signs);

}
