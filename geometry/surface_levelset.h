//#####################################################################
// Evaluate signed distances between a point cloud and a mesh
//#####################################################################
#pragma once

#include <other/core/array/forward.h>
#include <other/core/geometry/forward.h>
#include <other/core/vector/Vector3d.h>
namespace other {

struct CloseTriangleInfo {
  real phi;
  Vector<real,3> normal;
  int triangle;
  Vector<real,3> weights;
};

OTHER_CORE_EXPORT void evaluate_surface_levelset(const ParticleTree<Vector<real,3>>& particles, const SimplexTree<Vector<real,3>,2>& surface, RawArray<CloseTriangleInfo> info, real max_distance, bool compute_signs);

}
