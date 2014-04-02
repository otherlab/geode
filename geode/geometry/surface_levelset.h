//#####################################################################
// Evaluate signed distances between a point cloud and a mesh
//#####################################################################
#pragma once

#include <geode/array/Array.h>
#include <geode/geometry/SimplexTree.h>
#include <geode/math/constants.h>
#include <geode/structure/Tuple.h>
#include <geode/vector/Vector3d.h>
namespace geode {

template<int d> struct CloseInfo {
  static_assert(d==1 || d==2,"");
  typedef Vector<real,3> TV;

  real phi;
  TV normal;
  int simplex;
  typename SimplexTree<TV,d>::Weights weights;
};

template<int d> GEODE_CORE_EXPORT void surface_levelset(const ParticleTree<Vector<real,3>>& particles,
                                                        const SimplexTree<Vector<real,3>,d>& surface,
                                                        RawArray<typename Hide<CloseInfo<d>>::type> info,
                                                        const real max_distance=inf, const bool compute_signs=true);

// Functional-style version: returns distance, normals, closest simplex, and barycentric weights per point.
template<int d> GEODE_CORE_EXPORT Tuple<Array<real>,Array<Vector<real,3>>,
                                        Array<int>,Array<typename SimplexTree<Vector<real,3>,d>::Weights>>
surface_levelset(const ParticleTree<Vector<real,3>>& particles, const SimplexTree<Vector<real,3>,d>& surface,
                 const real max_distance=inf, const bool compute_signs=true);

// For testing purposes
GEODE_CORE_EXPORT Tuple<Array<real>,Array<Vector<real,3>>,Array<int>,Array<Vector<real,3>>>
  slow_surface_levelset(const ParticleTree<Vector<real,3>>& particles,
                        const SimplexTree<Vector<real,3>,2>& surface);

}
