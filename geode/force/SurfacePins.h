//#####################################################################
// Class SurfacePins
//#####################################################################
//
// Linear spring force between particles and their closest corresponding points on a surface.
// The stiffness is mass proportional for resolution invariance.
//
//#####################################################################
#pragma once

#include <geode/array/Array.h>
#include <geode/force/Force.h>
#include <geode/geometry/surface_levelset.h>
#include <geode/mesh/TriangleSoup.h>
#include <geode/vector/Vector.h>
namespace geode {

class SurfacePins : public Force<Vector<real,3>> {
public:
  GEODE_DECLARE_TYPE(GEODE_CORE_EXPORT)
  typedef real T;
  typedef Vector<T,3> TV;
  typedef Force<TV> Base;
  enum {m=TV::m};

  const Array<const int> particles;
  Ref<TriangleSoup> target_mesh;
  const Array<const TV> target_X;
private:
  int max_node;
  const Array<const T> mass;
  const Array<T> k,kd;
  const Array<TV> node_X;
  Ptr<ParticleTree<TV>> node_tree;
  const Ref<SimplexTree<TV,2>> target_tree;
  const Array<CloseInfo<2>> info;
protected:
  SurfacePins(Array<const int> particles, Array<const T> mass, TriangleSoup& target_mesh, Array<const TV> target_X, NdArray<const T> stiffness, NdArray<const T> damping_ratio);
public:
  ~SurfacePins();

  Array<TV> closest_points(Array<const TV> X);

  void update_position(Array<const TV> X, bool definite);
  void add_frequency_squared(RawArray<T> frequency_squared) const;
  T elastic_energy() const;
  void add_elastic_force(RawArray<TV> F) const;
  void add_elastic_differential(RawArray<TV> dF, RawArray<const TV> dX) const;
  void add_elastic_gradient_block_diagonal(RawArray<SymmetricMatrix<T,m>> dFdX) const;
  T damping_energy(RawArray<const TV> V) const;
  void add_damping_force(RawArray<TV> F, RawArray<const TV> V) const;
  T strain_rate(RawArray<const TV> V) const;

  int nodes() const;
  void structure(SolidMatrixStructure& structure) const;
  void add_elastic_gradient(SolidMatrix<TV>& matrix) const;
  void add_damping_gradient(SolidMatrix<TV>& matrix) const;
};

}
