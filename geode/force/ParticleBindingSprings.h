//#####################################################################
// Class ParticleBindingSprings
//#####################################################################
//
// Zero length spring force between particles.
// The parameters are mass proportional for resolution invariance.
//
//#####################################################################
#pragma once

#include <geode/array/Array.h>
#include <geode/force/Force.h>
#include <geode/vector/Vector.h>
namespace geode {

struct ParticleBindingInfo {
  Vector<int,2> nodes;
  real k,kd;
};

class ParticleBindingSprings : public Force<Vector<real,3>> {
public:
  GEODE_DECLARE_TYPE(GEODE_CORE_EXPORT)
  typedef real T;
  typedef Vector<T,3> TV;
  typedef Force<TV> Base;
  enum {m=TV::m};

private:
  const Array<const T> mass;
  Array<ParticleBindingInfo> info;
  Array<const TV> X;
protected:
  ParticleBindingSprings(Array<const Vector<int,2>> nodes, Array<const T> mass, NdArray<const T> stiffness, NdArray<const T> damping_ratio);
public:
  ~ParticleBindingSprings();

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
