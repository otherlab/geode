//#####################################################################
// Class BindingSprings
//#####################################################################
//
// Zero length spring force between particles and linear combinations of other particles.
// The parameters are mass proportional for resolution invariance.
//
//#####################################################################
#pragma once

#include <other/core/array/Array.h>
#include <other/core/force/Force.h>
#include <other/core/vector/Vector.h>
namespace other {

template<int k_> struct BindingInfo {
  typedef real T;

  int node;
  Vector<int,k_> parents;
  Vector<T,k_> weights;
  T k,kd;
};

template<int k> class BindingSprings : public Force<Vector<real,3>> {
public:
  OTHER_DECLARE_TYPE(OTHER_CORE_EXPORT)
  typedef real T;
  typedef Vector<T,3> TV;
  typedef Force<TV> Base;
  enum {m=TV::m};

private:
  const Array<const T> mass;
  Array<BindingInfo<k>> info;
  Array<const TV> X;
protected:
  BindingSprings(Array<const int> nodes, Array<const Vector<int,k>> parents, Array<const Vector<T,k>> weights, Array<const T> mass, NdArray<const T> stiffness, NdArray<const T> damping_ratio);
public:
  ~BindingSprings();

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
