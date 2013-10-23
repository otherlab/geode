//#####################################################################
// Class AxisPins
//#####################################################################
//
// Linear spring force between particles and fixed world space locations.
// The stiffness is mass proportional for resolution invariance.
//
//#####################################################################
#pragma once

#include <geode/array/Array.h>
#include <geode/force/Force.h>
#include <geode/vector/Vector.h>
namespace geode {

class AxisPins : public Force<Vector<real,3>> {
public:
  GEODE_DECLARE_TYPE(GEODE_CORE_EXPORT)
  typedef real T;
  typedef Vector<T,3> TV;
  typedef Force<TV> Base;
  enum {m=TV::m};

  const Array<const int> particles;
  const Array<const TV> targets;
  TV axis;
private:
  int max_node;
  const Array<const T> mass;
  const Array<T> k,kd;
  Array<const TV> X;
protected:
  AxisPins(Array<const int> particles, Array<const T> mass, Array<const TV> targets, NdArray<const T> stiffness, NdArray<const T> damping_ratio);
public:
  ~AxisPins();

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
