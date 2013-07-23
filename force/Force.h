//#####################################################################
// Class Force
//#####################################################################
//
// Abstract base class for solids forces
//
//#####################################################################
#pragma once

#include <other/core/array/RawArray.h>
#include <other/core/python/Object.h>
namespace other {

template<class TV>
class Force : public Object {
public:
  OTHER_DECLARE_TYPE(OTHER_CORE_EXPORT)
  typedef Object Base;
  typedef typename TV::Scalar T;
  static const int d = TV::m;

protected:
  OTHER_CORE_EXPORT Force();
public:
  OTHER_CORE_EXPORT ~Force();

  virtual void update_position(Array<const TV> X, bool definite) = 0;
  virtual T elastic_energy() const = 0;
  virtual void add_elastic_force(RawArray<TV> F) const = 0;
  virtual void add_elastic_differential(RawArray<TV> dF, RawArray<const TV> dX) const = 0;
  virtual void add_elastic_gradient_block_diagonal(RawArray<SymmetricMatrix<T,d>> dFdX) const = 0;
  virtual T damping_energy(RawArray<const TV> V) const = 0;
  virtual void add_damping_force(RawArray<TV> F, RawArray<const TV> V) const = 0;
  virtual void add_frequency_squared(RawArray<T> frequency_squared) const = 0;
  virtual T strain_rate(RawArray<const TV> V) const = 0;

  virtual int nodes() const = 0; // Minimum number of nodes
  virtual void structure(SolidMatrixStructure& structure) const = 0;
  virtual void add_elastic_gradient(SolidMatrix<TV>& matrix) const = 0;
  virtual void add_damping_gradient(SolidMatrix<TV>& matrix) const = 0;

  // For testing purposes
  Array<TV> elastic_gradient_block_diagonal_times(RawArray<TV> dX) const;
};

}
