//#####################################################################
// Class Gravity
//#####################################################################
#pragma once

#include <geode/array/Array.h>
#include <geode/force/Force.h>
#include <geode/vector/Vector.h>
namespace geode {

template<class TV>
class Gravity : public Force<TV> {
public:
  GEODE_DECLARE_TYPE(GEODE_CORE_EXPORT)
  typedef Force<TV> Base;
  typedef typename TV::Scalar T;
  enum {m=TV::m};

  TV gravity;
private:
  Array<const T> mass;
  Array<const TV> X;
protected:
  Gravity(Array<const T> mass);
public:
  ~Gravity();

  void update_position(Array<const TV> X,bool definite);
  void add_frequency_squared(RawArray<T> frequency_squared) const;
  T elastic_energy() const;
  void add_elastic_force(RawArray<TV> F) const;
  void add_elastic_differential(RawArray<TV> dF,RawArray<const TV> dX) const;
  void add_elastic_gradient_block_diagonal(RawArray<SymmetricMatrix<T,m> > dFdX) const;
  T damping_energy(RawArray<const TV> V) const;
  void add_damping_force(RawArray<TV> F,RawArray<const TV> V) const;
  T strain_rate(RawArray<const TV> V) const;

  int nodes() const;
  void structure(SolidMatrixStructure& structure) const;
  void add_elastic_gradient(SolidMatrix<TV>& matrix) const;
  void add_damping_gradient(SolidMatrix<TV>& matrix) const;
};
}
