//#####################################################################
// Class LinearFiniteVolumeHex
//#####################################################################
#pragma once

#include <other/core/force/Force.h>
#include <other/core/force/StrainMeasureHex.h>
namespace other{

class LinearFiniteVolumeHex : public Force<Vector<real,3>>
{
  typedef real T;
  typedef Vector<T,3> TV;
  enum {m=3};
public:
  OTHER_DECLARE_TYPE(OTHER_CORE_EXPORT)
  typedef Force<TV> Base;

  const Ref<const StrainMeasureHex> strain;
  T youngs_modulus;
  T poissons_ratio;
  T rayleigh_coefficient;
  const T density;
private:
  Array<const TV> X;

protected:
  LinearFiniteVolumeHex(const StrainMeasureHex& strain, const T density, const T youngs_modulus, const T poissons_ratio, const T rayleigh_coefficient);
public:
  virtual ~LinearFiniteVolumeHex();

  Vector<T,2> mu_lambda() const;

  void update_position(Array<const TV> X, bool definite);
  T elastic_energy() const;
  void add_elastic_force(RawArray<TV> F) const;
  void add_elastic_differential(RawArray<TV> dF, RawArray<const TV> dX) const;
  void add_elastic_gradient_block_diagonal(RawArray<SymmetricMatrix<T,m>> dFdX) const;
  T damping_energy(RawArray<const TV> V) const;
  void add_damping_force(RawArray<TV> F, RawArray<const TV> V) const;
  void add_frequency_squared(RawArray<T> frequency_squared) const;
  T strain_rate(RawArray<const TV> V) const;

  int nodes() const;
  void structure(SolidMatrixStructure& structure) const;
  void add_elastic_gradient(SolidMatrix<TV>& matrix) const;
  void add_damping_gradient(SolidMatrix<TV>& matrix) const;
private:
  void add_differential_helper(RawArray<TV> dF, RawArray<const TV> dX, T scale) const;
};

}
