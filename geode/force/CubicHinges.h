// Cubic hinges based on Garg et al. 2007
#pragma once

#include <geode/force/Force.h>
#include <geode/mesh/forward.h>
#include <geode/vector/forward.h>
namespace geode {

// For details, see
//   flat: Wardetzky et al., "Discrete Quadratic Curvature energies", Computer Aided Geometric design, 2007.
//   general: Garg et al., "Cubic shells", SCA 2007.

template<class TV_> class CubicHinges : public Force<TV_> {
  typedef TV_ TV;
  BOOST_MPL_ASSERT((is_same<typename TV::Scalar,real>));
public:
  GEODE_DECLARE_TYPE(GEODE_CORE_EXPORT)
  typedef Force<TV> Base;
  enum Workaround {d=TV::m-1}; // Topological mesh dimension
  typedef real T;

  struct Info {
    T base; // constant energy term
    T dot; // scale on the dot product energy term (quadratic in 2D and 3D)
    Vector<T,d+2> c; // coefficients of the linear form making up the quadratic energy
    T det; // scale on the determinant energy term (quadratic in 2D, cubic in 3D)
  };

  // Bends around the middle point for segments, or the middle two for triangles
  const Array<const Vector<int,d+2>> bends;
  T stiffness, damping;
  bool simple_hessian;
private:
  const int nodes_;
  const Array<Info> info;
  Array<const TV> X;

protected:
  CubicHinges(Array<const Vector<int,d+2>> bends, RawArray<const T> angles, RawArray<const TV> X);
public:
  ~CubicHinges();

  static Array<T> angles(RawArray<const Vector<int,d+2>> bends, RawArray<const TV> X);

  void update_position(Array<const TV> X, bool definite);
  void add_frequency_squared(RawArray<T> frequency_squared) const;
  T strain_rate(RawArray<const TV> V) const;
  T elastic_energy() const;
  void add_elastic_force(RawArray<TV> F) const;
  void add_elastic_differential(RawArray<TV> dF, RawArray<const TV> dX) const;
  void add_elastic_gradient_block_diagonal(RawArray<SymmetricMatrix<T,d+1>> dFdX) const;
  T damping_energy(RawArray<const TV> V) const;
  void add_damping_force(RawArray<TV> F, RawArray<const TV> V) const;

  int nodes() const;
  void structure(SolidMatrixStructure& structure) const;
  void add_elastic_gradient(SolidMatrix<TV>& matrix) const;
  void add_damping_gradient(SolidMatrix<TV>& matrix) const;

  // For testing purposes: equal to elastic_energy only for isometric deformations and nice rest triangles
  T slow_elastic_energy(RawArray<const T> angles, RawArray<const TV> restX, RawArray<const TV> X) const;
};

}
