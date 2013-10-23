// Specialized finite volume model for in-plane, anisotropic shell forces.
#pragma once

// Given a 3x2 deformation gradient F with polar decomposition F = Q Fh,
// our energy function has the form
//
//   E = E_00(Fh_00) + E_01(Fh_01) + E_11(Fh_11)
// 
// In cloth terminology, we independently penalize weft, warp, and shear deformation.
// For now, we ignore damping forces.
//
// Since SimpleShell is natively anisotropic, Dm cannot be QR-decomposed, and we can't
// use StrainMeasure<T,2> directly.

#include <geode/force/Force.h>
#include <geode/mesh/TriangleMesh.h>
#include <geode/vector/Matrix.h>
namespace geode {

class SimpleShell : public Force<Vector<real,3>> {
  typedef real T;
  typedef Vector<T,3> TV;
  typedef SymmetricMatrix<T,2> SM2;
public:
  GEODE_DECLARE_TYPE(GEODE_CORE_EXPORT)
  typedef Force<TV> Base;

  const T density;
  Vector<T,2> stretch_stiffness;
  T shear_stiffness;
  T F_threshold; // Threshold to prevent Hessian blowup for small F
protected:
  const int nodes_;
  Array<const TV> X;
  bool definite_;
  struct Info {
    // Rest state
    Vector<int,3> nodes;
    Matrix<T,2> inv_Dm;
    T scale; // -volume
    // Current state
    Matrix<T,3> Q; // Rotation from polar decomposition
    SM2 Fh; // F = Q Fh
    // Differential information
    Vector<T,4> H_planar; // Component of negative Hessian due to existing forces rotating in plane (meaning depends on definite flag)
    SM2 H_nonplanar; // Component of negative Hessian due to existing forces rotating out of the plane
    T c0,c1; // Constants for 4x4 in-plane block due to DPhs
  };
  Array<Info> info;

  SimpleShell(const TriangleMesh& mesh, RawArray<const Matrix<T,2>> Dm, const T density);
public:
  virtual ~SimpleShell();

  void update_position(Array<const TV> X, bool definite);
  T elastic_energy() const;
  void add_elastic_force(RawArray<TV> F) const;
  void add_elastic_differential(RawArray<TV> dF, RawArray<const TV> dX) const;
  void add_elastic_gradient_block_diagonal(RawArray<SymmetricMatrix<T,3>> dFdX) const;
  T damping_energy(RawArray<const TV> V) const;
  void add_damping_force(RawArray<TV> F, RawArray<const TV> V) const;
  void add_frequency_squared(RawArray<T> frequency_squared) const;
  T strain_rate(RawArray<const TV> V) const;

  int nodes() const;
  void structure(SolidMatrixStructure& structure) const;
  void add_elastic_gradient(SolidMatrix<TV>& matrix) const;
  void add_damping_gradient(SolidMatrix<TV>& matrix) const;

private:
  SM2 stiffness() const;
  SM2 simple_P(const Info& I) const; // Stress pretending that F stays symmetric 2x2
  template<bool definite> SM2 simple_DP(const Info& I) const; // Stress derivative pretending that F stays symmetric 2x2
  template<bool definite> Matrix<T,3,2> force_differential(const Info& I, const Matrix<T,3,2>& dDs) const;
  template<bool definite> void update_position_helper();
  template<bool definite> void add_elastic_gradient_helper(SolidMatrix<TV>& matrix) const;
};

}
