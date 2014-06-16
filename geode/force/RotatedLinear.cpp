//#####################################################################
// Class RotatedLinear
//#####################################################################
#include <geode/force/IsotropicConstitutiveModel.h>
#include <geode/python/Class.h>
#include <geode/vector/DiagonalMatrix.h>
#include <geode/vector/Matrix.h>
#include <geode/vector/SymmetricMatrix.h>
namespace geode {

template<class T,int d> class RotatedLinear : public IsotropicConstitutiveModel<T,d> {
  typedef Vector<T,d> TV;
public:
  GEODE_DECLARE_TYPE(GEODE_CORE_EXPORT)
  typedef IsotropicConstitutiveModel<T,d> Base;
  using Base::lambda;using Base::mu;using Base::alpha;using Base::beta;

  NdArray<const T> youngs_modulus,poissons_ratio;

protected:
  RotatedLinear(NdArray<const T> youngs_modulus, NdArray<const T> poissons_ratio=(T).475, NdArray<const T> rayleigh_coefficient=(T).05)
    : youngs_modulus(youngs_modulus), poissons_ratio(poissons_ratio) {
    Array<const int> shape = youngs_modulus.shape;
    GEODE_ASSERT(shape.size()<=1 && shape==poissons_ratio.shape && shape==rayleigh_coefficient.shape);
    lambda = NdArray<T>(shape,uninit);
    mu = NdArray<T>(shape,uninit);
    alpha = NdArray<T>(shape,uninit);
    beta = NdArray<T>(shape,uninit);
    for (int i=0;i<youngs_modulus.flat.size();i++) {
      GEODE_ASSERT(poissons_ratio.flat[i]>-1 && poissons_ratio.flat[i]<(T).5);
      lambda.flat[i] = youngs_modulus.flat[i]*poissons_ratio.flat[i]/((1+poissons_ratio.flat[i])*(1-2*poissons_ratio.flat[i]));
      mu.flat[i] = youngs_modulus.flat[i]/(2*(1+poissons_ratio.flat[i]));
      alpha.flat[i] = rayleigh_coefficient.flat[i]*lambda.flat[i];
      beta.flat[i] = rayleigh_coefficient.flat[i]*mu.flat[i];
    }
  }
public:

  T elastic_energy(const DiagonalMatrix<T,d>& F, const int simplex) const {
    DiagonalMatrix<T,d> strain = F-1;
    if (!mu.rank())
      return mu()*strain.sqr_frobenius_norm()+(T).5*lambda()*sqr(strain.trace());
    else
      return mu[simplex]*strain.sqr_frobenius_norm()+(T).5*lambda[simplex]*sqr(strain.trace());
  }

  DiagonalMatrix<T,d> P_From_Strain(const DiagonalMatrix<T,d>& F, const T scale, const int simplex) const {
    DiagonalMatrix<T,d> strain = F-1;
    if (!mu.rank())
      return 2*scale*mu()*strain+scale*lambda()*strain.trace();
    else
      return 2*scale*mu[simplex]*strain+scale*lambda[simplex]*strain.trace();
  }

  T damping_energy(const DiagonalMatrix<T,d>& F,const Matrix<T,d>& F_dot, const int simplex) const {
    SymmetricMatrix<T,d> strain_rate = symmetric_part(F_dot);
    if (!beta.rank())
      return beta()*strain_rate.sqr_frobenius_norm()+(T).5*alpha()*sqr(strain_rate.trace());
    else
      return beta[simplex]*strain_rate.sqr_frobenius_norm()+(T).5*alpha[simplex]*sqr(strain_rate.trace());
  }

  Matrix<T,d> P_From_Strain_Rate(const DiagonalMatrix<T,d>& F, const Matrix<T,d>& F_dot, const T scale, const int simplex) const {
    SymmetricMatrix<T,d> strain_rate = symmetric_part(F_dot);
    if (!beta.rank())
      return 2*scale*beta()*strain_rate+scale*alpha()*strain_rate.trace();
    else
      return 2*scale*beta[simplex]*strain_rate+scale*alpha[simplex]*strain_rate.trace();
  }
};

typedef real T;
template<> GEODE_DEFINE_TYPE(RotatedLinear<T,2>)
template<> GEODE_DEFINE_TYPE(RotatedLinear<T,3>)

}
using namespace geode;

void wrap_rotated_linear() {
  {typedef RotatedLinear<T,2> Self;
  Class<Self>("RotatedLinear2d")
    .GEODE_INIT(NdArray<const T>,NdArray<const T>,NdArray<const T>)
    ;}

  {typedef RotatedLinear<T,3> Self;
  Class<Self>("RotatedLinear3d")
    .GEODE_INIT(NdArray<const T>,NdArray<const T>,NdArray<const T>)
    ;}
}
