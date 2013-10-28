//#####################################################################
// Class AnisotropicConstitutiveModel
//##################################################################### 
#pragma once

#include <geode/force/ConstitutiveModel.h>
#include <geode/force/DiagonalizedStressDerivative.h>
namespace geode {

template<class T,int d> class AnisotropicConstitutiveModel : public ConstitutiveModel<T,d> {
public:
  typedef ConstitutiveModel<T,d> Base;

protected:
  AnisotropicConstitutiveModel(T failure_threshold=.1)
    : Base(failure_threshold) {}
public:

  bool use_isotropic_stress_derivative() const {
    return false;
  }

  virtual T elastic_energy(const DiagonalMatrix<T,d>& F,const Matrix<T,d>& V,const int simplex) const=0;
  virtual Matrix<T,d> P_From_Strain(const DiagonalMatrix<T,d>& F,const Matrix<T,d>& V,const T scale,const int simplex) const=0;
  virtual DiagonalizedStressDerivative<T,d> stress_derivative(const DiagonalMatrix<T,d>& F,const Matrix<T,d>& V,const int simplex) const=0;
  virtual void update_position(const DiagonalMatrix<T,d>& F,const Matrix<T,d>& V,const int simplex){}
};

}
