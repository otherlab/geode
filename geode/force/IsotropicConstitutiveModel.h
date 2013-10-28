//#####################################################################
// Class IsotropicConstitutiveModel
//#####################################################################
#pragma once

#include <geode/force/ConstitutiveModel.h>
namespace geode {

template<class T,int d> class IsotropicConstitutiveModel : public ConstitutiveModel<T,d> {
public:
  typedef ConstitutiveModel<T,d> Base;

protected:
  IsotropicConstitutiveModel(T failure_threshold=.1)
    : Base(failure_threshold) {}
public:

  virtual T elastic_energy(const DiagonalMatrix<T,d>& F,const int simplex) const=0;
  virtual DiagonalMatrix<T,d> P_From_Strain(const DiagonalMatrix<T,d>& F,const T scale,const int simplex) const=0;
  virtual void update_position(const DiagonalMatrix<T,d>& F,const int simplex){}
};

}
