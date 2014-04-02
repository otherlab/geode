//#####################################################################
// Class ConstitutiveModel
//##################################################################### 
#include <geode/force/ConstitutiveModel.h>
#include <geode/force/DiagonalizedIsotropicStressDerivative.h>
#include <geode/vector/DiagonalMatrix.h>
namespace geode {

typedef real T;

template<class T,int d> ConstitutiveModel<T,d>::ConstitutiveModel(T failure_threshold) 
  : failure_threshold(failure_threshold) {}

template<class T,int d> ConstitutiveModel<T,d>::~ConstitutiveModel() {}

template<class T,int d> DiagonalMatrix<T,d> ConstitutiveModel<T,d>::clamp_f(const DiagonalMatrix<T,d>& F) const {
  return F.clamp_min(failure_threshold);
}

template class ConstitutiveModel<T,2>;
template class ConstitutiveModel<T,3>;

}
