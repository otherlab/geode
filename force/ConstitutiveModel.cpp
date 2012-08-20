//#####################################################################
// Class ConstitutiveModel
//##################################################################### 
#include <other/core/force/ConstitutiveModel.h>
#include <other/core/force/DiagonalizedIsotropicStressDerivative.h>
#include <other/core/python/Class.h>
#include <other/core/vector/DiagonalMatrix.h>
namespace other{

typedef real T;
template<> OTHER_DEFINE_TYPE(ConstitutiveModel<T,2>)
template<> OTHER_DEFINE_TYPE(ConstitutiveModel<T,3>)

template<class T,int d> ConstitutiveModel<T,d>::
ConstitutiveModel(T failure_threshold)
    :failure_threshold(failure_threshold)
{}

template<class T,int d> ConstitutiveModel<T,d>::
~ConstitutiveModel()
{}

template<class T,int d> DiagonalMatrix<T,d> ConstitutiveModel<T,d>::
clamp_f(const DiagonalMatrix<T,d>& F) const
{
    return F.clamp_min(failure_threshold);
}

template class ConstitutiveModel<T,2>;
template class ConstitutiveModel<T,3>;

}

void wrap_constitutive_model()
{
    using namespace other;

    {typedef ConstitutiveModel<T,2> Self;
    Class<Self>("ConstitutiveModel2d");}

    {typedef ConstitutiveModel<T,3> Self;
    Class<Self>("ConstitutiveModel3d");}
}
