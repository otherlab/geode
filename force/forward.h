//#####################################################################
// Header Forces/Forward
//#####################################################################
#pragma once

#include <other/core/vector/forward.h>
namespace other{

template<class TV> class Springs;
template<class TV,int d> class FiniteVolume;

template<class T,int d> class ConstitutiveModel;
template<class T,int d> class IsotropicConstitutiveModel;
template<class T,int d> class AnisotropicConstitutiveModel;
template<class T,int d> class DiagonalizedStressDerivative;
template<class T,int m,int d=m> class DiagonalizedIsotropicStressDerivative;
template<class T,int d> class PlasticityModel;

}
