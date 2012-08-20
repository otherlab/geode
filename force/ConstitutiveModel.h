//#####################################################################
// Class ConstitutiveModel
//##################################################################### 
#pragma once

#include <other/core/array/NdArray.h>
#include <other/core/force/forward.h>
#include <other/core/force/DiagonalizedIsotropicStressDerivative.h>
#include <other/core/python/Object.h>
#include <other/core/utility/debug.h>
namespace other{

template<class T,int d>
class ConstitutiveModel:public Object
{
public:
    OTHER_DECLARE_TYPE
    typedef Object Base;

    NdArray<T> lambda,mu; // constant or spatially varying Lame coefficients (used by almost all derived models)
    NdArray<T> alpha,beta; // constant or spatially varying damping parameters (used by all current derived models)
    T failure_threshold; // declared here so that FiniteVolume can use it for S3d out-of-plane derivatives

private:
    ConstitutiveModel(T failure_threshold=.1);

    // all constitutive models should derive from one of these
    friend class IsotropicConstitutiveModel<T,d>;
    friend class AnisotropicConstitutiveModel<T,d>;
public:

    virtual ~ConstitutiveModel();

    virtual T maximum_elastic_stiffness(const int simplex) const // for elastic CFL computation
    {return lambda.rank()?lambda(simplex)+2*mu(simplex):lambda()+2*mu();}

    virtual T maximum_damping_stiffness(const int simplex) const // for damping CFL computation
    {return alpha.rank()?alpha(simplex)+2*beta(simplex):alpha()+2*beta();}

    virtual DiagonalMatrix<T,d> clamp_f(const DiagonalMatrix<T,d>& F) const;
    virtual T damping_energy(const DiagonalMatrix<T,d>& F,const Matrix<T,d>& F_dot,const int simplex) const=0;
    virtual Matrix<T,d> P_From_Strain_Rate(const DiagonalMatrix<T,d>& F,const Matrix<T,d>& F_dot,const T scale,const int simplex) const=0;
    virtual DiagonalizedIsotropicStressDerivative<T,d,d> isotropic_stress_derivative(const DiagonalMatrix<T,d>& F,const int simplex) const {OTHER_FUNCTION_IS_NOT_DEFINED();}
};
}
