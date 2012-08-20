//#####################################################################
// Class NeoHookean
//#####################################################################
#include <other/core/force/IsotropicConstitutiveModel.h>
#include <other/core/force/DiagonalizedIsotropicStressDerivative.h>
#include <other/core/math/cube.h>
#include <other/core/math/pow.h>
#include <other/core/python/Class.h>
#include <other/core/utility/Log.h>
#include <other/core/vector/Matrix.h>
#include <other/core/vector/DiagonalMatrix.h>
#include <other/core/vector/SymmetricMatrix.h>
namespace other{

using ::std::log;
using Log::cout;
using std::endl;

template<class T,int d>
class NeoHookean:public IsotropicConstitutiveModel<T,d>
{
public:
    OTHER_DECLARE_TYPE
    typedef IsotropicConstitutiveModel<T,d> Base;
    using Base::lambda;using Base::mu;using Base::alpha;using Base::beta;
    using Base::failure_threshold;

    T youngs_modulus,poissons_ratio;
private:
    T dth_root_failure_threshold;

protected:
    NeoHookean(const T youngs_modulus=3e6,const T poissons_ratio=.475,const T rayleigh_coefficient=.05,const T failure_threshold=.1)
        :Base(failure_threshold),youngs_modulus(youngs_modulus),poissons_ratio(poissons_ratio)
    {
        OTHER_ASSERT(poissons_ratio>-1&&poissons_ratio<.5);
        lambda()=youngs_modulus*poissons_ratio/((1+poissons_ratio)*(1-2*poissons_ratio));
        mu()=youngs_modulus/(2*(1+poissons_ratio));
        alpha()=rayleigh_coefficient*lambda();
        beta()=rayleigh_coefficient*mu();
        dth_root_failure_threshold=pow<1,d>(failure_threshold);
    }
public:

private:
    DiagonalMatrix<T,2> clamp_to_hyperbola(const DiagonalMatrix<T,2>& F) const
    {if(sqr(F.x00)>failure_threshold) return DiagonalMatrix<T,2>(F.x00,failure_threshold/F.x00);
    else return DiagonalMatrix<T,2>(dth_root_failure_threshold,dth_root_failure_threshold);}

    DiagonalMatrix<T,3> clamp_to_hyperbola(const DiagonalMatrix<T,3>& F) const
    {if(F.x00*F.x11*F.x11>failure_threshold) return DiagonalMatrix<T,3>(F.x00,F.x11,failure_threshold/(F.x00*F.x11));
    else if(cube(F.x00)>failure_threshold){T clamped=sqrt(failure_threshold/F.x00);return DiagonalMatrix<T,3>(F.x00,clamped,clamped);}
    else return DiagonalMatrix<T,3>(dth_root_failure_threshold,dth_root_failure_threshold,dth_root_failure_threshold);}
public:

    T elastic_energy(const DiagonalMatrix<T,d>& F,const int simplex) const
    {DiagonalMatrix<T,d> F_;T J=F.determinant(),J_;
    if(J>=failure_threshold){F_=F;J_=J;}
    else{F_=clamp_to_hyperbola(F);J_=failure_threshold;}
    T log_J=log(J_);
    return (T).5*mu()*F_.sqr_frobenius_norm()+((T).5*lambda()*log_J-mu())*log_J;}

    // clamp to hyperbola to avoid indefiniteness "automatically"
    DiagonalMatrix<T,d> P_From_Strain(const DiagonalMatrix<T,d>& F,const T scale,const int simplex) const
    {T scale_mu=scale*mu(),scale_lambda=scale*lambda(),J=F.determinant();
    if(J>=failure_threshold) return scale_mu*F-(scale_mu-scale_lambda*log(J))*F.inverse();
    DiagonalMatrix<T,d> F_clamp=clamp_to_hyperbola(F),dF=F-F_clamp,F_inverse=F_clamp.inverse();
    T scale_mu_minus_lambda_log_J=scale_mu-scale_lambda*log(failure_threshold);
    return scale_mu*F+scale_mu_minus_lambda_log_J*(sqr(F_inverse)*dF-F_inverse)+scale_lambda*inner_product(F_inverse,dF)*F_inverse;}

    T damping_energy(const DiagonalMatrix<T,d>& F,const Matrix<T,d>& F_dot,const int simplex) const
    {SymmetricMatrix<T,d> strain_rate=symmetric_part(F_dot);
    return beta()*strain_rate.sqr_frobenius_norm()+(T).5*alpha()*sqr(strain_rate.trace());}

    Matrix<T,d> P_From_Strain_Rate(const DiagonalMatrix<T,d>& F,const Matrix<T,d>& F_dot,const T scale,const int simplex) const
    {SymmetricMatrix<T,d> strain_rate=symmetric_part(F_dot); // use linear damping because of problems with inverting elements...
    return 2*scale*beta()*strain_rate+scale*alpha()*strain_rate.trace();}

    DiagonalizedIsotropicStressDerivative<T,2> isotropic_stress_derivative(const DiagonalMatrix<T,2>& F,const int triangle) const
    {DiagonalMatrix<T,2> F_inverse=F.clamp_min(failure_threshold).inverse();
    T mu_minus_lambda_logJ=mu()+lambda()*log(F_inverse.determinant());
    SymmetricMatrix<T,2> F_inverse_outer=outer_product(F_inverse.To_Vector());
    DiagonalizedIsotropicStressDerivative<T,2> dP_dF;
    dP_dF.x0000=mu()+(lambda()+mu_minus_lambda_logJ)*F_inverse_outer.x00;//alpha+beta+gamma
    dP_dF.x1111=mu()+(lambda()+mu_minus_lambda_logJ)*F_inverse_outer.x11;
    dP_dF.x1100=lambda()*F_inverse_outer.x10; //gamma
    dP_dF.x1010=mu(); //alpha
    dP_dF.x1001=mu_minus_lambda_logJ*F_inverse_outer.x10; //beta
    return dP_dF;}

    DiagonalizedIsotropicStressDerivative<T,3> isotropic_stress_derivative(const DiagonalMatrix<T,3>& F,const int tetrahedron) const
    {DiagonalMatrix<T,3> F_inverse=F.clamp_min(failure_threshold).inverse();
    T mu_minus_lambda_logJ=mu()+lambda()*log(F_inverse.determinant());
    SymmetricMatrix<T,3> F_inverse_outer=outer_product(F_inverse.To_Vector());
    DiagonalizedIsotropicStressDerivative<T,3> dP_dF;
    dP_dF.x0000=mu()+(lambda()+mu_minus_lambda_logJ)*F_inverse_outer.x00;
    dP_dF.x1111=mu()+(lambda()+mu_minus_lambda_logJ)*F_inverse_outer.x11;
    dP_dF.x2222=mu()+(lambda()+mu_minus_lambda_logJ)*F_inverse_outer.x22;
    dP_dF.x1100=lambda()*F_inverse_outer.x10;
    dP_dF.x2200=lambda()*F_inverse_outer.x20;
    dP_dF.x2211=lambda()*F_inverse_outer.x21;
    dP_dF.x1010=dP_dF.x2020=dP_dF.x2121=mu();
    dP_dF.x1001=mu_minus_lambda_logJ*F_inverse_outer.x10;
    dP_dF.x2002=mu_minus_lambda_logJ*F_inverse_outer.x20;
    dP_dF.x2112=mu_minus_lambda_logJ*F_inverse_outer.x21;
    return dP_dF;}
};

typedef real T;
template<> OTHER_DEFINE_TYPE(NeoHookean<T,2>)
template<> OTHER_DEFINE_TYPE(NeoHookean<T,3>)

}
using namespace other;

void wrap_neo_hookean()
{
    {typedef NeoHookean<T,2> Self;
    Class<Self>("NeoHookean2d") 
        .OTHER_INIT(T,T,T,T)
        ;}

    {typedef NeoHookean<T,3> Self;
    Class<Self>("NeoHookean3d") 
        .OTHER_INIT(T,T,T,T)
        ;}
}
