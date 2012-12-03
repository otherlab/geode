//#####################################################################
// Class DiagonalizedStressDerivative
//##################################################################### 
#pragma once

#include <other/core/vector/forward.h>
#include <other/core/utility/debug.h>
namespace other{

template<class T,int d> class DiagonalizedStressDerivative;
using ::std::sqrt;

template<class T>
class DiagonalizedStressDerivative<T,2>
{
    typedef Vector<T,2> TV;
public:
    DiagonalMatrix<T,2> F;
    SymmetricMatrix<T,2> S;
    SymmetricMatrix<T,3> dSdC;

    Matrix<T,2> differential(const Matrix<T,2>& dF) const
    {SymmetricMatrix<T,2> dC = twice_symmetric_part(F*dF);
    Vector<T,3> ds=dSdC*Vector<T,3>(dC.x00,dC.x11,(T)sqrt(2.)*dC.x10);
    SymmetricMatrix<T,2> dS(ds.x,T(1/sqrt(2.))*ds.z,ds.y);
    return dF*S+F*dS;}

    void enforce_definiteness()
    {S=S.positive_definite_part();dSdC=dSdC.positive_definite_part();}
};

template<class T>
class DiagonalizedStressDerivative<T,3>
{
    typedef Vector<T,3> TV;
public:
    DiagonalMatrix<T,3> F;
    SymmetricMatrix<T,3> S,dSdC_d,dSdC_s;
    Matrix<T,3> dSdC_ds;

    Matrix<T,3> differential(const Matrix<T,3>& dF) const
    {SymmetricMatrix<T,3> dC = twice_symmetric_part(F*dF);
    Vector<T,3> dC_d(dC.x00,dC.x11,dC.x22),dC_s((T)sqrt(2.)*dC.x10,(T)sqrt(2.)*dC.x20,(T)sqrt(2.)*dC.x21);
    Vector<T,3> dS_d(dSdC_d*dC_d+dSdC_ds*dC_s),dS_s(dSdC_ds.transpose_times(dC_d)+dSdC_s*dC_s);
    SymmetricMatrix<T,3> dS(dS_d.x,T(1/sqrt(2.))*dS_s.x,T(1/sqrt(2.))*dS_s.y,dS_d.y,T(1/sqrt(2.))*dS_s.z,dS_d.z);
    return dF*S+F*dS;}

    void enforce_definiteness()
    {S=S.positive_definite_part();
    DiagonalMatrix<T,3> D_d,D_s;Matrix<T,3> V_d,V_s;dSdC_d.fast_solve_eigenproblem(D_d,V_d);dSdC_s.fast_solve_eigenproblem(D_s,V_s);Matrix<T,3> M(V_d.transpose_times(dSdC_ds*V_s));
    D_d.x00=max(D_d.x00,abs(M(0,0))+abs(M(0,1))+abs(M(0,2)));D_d.x11=max(D_d.x11,abs(M(1,0))+abs(M(1,1))+abs(M(1,2)));D_d.x22=max(D_d.x22,abs(M(2,0))+abs(M(2,1))+abs(M(2,2)));
    D_s.x00=max(D_s.x00,abs(M(0,0))+abs(M(1,0))+abs(M(2,0)));D_s.x11=max(D_s.x11,abs(M(0,1))+abs(M(1,1))+abs(M(2,1)));D_s.x22=max(D_s.x22,abs(M(0,2))+abs(M(1,2))+abs(M(2,2)));
    dSdC_d=conjugate(V_d,D_d);dSdC_s=conjugate(V_s,D_s);}
};
}
