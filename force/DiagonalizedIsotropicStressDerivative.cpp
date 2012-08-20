//#####################################################################
// Class DiagonalizedIsotropicStressDerivative
//#####################################################################
#include <other/core/force/DiagonalizedIsotropicStressDerivative.h>
#include <other/core/vector/DiagonalMatrix2x2.h>
#include <other/core/vector/DiagonalMatrix3x3.h>
#include <other/core/vector/Matrix2x2.h>
#include <other/core/vector/Matrix3x3.h>
#include <other/core/vector/SymmetricMatrix2x2.h>
#include <other/core/vector/SymmetricMatrix3x3.h>
#include <other/core/vector/Vector.h>
namespace other{

template<class T> void DiagonalizedIsotropicStressDerivative<T,2>::
enforce_definiteness()
{
    SymmetricMatrix<T,2> A1(x0000,x1100,x1111);DiagonalMatrix<T,2> D1;Matrix<T,2> V1;
    A1.solve_eigenproblem(D1,V1);D1=D1.clamp_min(0);A1=conjugate(V1,D1);
    x0000=A1.x00;x1100=A1.x10;x1111=A1.x11;
    SymmetricMatrix<T,2> A2(x1010,x1001,x1010);DiagonalMatrix<T,2> D2;Matrix<T,2> V2;
    A2.solve_eigenproblem(D2,V2);D2=D2.clamp_min(0);A2=conjugate(V2,D2);
    x1010=A2.x00;x1001=A2.x10;
}

template<class T> void DiagonalizedIsotropicStressDerivative<T,3,2>::
enforce_definiteness()
{
    A.enforce_definiteness();
    x2020 = max(x2020,(T)0);
    x2121 = max(x2121,(T)0);
}

template<class T> void DiagonalizedIsotropicStressDerivative<T,3>::
enforce_definiteness(const T eigenvalue_clamp_percentage,const T epsilon)
{
    SymmetricMatrix<T,3> A1(x0000,x1100,x2200,x1111,x2211,x2222);DiagonalMatrix<T,3> D1;Matrix<T,3> V1;A1.fast_solve_eigenproblem(D1,V1);
    SymmetricMatrix<T,2> A2(x1010,x1001,x1010);DiagonalMatrix<T,2> D2;Matrix<T,2> V2;A2.solve_eigenproblem(D2,V2);
    SymmetricMatrix<T,2> A3(x2020,x2002,x2020);DiagonalMatrix<T,2> D3;Matrix<T,2> V3;A3.solve_eigenproblem(D3,V3);
    SymmetricMatrix<T,2> A4(x2121,x2112,x2121);DiagonalMatrix<T,2> D4;Matrix<T,2> V4;A4.solve_eigenproblem(D4,V4);
    T min_nonzero_absolute_eigenvalue=min(epsilon,D1.minabs(),D2.minabs(),D3.minabs(),D4.minabs());
    if(D1.x00<-epsilon) D1.x00=eigenvalue_clamp_percentage*min_nonzero_absolute_eigenvalue;else if (D1.x00<(T)0) D1.x00=(T)0;
    if(D1.x11<-epsilon) D1.x11=eigenvalue_clamp_percentage*min_nonzero_absolute_eigenvalue;else if (D1.x11<(T)0) D1.x11=(T)0;
    if(D1.x22<-epsilon) D1.x22=eigenvalue_clamp_percentage*min_nonzero_absolute_eigenvalue;else if (D1.x22<(T)0) D1.x22=(T)0;
    if(D2.x00<-epsilon) D2.x00=eigenvalue_clamp_percentage*min_nonzero_absolute_eigenvalue;else if (D2.x00<(T)0) D2.x00=(T)0;
    if(D2.x11<-epsilon) D2.x11=eigenvalue_clamp_percentage*min_nonzero_absolute_eigenvalue;else if (D2.x11<(T)0) D2.x11=(T)0;
    if(D3.x00<-epsilon) D3.x00=eigenvalue_clamp_percentage*min_nonzero_absolute_eigenvalue;else if (D3.x00<(T)0) D3.x00=(T)0;
    if(D3.x11<-epsilon) D3.x11=eigenvalue_clamp_percentage*min_nonzero_absolute_eigenvalue;else if (D3.x11<(T)0) D3.x11=(T)0;
    if(D4.x00<-epsilon) D4.x00=eigenvalue_clamp_percentage*min_nonzero_absolute_eigenvalue;else if (D4.x00<(T)0) D4.x00=(T)0;
    if(D4.x11<-epsilon) D4.x11=eigenvalue_clamp_percentage*min_nonzero_absolute_eigenvalue;else if (D4.x11<(T)0) D4.x11=(T)0;
    A1=conjugate(V1,D1);x0000=A1.x00;x1100=A1.x10;x2200=A1.x20;x1111=A1.x11;x2211=A1.x21;x2222=A1.x22;
    A2=conjugate(V2,D2);x1010=A2.x00;x1001=A2.x10;
    A3=conjugate(V3,D3);x2020=A3.x00;x2002=A3.x10;
    A4=conjugate(V4,D4);x2121=A4.x00;x2112=A4.x10;
}

template class DiagonalizedIsotropicStressDerivative<real,2>;
template class DiagonalizedIsotropicStressDerivative<real,3,2>;
template class DiagonalizedIsotropicStressDerivative<real,3>;
}
