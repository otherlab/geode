//#####################################################################
// Class Matrix2x2
//#####################################################################
#include <geode/vector/Matrix2x2.h>
#include <geode/vector/DiagonalMatrix2x2.h>
#include <geode/vector/SymmetricMatrix2x2.h>
namespace geode {

template<class T> void Matrix<T,2>::
indefinite_polar_decomposition(Matrix& Q,SymmetricMatrix<T,2>& S) const
{
    T x03=x[0][0]+x[1][1],cosine,sine;if(x03==0){cosine=0;sine=1;}else{T t=(x[1][0]-x[0][1])/x03;cosine=1/sqrt(1+t*t);sine=t*cosine;}
    Q=Matrix(cosine,sine,-sine,cosine);S=SymmetricMatrix<T,2>(Q.x[0][0]*x[0][0]+Q.x[1][0]*x[1][0],Q.x[0][0]*x[0][1]+Q.x[1][0]*x[1][1],Q.x[0][1]*x[0][1]+Q.x[1][1]*x[1][1]);
}

template<class T> void Matrix<T,2>::
fast_singular_value_decomposition(Matrix& U,DiagonalMatrix<T,2>& singular_values,Matrix& V) const
{
    Matrix Q;SymmetricMatrix<T,2> S;indefinite_polar_decomposition(Q,S);S.solve_eigenproblem(singular_values,V);
    if(singular_values.x11<0 && abs(singular_values.x11)>=abs(singular_values.x00)){
        singular_values=DiagonalMatrix<T,2>(-singular_values.x11,-singular_values.x00);
        Q=-Q;V=Matrix(V.x[0][1],V.x[1][1],-V.x[0][0],-V.x[1][0]);}
    U=Q*V;
}

typedef real T;
template void Matrix<T,2>::indefinite_polar_decomposition(Matrix<T,2>& Q,SymmetricMatrix<T,2>& S) const;
template void Matrix<T,2>::fast_singular_value_decomposition(Matrix<T,2>&,DiagonalMatrix<T,2>&,Matrix<T,2>&) const;
}
