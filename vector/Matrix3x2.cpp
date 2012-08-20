//#####################################################################
// Class Matrix3x2
//#####################################################################
#include <other/core/vector/Matrix.h>
#include <other/core/vector/DiagonalMatrix2x2.h>
#include <other/core/vector/SymmetricMatrix2x2.h>
using namespace other;
//#####################################################################
// Function Fast_Singular_Value_Decomposition
//#####################################################################
template<class T> void Matrix<T,3,2>::
fast_singular_value_decomposition(Matrix<T,3,2>& U,DiagonalMatrix<T,2>& singular_values,Matrix<T,2>& V) const
{
    if(!boost::is_same<T,double>::value){
        Matrix<double,3,2> U_double;DiagonalMatrix<double,2> singular_values_double;Matrix<double,2> V_double;
        Matrix<double,3,2>(*this).fast_singular_value_decomposition(U_double,singular_values_double,V_double);
        U=Matrix<T,3,2>(U_double);singular_values=DiagonalMatrix<T,2>(singular_values_double);V=Matrix<T,2>(V_double);return;}
    // now T is double

    DiagonalMatrix<T,2> lambda;normal_equations_matrix().solve_eigenproblem(lambda,V);
    if(lambda.x11<0) lambda=lambda.clamp_min(0);
    singular_values=lambda.sqrt();
    U.set_column(0,(*this*V.column(0)).normalized());
    Vector<T,3> other=cross(weighted_normal(),U.column(0));
    T other_magnitude=other.magnitude();
    U.set_column(1,other_magnitude?other/other_magnitude:U.column(0).unit_orthogonal_vector());
}
//#####################################################################
// Function Fast_Indefinite_Polar_Decomposition
//#####################################################################
template<class T> void Matrix<T,3,2>::
fast_indefinite_polar_decomposition(Matrix<T,3,2>& Q,SymmetricMatrix<T,2>& S) const
{
    Matrix<T,3,2> U;Matrix<T,2> V;DiagonalMatrix<T,2> D;fast_singular_value_decomposition(U,D,V);
    Q=U.times_transpose(V);S=conjugate(V,D);
}
//#####################################################################
typedef real T;
template void Matrix<T,3,2>::fast_singular_value_decomposition(Matrix<T,3,2>&,DiagonalMatrix<T,2>&,Matrix<T,2>&) const;
template void Matrix<T,3,2>::fast_indefinite_polar_decomposition(Matrix<T,3,2>& Q,SymmetricMatrix<T,2>& S) const;
