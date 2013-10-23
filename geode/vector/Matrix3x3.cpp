//#####################################################################
// Class Matrix3x3
//#####################################################################
#include <geode/vector/Matrix.h>
#include <geode/vector/DiagonalMatrix3x3.h>
#include <geode/vector/Rotation.h>
#include <geode/vector/SymmetricMatrix3x3.h>
#include <geode/vector/UpperTriangularMatrix3x3.h>
#include <geode/array/RawArray.h>
#include <geode/math/maxabs.h>
#include <geode/math/minabs.h>
#include <geode/utility/Log.h>
namespace geode {
//#####################################################################
// Constructor
//#####################################################################
template<class T> Matrix<T,3>::
Matrix(RawArray<const T,2> matrix)
{
    assert(matrix.m==3 && matrix.n==3);for(int i=0;i<3;i++) for(int j=0;j<3;j++) x[i][j]=matrix(i,j);
}
//#####################################################################
// Function Higham_Iterate
//#####################################################################
template<class T> Matrix<T,3> Matrix<T,3>::
higham_iterate(const T tolerance,const int max_iterations,const bool exit_on_max_iterations) const
{
    Matrix<T,3> X=*this;int iterations=0;
    for(;;){
        Matrix<T,3> Y=(T).5*(X+X.inverse_transposed());
        if((X-Y).maxabs()<tolerance) return Y;
        X=Y;
        if(++iterations>=max_iterations){
            if(exit_on_max_iterations) GEODE_FATAL_ERROR();
            return X;}}
}
//#####################################################################
// Function Fast_Singular_Value_Decomposition
//#####################################################################
// U and V rotations, smallest singular value possibly negative
template<class T> void Matrix<T,3>::
fast_singular_value_decomposition(Matrix<T,3>& U,DiagonalMatrix<T,3>& singular_values,Matrix<T,3>& V) const // 182 mults, 112 adds, 6 divs, 11 sqrts, 1 atan2, 1 sincos
{
    if(!boost::is_same<T,double>::value){
        Matrix<double,3> U_double,V_double;DiagonalMatrix<double,3> singular_values_double;
        Matrix<double,3>(*this).fast_singular_value_decomposition(U_double,singular_values_double,V_double);
        U=Matrix<T,3>(U_double);singular_values=DiagonalMatrix<T,3>(singular_values_double);V=Matrix<T,3>(V_double);return;}
    // now T is double

    // decompose normal equations
    DiagonalMatrix<T,3> lambda;
    normal_equations_matrix().fast_solve_eigenproblem(lambda,V); // 18m+12a + 95m+64a+3d+5s+1atan2+1sincos

    // compute singular values
    if(lambda.x22<0) lambda=lambda.clamp_min(0);
    singular_values=lambda.sqrt(); // 3s
    if(determinant()<0) singular_values.x22=-singular_values.x22; // 9m+5a

    // compute singular vectors
    U.set_column(0,(*this*V.column(0)).normalized()); // 15m+8a+1d+1s
    Vector<T,3> v1_orthogonal=U.column(0).unit_orthogonal_vector(); // 6m+2a+1d+1s
    Matrix<T,3,2> other_v=HStack(v1_orthogonal,cross(U.column(0),v1_orthogonal)); // 6m+3a
    U.set_column(1,other_v*(other_v.transpose_times(*this*V.column(1))).normalized()); // 6m+3a + 6m+4a + 9m+6a + 6m+2a+1d+1s = 27m+15a+1d+1s
    U.set_column(2,cross(U.column(0),U.column(1))); // 6m+3a
}
//#####################################################################
// Function Fast_Indefinite_Polar_Decomposition
//#####################################################################
template<class T> void Matrix<T,3>::
fast_indefinite_polar_decomposition(Matrix<T,3>& Q,SymmetricMatrix<T,3>& S) const
{
    Matrix<T,3> U,V;DiagonalMatrix<T,3> D;fast_singular_value_decomposition(U,D,V);
    Q=U.times_transpose(V);S=conjugate(V,D);
}
//#####################################################################
// Function Simplex_Minimum_Altitude
//#####################################################################
template<class T> T Matrix<T,3>::
simplex_minimum_altitude() const
{
    typedef Vector<T,3> TV;
    TV X1=column(0),X2=column(1),X3=column(2);
    return minabs(
        dot(X1,cross(X2-X1,X3-X1).normalized()),
        dot(X2-X1,cross(X3,X2).normalized()),
        dot(X3-X2,cross(X1,X3).normalized()),
        dot(X3,cross(X1,X2).normalized()));
}
//#####################################################################
template class Matrix<real,3>;
}
