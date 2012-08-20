//#####################################################################
// Class VectorPolicy
//#####################################################################
#pragma once

#include <other/core/vector/forward.h>
namespace other {

template<class TV> struct VectorPolicy;

//#####################################################################
// 0D
//#####################################################################
template<class T>
struct VectorPolicy<Vector<T,0> >
{
    typedef other::Matrix<T,0> Matrix;
    typedef Matrix SymmetricMatrix;
    typedef Matrix DiagonalMatrix;
};
//#####################################################################
// 1D
//#####################################################################
template<class T>
struct VectorPolicy<Vector<T,1> >
{
    typedef other::Matrix<T,1> Matrix;
    typedef Matrix SymmetricMatrix;
    typedef Matrix DiagonalMatrix;
    typedef other::Matrix<T,2> TransformationMatrix;
    typedef Vector<T,0> Spin;
};
//#####################################################################
// 2D
//#####################################################################
template<class T>
struct VectorPolicy<Vector<T,2> >
{
    typedef other::Matrix<T,2> Matrix;
    typedef other::SymmetricMatrix<T,2> SymmetricMatrix;
    typedef other::DiagonalMatrix<T,2> DiagonalMatrix;
    typedef other::Matrix<T,3> TransformationMatrix;
    typedef Vector<T,1> Spin;
};
//#####################################################################
// 3D
//#####################################################################
template<class T>
struct VectorPolicy<Vector<T,3> >
{
    typedef other::Matrix<T,3> Matrix;
    typedef other::SymmetricMatrix<T,3> SymmetricMatrix;
    typedef other::DiagonalMatrix<T,3> DiagonalMatrix;
    typedef other::Matrix<T,4> TransformationMatrix;
    typedef Vector<T,3> Spin;
};
//#####################################################################
}
