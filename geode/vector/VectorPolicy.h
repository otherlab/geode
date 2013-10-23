//#####################################################################
// Class VectorPolicy
//#####################################################################
#pragma once

#include <geode/vector/forward.h>
namespace geode {

template<class TV> struct VectorPolicy;

//#####################################################################
// 0D
//#####################################################################
template<class T>
struct VectorPolicy<Vector<T,0> >
{
    typedef geode::Matrix<T,0> Matrix;
    typedef Matrix SymmetricMatrix;
    typedef Matrix DiagonalMatrix;
};
//#####################################################################
// 1D
//#####################################################################
template<class T>
struct VectorPolicy<Vector<T,1> >
{
    typedef geode::Matrix<T,1> Matrix;
    typedef Matrix SymmetricMatrix;
    typedef Matrix DiagonalMatrix;
    typedef geode::Matrix<T,2> TransformationMatrix;
    typedef Vector<T,0> Spin;
};
//#####################################################################
// 2D
//#####################################################################
template<class T>
struct VectorPolicy<Vector<T,2> >
{
    typedef geode::Matrix<T,2> Matrix;
    typedef geode::SymmetricMatrix<T,2> SymmetricMatrix;
    typedef geode::DiagonalMatrix<T,2> DiagonalMatrix;
    typedef geode::Matrix<T,3> TransformationMatrix;
    typedef Vector<T,1> Spin;
};
//#####################################################################
// 3D
//#####################################################################
template<class T>
struct VectorPolicy<Vector<T,3> >
{
    typedef geode::Matrix<T,3> Matrix;
    typedef geode::SymmetricMatrix<T,3> SymmetricMatrix;
    typedef geode::DiagonalMatrix<T,3> DiagonalMatrix;
    typedef geode::Matrix<T,4> TransformationMatrix;
    typedef Vector<T,3> Spin;
};
//#####################################################################
}
