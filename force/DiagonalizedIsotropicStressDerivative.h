//#####################################################################
// Class DiagonalizedIsotropicStressDerivative
//#####################################################################
#pragma once

#include <other/core/force/forward.h>
#include <other/core/vector/forward.h>
namespace other {

template<class T> class DiagonalizedIsotropicStressDerivative<T,2> {
public:
  T x0000,x1100,x1111; // 2x2 block
  T x1010,x1001; // 2x2 block

  Matrix<T,2> differential(const Matrix<T,2>& dF) const {
    return Matrix<T,2>(x0000*dF(0,0)+x1100*dF(1,1),x1010*dF(1,0)+x1001*dF(0,1),
                       x1001*dF(1,0)+x1010*dF(0,1),x1100*dF(0,0)+x1111*dF(1,1));
  }

  void enforce_definiteness();
};

template<class T> class DiagonalizedIsotropicStressDerivative<T,3,2> {
public:
  DiagonalizedIsotropicStressDerivative<T,2> A; // in plane
  T x2020,x2121; // out of plane

  Matrix<T,3,2> differential(const Matrix<T,3,2>& dF) const {
    return Matrix<T,3,2>(A.x0000*dF(0,0)+A.x1100*dF(1,1),A.x1010*dF(1,0)+A.x1001*dF(0,1),x2020*dF(2,0),
                         A.x1001*dF(1,0)+A.x1010*dF(0,1),A.x1100*dF(0,0)+A.x1111*dF(1,1),x2121*dF(2,1));
  }

  void enforce_definiteness();
};

template<class T> class DiagonalizedIsotropicStressDerivative<T,3> {
public:
  T x0000,x1100,x2200,x1111,x2211,x2222; // 3x3 block
  T x1010,x1001; // 2x2 block
  T x2020,x2002; // 2x2 block
  T x2112,x2121; // 2x2 block

  Matrix<T,3> differential(const Matrix<T,3>& dF) const {
    return Matrix<T,3>(x0000*dF(0,0)+x1100*dF(1,1)+x2200*dF(2,2),x1010*dF(1,0)+x1001*dF(0,1),x2020*dF(2,0)+x2002*dF(0,2),
                       x1001*dF(1,0)+x1010*dF(0,1),x1100*dF(0,0)+x1111*dF(1,1)+x2211*dF(2,2),x2121*dF(2,1)+x2112*dF(1,2),
                       x2002*dF(2,0)+x2020*dF(0,2),x2112*dF(2,1)+x2121*dF(1,2),x2200*dF(0,0)+x2211*dF(1,1)+x2222*dF(2,2));
  }

  void enforce_definiteness(const T eigenvalue_clamp_percentage=(T)0, const T epsilon=(T)1e-4);
};

}
