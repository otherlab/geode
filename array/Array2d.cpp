//#####################################################################
// Class Array<T,2>
//#####################################################################
#include <other/core/array/Array2d.h>
#include <other/core/python/numpy.h>
#include <other/core/vector/Vector.h>
#ifdef USE_MKL
#include <other/core/vector/blas.h>
#include <mkl_trans.h>
#endif
namespace other {

Array<real,2> identity_matrix(int m,int n) {
  if (n<0) n = m;
  Array<real,2> A(m,n,false);
#ifdef USE_MKL
  laset(' ',0,1,A);
#else
  memset(A.data(),0,m*n*sizeof(real));
  for (int i=0;i<min(m,n);i++)
    A(i,i) = 1;
#endif
  return A;
}

template<class T> static inline Array<T,2> dot_helper(Subarray<const T,2> A, Subarray<const T,2> B) {
  OTHER_ASSERT(A.n==B.m);
  Array<T,2> C(A.m,B.n,false);
#ifdef USE_MKL
  gemm(1,A,B,0,C);
#else
  for (int i=0;i<C.m;i++) for (int j=0;j<C.n;j++) {
    T sum = 0;
    for (int k=0;k<A.n;k++)
      sum += A(i,k)*B(k,j);
    C(i,j)=sum;
  }
#endif
  return C;
}
Array<float,2> dot(Subarray<const float,2> A,Subarray<const float,2> B){return dot_helper(A,B);}
Array<double,2> dot(Subarray<const double,2> A,Subarray<const double,2> B){return dot_helper(A,B);}

template<class T> static inline Array<T> dot_helper(Subarray<const T,2> A, RawArray<const T> x) {
  OTHER_ASSERT(A.n==x.size());
  Array<T> y(A.m,false);
#ifdef USE_MKL
  gemv(1,A,x,0,y);
#else
  for (int i=0;i<A.m;i++) {
    T sum = 0;
    for (int j=0;j<A.n;j++)
      sum += A(i,j)*x[j];
    y(i) = sum;
  }
#endif
  return y;
}
Array<float> dot(Subarray<const float,2> A,RawArray<const float> x){return dot_helper(A,x);}
Array<double> dot(Subarray<const double,2> A,RawArray<const double> x){return dot_helper(A,x);}

}
