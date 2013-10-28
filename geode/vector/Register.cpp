//#####################################################################
// Header Register
//#####################################################################
#include <geode/vector/Register.h>
#include <geode/array/RawArray.h>
#include <geode/array/Array2d.h>
#include <geode/python/wrap.h>
#include <geode/vector/Frame.h>
#include <geode/vector/DiagonalMatrix.h>
#include <geode/vector/Matrix.h>
#include <geode/vector/SymmetricMatrix.h>
namespace geode {

typedef real T;

template<class TV> static inline Tuple<T,TV> weighted_average(RawArray<const T> mass, RawArray<const TV> X) {
  const int n = X.size();
  if (mass.size()) {
    GEODE_ASSERT(mass.size()==X.size());
    const T total = mass.sum();
    TV center;
    for (int i=0;i<n;i++)
      center += mass[i]*X[i];
    return tuple(total,center/total);
  } else {
    // Unweighted
    return tuple(n,X.mean());
  }
}

template<class TV> static Frame<TV> rigid_register_helper(RawArray<const TV> X0, RawArray<const TV> X1, RawArray<const T> mass) {
  typedef typename TV::Scalar T;
  static const int d = TV::m;
  const int n = X0.size();
  GEODE_ASSERT(n==X1.size() && (mass.size()==0 || mass.size()==n));
  const bool weighted = mass.size() == n;
  const T total = weighted ? mass.sum() : n;
  if (!total)
    return Frame<TV>();

  // Compute centers of mass
  TV c0, c1;
  if (weighted) {
    for (int i=0;i<n;i++) {
      c0 += mass[i]*X0[i];
      c1 += mass[i]*X1[i];
    }
  } else {
    c0 = X0.sum();
    c1 = X1.sum();
  }
  c0 /= total;
  c1 /= total;

  // Compute covariance
  Matrix<T,d> cov = outer_product(-total*c1,c0);
  if (weighted)
    for(int i=0;i<n;i++)
      cov += outer_product(mass[i]*X1[i],X0[i]);
  else
    for (int i=0;i<n;i++)
      cov += outer_product(X1[i],X0[i]);

  // Compute frame from covariance
  Matrix<T,d> U,V;DiagonalMatrix<T,d> D;
  cov.fast_singular_value_decomposition(U,D,V);
  Rotation<TV> q(U.times_transpose(V));
  return Frame<TV>(c1-q*c0,q);
}

template<class TV> static Matrix<T,TV::m+1> affine_register_helper(RawArray<const TV> X0, RawArray<const TV> X1, RawArray<const T> mass) {
  typedef typename TV::Scalar T;
  static const int d = TV::m;
  const int n = X0.size();
  GEODE_ASSERT(n==X1.size() && (mass.size()==0 || mass.size()==n));
  const bool weighted = mass.size() == n;
  const T total = weighted ? mass.sum() : n;
  if (!total)
    return Matrix<T,d+1>::identity_matrix();

  // Compute centers of mass
  TV c0, c1;
  if (weighted) {
    for (int i=0;i<n;i++) {
      c0 += mass[i]*X0[i];
      c1 += mass[i]*X1[i];
    }
  } else {
    c0 = X0.sum();
    c1 = X1.sum();
  }
  c0 /= total;
  c1 /= total;

  // Compute covariances
  SymmetricMatrix<T,d> cov00 = scaled_outer_product(-total,c0);
  Matrix<T,d> cov01 = outer_product(-total*c1,c0);
  if (weighted)
    for(int i=0;i<n;i++) {
      cov00 += scaled_outer_product(mass[i],X0[i]);
      cov01 += outer_product(mass[i]*X1[i],X0[i]);
    }
  else
    for (int i=0;i<n;i++) {
      cov00 += outer_product(X0[i]);
      cov01 += outer_product(X1[i],X0[i]);
    }

  // Compute transform
  const Matrix<T,d> A = cov01*cov00.inverse();
  const TV t = c1-A*c0;
  auto tA = Matrix<T,d+1>::from_linear(A);
  tA.set_translation(t);
  return tA;
}

Frame<Vector<T,2>> rigid_register(RawArray<const Vector<T,2>> X0, RawArray<const Vector<T,2>> X1, RawArray<const T> mass) {
  return rigid_register_helper(X0,X1,mass);
}

Frame<Vector<T,3>> rigid_register(RawArray<const Vector<T,3>> X0, RawArray<const Vector<T,3>> X1, RawArray<const T> mass) {
  return rigid_register_helper(X0,X1,mass);
}

Matrix<T,3> affine_register(RawArray<const Vector<T,2>> X0, RawArray<const Vector<T,2>> X1, RawArray<const T> mass) {
  return affine_register_helper(X0,X1,mass);
}

Matrix<T,4> affine_register(RawArray<const Vector<T,3>> X0, RawArray<const Vector<T,3>> X1, RawArray<const T> mass) {
  return affine_register_helper(X0,X1,mass);
}

#ifdef GEODE_PYTHON
Ref<PyObject> rigid_register_python(RawArray<const T,2> X0, RawArray<const T,2> X1) {
  GEODE_ASSERT(X0.n==X1.n);
  if (X0.n==2)
    return to_python_ref(rigid_register(X0.vector_view<2>(),X1.vector_view<2>()));
  else if (X0.n==3)
    return to_python_ref(rigid_register(X0.vector_view<3>(),X1.vector_view<3>()));
  else
    GEODE_FATAL_ERROR();
}

Ref<PyObject> affine_register_python(RawArray<const T,2> X0, RawArray<const T,2> X1) {
  GEODE_ASSERT(X0.n==X1.n);
  if (X0.n==2)
    return to_python_ref(affine_register(X0.vector_view<2>(),X1.vector_view<2>()));
  else if (X0.n==3)
    return to_python_ref(affine_register(X0.vector_view<3>(),X1.vector_view<3>()));
  else
    GEODE_FATAL_ERROR();
}
#endif

}
using namespace geode;

void wrap_register() {
#ifdef GEODE_PYTHON
  GEODE_FUNCTION_2(rigid_register,rigid_register_python)
  GEODE_FUNCTION_2(affine_register,affine_register_python)
#endif
}
