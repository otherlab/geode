//#####################################################################
// Header Register
//#####################################################################
#include <other/core/array/RawArray.h>
#include <other/core/array/Array2d.h>
#include <other/core/python/wrap.h>
#include <other/core/vector/Register.h>
#include <other/core/vector/Frame.h>
#include <other/core/vector/DiagonalMatrix.h>
#include <other/core/vector/Matrix.h>
#include <other/core/vector/SymmetricMatrix.h>
namespace other{

typedef real T;

template<class TV> static Frame<TV> rigid_register_helper(RawArray<const TV> X0,RawArray<const TV> X1,RawArray<const real> M) {
  typedef typename TV::Scalar T;
  static const int d=TV::m;
  int n=X0.size();
  OTHER_ASSERT(M.size() == 0 || M.size() == n);
  bool weighted = M.size() == n;
  real total_mass = weighted ? M.sum() : n;

  OTHER_ASSERT(total_mass!=0);
  OTHER_ASSERT(n==X1.size());
  if(!n) return Frame<TV>();

  TV c0,c1;
  if(weighted){
    for(int i=0; i < n; ++i){
      c0+=X0[i]*M[i];
      c1+=X1[i]*M[i];
    }
  }else{
    for(int i=0; i < n; ++i){
      c0+=X0[i];
      c1+=X1[i];
    }
  }
  c0/=total_mass;
  c1/=total_mass;
  Matrix<T,d> cov=outer_product(-total_mass*c1,c0);
  if(weighted){
    for(int i=0;i<n;i++)
      cov+=outer_product(X1[i],X0[i])*M[i];
  }else{
    for(int i=0;i<n;i++)
      cov+=outer_product(X1[i],X0[i]);
  }
  cov/=total_mass;
  Matrix<T,d> U,V;DiagonalMatrix<T,d> D;
  cov.fast_singular_value_decomposition(U,D,V);
  Rotation<TV> q(U.times_transpose(V));
  return Frame<TV>(c1-q*c0,q);
}

Frame<Vector<T,2> > rigid_register(RawArray<const Vector<T,2> > X0,RawArray<const Vector<T,2> > X1,RawArray<const real> M) {
  return rigid_register_helper(X0,X1,M);
}

Frame<Vector<T,3> > rigid_register(RawArray<const Vector<T,3> > X0,RawArray<const Vector<T,3> > X1,RawArray<const real> M) {
  return rigid_register_helper(X0,X1,M);
}

#ifdef OTHER_PYTHON
Ref<PyObject> rigid_register_python(RawArray<const T,2> X0,RawArray<const T,2> X1) {
  if (X0.n==2)
    return to_python_ref(rigid_register(X0.vector_view<2>(),X1.vector_view<2>()));
  else if (X0.n==3)
    return to_python_ref(rigid_register(X0.vector_view<3>(),X1.vector_view<3>()));
  else
    OTHER_FATAL_ERROR();
}
#endif

}
using namespace other;

void wrap_register() {
#ifdef OTHER_PYTHON
  python::function("rigid_register",rigid_register_python);
#endif
}
