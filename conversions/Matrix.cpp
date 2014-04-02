//#####################################################################
// Class Matrix
//#####################################################################
#include <geode/vector/Matrix.h>
#include <geode/vector/DiagonalMatrix.h>
#include <geode/array/view.h>
namespace geode {

#ifdef GEODE_PYTHON

static PyTypeObject* matrix_type;

namespace {
template<class T,int m,int n> struct NumpyArrayType<Matrix<T,m,n> >{static PyTypeObject* type(){return matrix_type;}};
}

static void set_matrix_type(PyObject* type) {
  GEODE_ASSERT(PyType_Check(type));
  Py_INCREF(type);
  matrix_type = (PyTypeObject*)type;
}

template<class T,int m,int n> PyObject*
to_python(const Matrix<T,m,n>& matrix) {
  return to_numpy(matrix);
}

template<class T,int m,int n> Matrix<T,m,n> FromPython<Matrix<T,m,n> >::
convert(PyObject* object) {
  return from_numpy<Matrix<T,n,m> >(object);
}

static NdArray<real> fast_singular_values_py(const NdArray<const real>& A) {
  GEODE_ASSERT(A.rank()>=2);
  const int r = A.rank();
  if (!(A.shape[r-2]==A.shape[r-1] && 2<=A.shape[r-1] && A.shape[r-1]<=3))
    throw NotImplementedError(format(
      "fast_singular_values: got shape %s, only arrays of 2x2 and 3x3 matrices implemented for now",str(A.shape)));
  NdArray<real> D(concatenate(A.shape.slice(0,r-2),asarray(vec(min(A.shape[r-1],A.shape[r-2])))),false);
  const auto Ds = D.flat.reshape(D.flat.size()/D.shape.back(),D.shape.back());
  switch (A.shape[r-1]) {
    #define CASE(d) \
      case d: { \
        const auto As = vector_view<Matrix<real,d>>(A.flat); \
        for (const int i : range(Ds.m)) \
          Ds[i] = asarray(fast_singular_values(As[i]).to_vector()); \
        break; \
      }
    CASE(2) CASE(3)
    #undef CASE
    default: GEODE_UNREACHABLE();
  }
  return D;
}

#define INSTANTIATE(T,m,n) \
  template GEODE_CORE_EXPORT PyObject* to_python<T,m,n>(const Matrix<T,m,n>&); \
  template GEODE_CORE_EXPORT Matrix<T,m,n> FromPython<Matrix<T,m,n> >::convert(PyObject*); \
  ARRAY_CONVERSIONS(1,Matrix<T,m,n>)
INSTANTIATE(float,2,2)
INSTANTIATE(float,3,3)
INSTANTIATE(float,4,4)
INSTANTIATE(double,2,2)
INSTANTIATE(double,3,3)
INSTANTIATE(double,4,4)

#endif

}
using namespace geode;

void wrap_matrix() {
#ifdef GEODE_PYTHON
  using namespace python;
  function("_set_matrix_type",set_matrix_type);
  GEODE_FUNCTION_2(fast_singular_values,fast_singular_values_py)
#endif
}
