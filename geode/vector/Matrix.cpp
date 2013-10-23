//#####################################################################
// Class Matrix
//#####################################################################
#include <geode/vector/Matrix.h>
#include <geode/array/convert.h>
#include <geode/python/numpy.h>
#include <geode/python/wrap.h>
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

#define INSTANTIATE(T,m,n) \
  template GEODE_CORE_EXPORT PyObject* to_python<T,m,n>(const Matrix<T,m,n>&); \
  template GEODE_CORE_EXPORT Matrix<T,m,n> FromPython<Matrix<T,m,n> >::convert(PyObject*); \
  ARRAY_CONVERSIONS(1,Matrix<T,m,n>)
INSTANTIATE(real,2,2)
INSTANTIATE(real,3,3)
INSTANTIATE(real,4,4)

#endif

}
using namespace geode;

void wrap_matrix() {
#ifdef GEODE_PYTHON
  using namespace python;
  function("_set_matrix_type",set_matrix_type);
#endif
}
