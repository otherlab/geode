//#####################################################################
// Class Rotation
//#####################################################################
#include <other/core/python/from_python.h>
#include <other/core/array/convert.h>
#include <other/core/vector/Matrix.h>
#include <other/core/vector/Rotation.h>
#include <other/core/python/numpy.h>
#include <other/core/python/wrap.h>
namespace other{

typedef real T;

#ifdef OTHER_PYTHON

namespace {
template<class TV> struct NumpyDescr<Rotation<TV>>{static PyArray_Descr* d;static PyArray_Descr* descr(){OTHER_ASSERT(d);Py_INCREF(d);return d;}};
template<class TV> struct NumpyArrayType<Rotation<TV>>{static PyTypeObject* t;static PyTypeObject* type(){OTHER_ASSERT(t);Py_INCREF(t);return t;}};
template<class TV> struct NumpyIsStatic<Rotation<TV>>:public mpl::true_{};
template<class TV> struct NumpyRank<Rotation<TV>>:public mpl::int_<0>{};

template<class TV> PyArray_Descr* NumpyDescr<Rotation<TV>>::d;
template<class TV> PyTypeObject* NumpyArrayType<Rotation<TV>>::t;
}

static void set_rotation_types(PyObject* t2d,PyObject* t3d) {
  OTHER_ASSERT(PyType_Check(t2d));
  OTHER_ASSERT(PyType_Check(t3d));
  PyObject* d2d = PyObject_GetAttrString(t2d,"dtype");
  if (!d2d) throw_python_error();
  PyObject* d3d = PyObject_GetAttrString(t3d,"dtype");
  if (!d3d) throw_python_error();
  OTHER_ASSERT(PyArray_DescrCheck(d2d));
  OTHER_ASSERT(PyArray_DescrCheck(d3d));
  Py_INCREF(t2d);
  Py_INCREF(t3d);
  Py_INCREF(d2d);
  Py_INCREF(d3d);
  NumpyArrayType<Rotation<Vector<real,2>>>::t = (PyTypeObject*)t2d;
  NumpyArrayType<Rotation<Vector<real,3>>>::t = (PyTypeObject*)t3d;
  NumpyDescr<Rotation<Vector<real,2>>>::d = (PyArray_Descr*)d2d;
  NumpyDescr<Rotation<Vector<real,3>>>::d = (PyArray_Descr*)d3d;
}

template<class TV> PyObject* to_python(const Rotation<TV>& q) {
  return to_numpy(q);
}

template<class TV> bool rotations_check(PyObject* object) {
  return PyArray_Check(object) && PyArray_EquivTypes(PyArray_DESCR((PyArrayObject*)object),NumpyDescr<Rotation<TV>>::d);
}

template<class TV> Rotation<TV> FromPython<Rotation<TV>>::convert(PyObject* object) {
  if (!rotations_check<TV>(object))
    throw_array_conversion_error(object,0,0,NumpyDescr<Rotation<TV>>::d);
  return from_numpy<Rotation<TV>>(object);
}

#define INSTANTIATE(T,d) \
    template OTHER_CORE_EXPORT PyObject* to_python<Vector<T,d>>(const Rotation<Vector<T,d>>&); \
    template OTHER_CORE_EXPORT Rotation<Vector<T,d>> FromPython<Rotation<Vector<T,d>>>::convert(PyObject*); \
    template OTHER_CORE_EXPORT bool rotations_check<Vector<T,d>>(PyObject*); \
    ARRAY_CONVERSIONS(1,Rotation<Vector<T,d>>)
INSTANTIATE(real,2)
INSTANTIATE(real,3)

template<class TV> static Rotation<TV> rotation_test(const Rotation<TV>& r) {
    return r*r;
}

template<class TV> static Array<Rotation<TV>> rotation_array_test(Array<const Rotation<TV>> r) {
    Array<Rotation<TV>> rr(r.size());
    for (int i=0;i<r.size();i++)
        rr[i] = sqr(r[i]);
    return rr;
}

static PyObject* rotation_from_matrix(NdArray<const real> A) {
  OTHER_ASSERT(A.rank()>=2);
  const int r = A.rank();
  if (A.shape[r-1]==2 && A.shape[r-2]==2) {
    NdArray<Rotation<Vector<T,2>>> rs(A.shape.slice_own(0,r-2),false);
    for (const int i : range(rs.flat.size()))
      rs.flat[i] = Rotation<Vector<T,2>>(Matrix<real,2>(A.flat.slice(4*i,4*(i+1)).reshape(2,2)));
    return to_python(rs);
  } else if (A.shape[r-1]==3 && A.shape[r-2]==3) {
    NdArray<Rotation<Vector<T,3>>> rs(A.shape.slice_own(0,r-2),false);
    for (const int i : range(rs.flat.size()))
      rs.flat[i] = Rotation<Vector<T,3>>(Matrix<real,3>(A.flat.slice(9*i,9*(i+1)).reshape(3,3)));
    return to_python(rs);
  } else
    throw TypeError(format("expected 2x2 or 3x3 matrices, got shape %s",str(A.shape)));
}

#endif
}
using namespace other;

void wrap_rotation() {
#ifdef OTHER_PYTHON
  using namespace python;
  function("_set_rotation_types",set_rotation_types);
  function("rotation_test_2d",rotation_test<Vector<real,2>>);
  function("rotation_test_3d",rotation_test<Vector<real,3>>);
  function("rotation_array_test_2d",rotation_array_test<Vector<real,2>>);
  function("rotation_array_test_3d",rotation_array_test<Vector<real,3>>);
  OTHER_FUNCTION(rotation_from_matrix);
#endif
}
