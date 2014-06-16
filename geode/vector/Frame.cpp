//#####################################################################
// Class Frame
//#####################################################################
#include <geode/vector/Frame.h>
#include <geode/array/convert.h>
#include <geode/python/numpy.h>
#include <geode/python/wrap.h>
namespace geode {

typedef real T;

#ifdef GEODE_PYTHON

namespace {
PyTypeObject* frame_type;
template<class TV> struct NumpyDescr<Frame<TV>>{static PyArray_Descr* d;static PyArray_Descr* descr(){GEODE_ASSERT(d);Py_INCREF(d);return d;}};
template<class TV> struct NumpyArrayType<Frame<TV>>{static PyTypeObject* type(){GEODE_ASSERT(frame_type);Py_INCREF(frame_type);return frame_type;}};
template<class TV> struct NumpyIsStatic<Frame<TV>>:public mpl::true_{};
template<class TV> struct NumpyRank<Frame<TV>>:public mpl::int_<0>{};

template<class TV> PyArray_Descr* NumpyDescr<Frame<TV>>::d;
}

static void set_frame_type(PyObject* ft) {
  // Set frame type
  GEODE_ASSERT(PyType_Check(ft));
  Py_INCREF(ft);
  frame_type = (PyTypeObject*)ft;
  // Look up dtypes for 2d and 3d frames
  PyObject* dtypes = PyObject_GetAttrString(ft,"dtypes");
  if (!dtypes) throw_python_error();
  GEODE_ASSERT(PyDict_Check(dtypes));
  for (int d=2;d<=3;d++) {
    PyObject* dp = PyInt_FromLong(d);
    if (!dp) throw_python_error();
    PyObject* dtype = PyDict_GetItem(dtypes,dp);
    Py_DECREF(dp);
    if (!dtype) throw_python_error();
    GEODE_ASSERT(PyArray_DescrCheck(dtype));
    Py_INCREF(dtype);
    (d==2?NumpyDescr<Frame<Vector<T,2>>>::d:NumpyDescr<Frame<Vector<T,3>>>::d) = (PyArray_Descr*)dtype;
  }
}

template<class TV> PyObject* to_python(const Frame<TV>& q) {
    return to_numpy(q);
}

template<class TV> bool frames_check(PyObject* object) {
  return PyArray_Check(object) && PyArray_EquivTypes(PyArray_DESCR((PyArrayObject*)object),NumpyDescr<Frame<TV>>::d);
}

template<class TV> Frame<TV> FromPython<Frame<TV>>::convert(PyObject* object) {
    return from_numpy<Frame<TV>>(object);
}

#define INSTANTIATE(T,d) \
    template GEODE_CORE_EXPORT PyObject* to_python<Vector<T,d>>(const Frame<Vector<T,d>>&); \
    template GEODE_CORE_EXPORT Frame<Vector<T,d>> FromPython<Frame<Vector<T,d>>>::convert(PyObject*); \
    template GEODE_CORE_EXPORT bool frames_check<Vector<T,d>>(PyObject*); \
    ARRAY_CONVERSIONS(1,Frame<Vector<T,d>>)
INSTANTIATE(real,2)
INSTANTIATE(real,3)

template<class TV> static Frame<TV> frame_test(const Frame<TV>& f1,const Frame<TV>& f2,TV x) {
    return f1*f2*Frame<TV>(x);
}

template<class TV> static Array<Frame<TV>> frame_array_test(const Frame<TV>& f1,Array<const Frame<TV>> f2,TV x) {
    Array<Frame<TV>> ff(f2.size());
    for (int i=0;i<f2.size();i++)
        ff[i] = f1*f2[i]*Frame<TV>(x);
    return ff;
}

template<class TV> static Array<Frame<TV>> frame_interpolation(Array<const Frame<TV>> f1,Array<const Frame<TV>> f2,T s) {
  GEODE_ASSERT(f1.size()==f2.size());
  Array<Frame<TV>> r(f1.size(),uninit);
  for (int i=0;i<r.size();i++)
    r[i] = Frame<TV>::interpolation(f1[i],f2[i],s);
  return r;
}

#endif

}
using namespace geode;

void wrap_frame() {
#ifdef GEODE_PYTHON
  using namespace python;
  function("_set_frame_type",set_frame_type);
  function("frame_test_2d",frame_test<Vector<real,2>>);
  function("frame_test_3d",frame_test<Vector<real,3>>);
  function("frame_array_test_2d",frame_array_test<Vector<real,2>>);
  function("frame_array_test_3d",frame_array_test<Vector<real,3>>);
  function("frame_interpolation_2d",frame_interpolation<Vector<real,2>>);
  function("frame_interpolation_3d",frame_interpolation<Vector<real,3>>);
#endif
}
