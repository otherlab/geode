#pragma once

#include <geode/vector/Vector.h>
#include <geode/python/numpy.h>
#include <geode/utility/config.h>
namespace geode {

#ifdef GEODE_PYTHON

// In other to instantiate a Vector conversion, place GEODE_DECLARE_VECTOR_CONVERSIONS (see vector/forward.h)
// in a header and GEODE_DEFINE_VECTOR_CONVERSIONS in a .cpp.
#define GEODE_DEFINE_VECTOR_CONVERSIONS(EXPORT,d,...) \
  static_assert(NumpyIsStatic<__VA_ARGS__>::value,""); \
  PyObject* to_python(const Vector<__VA_ARGS__,d>& v) { return to_numpy(v); } \
  Vector<__VA_ARGS__,d> FromPython<Vector<__VA_ARGS__,d>>::convert(PyObject* o) { return from_numpy<Vector<__VA_ARGS__,d>>(o); }

// To python conversion for arbitrary vectors
template<class T,int d> typename enable_if<has_to_python<T>, PyObject*>::type to_python(const Vector<T,d>& v) {
  static_assert(!NumpyIsStatic<T>::value,"Numpy compatible types should use GEODE_DECLARE/DEFINE_VECTOR_CONVERSIONS");
  PyObject *o[d]={0},*tuple=0;
  for (int i=0;i<d;i++)
    if (!(o[i]=to_python(v[i])))
      goto fail;
  if (!(tuple=PyTuple_New(d)))
    goto fail;
  for (int i=0;i<d;i++)
    PyTuple_SET_ITEM(tuple,i,o[i]);
  return tuple;
fail:
  for (int i=0;i<d;i++)
    Py_XDECREF(o[0]);
  Py_XDECREF(tuple);
  return 0;
}

#else // non-python stub

#define GEODE_DEFINE_VECTOR_CONVERSIONS(...)

#endif

}
