#pragma once

#include <other/core/vector/Vector.h>
#include <other/core/python/numpy.h>
#include <other/core/utility/config.h>
#include <boost/mpl/assert.hpp>
namespace other {

#ifdef OTHER_PYTHON

// In other to instantiate a Vector conversion, place OTHER_DECLARE_VECTOR_CONVERSIONS (see vector/forward.h)
// in a header and OTHER_DEFINE_VECTOR_CONVERSIONS in a .cpp.
#define OTHER_DEFINE_VECTOR_CONVERSIONS(EXPORT,d,...) \
  BOOST_MPL_ASSERT((NumpyIsStatic<__VA_ARGS__>)); \
  PyObject* to_python(const Vector<__VA_ARGS__,d>& v) { return to_numpy(v); } \
  Vector<__VA_ARGS__,d> FromPython<Vector<__VA_ARGS__,d>>::convert(PyObject* o) { return from_numpy<Vector<__VA_ARGS__,d>>(o); }

// To python conversion for arbitrary vectors
template<class T,int d> PyObject* to_python(const Vector<T,d>& v) {
  BOOST_STATIC_ASSERT(!NumpyIsStatic<T>::value); // numpy compatible types should use OTHER_DECLARE/DEFINE_VECTOR_CONVERSIONS
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

#define OTHER_DEFINE_VECTOR_CONVERSIONS(...)

#endif

}
