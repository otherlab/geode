#ifndef _VECTOR_CONVERT_H
#define _VECTOR_CONVERT_H

#include <other/core/vector/Vector.h>
#include <other/core/python/numpy.h>
#include <other/core/utility/config.h>

namespace other{

#ifdef OTHER_PYTHON

// Python conversion for arbitrary vectors
template<class T,class Enable=void> struct VectorToPython {
  template<int d> static PyObject* convert(const Vector<T,d>& v) {
    PyObject *o[d]={0},*tuple=0;
    for(int i=0;i<d;i++) if(!(o[i]=to_python(v[i]))) goto fail;
    if(!(tuple=PyTuple_New(d))) goto fail;
    for(int i=0;i<d;i++) PyTuple_SET_ITEM(tuple,i,o[i]);
    return tuple;
  fail:
    for(int i=0;i<d;i++) Py_XDECREF(o[0]);
    Py_XDECREF(tuple);
    return 0;
  }
};

// Python conversion for numpy compatible vectors
template<class T> struct VectorToPython<T,typename boost::enable_if<NumpyIsStatic<T>>::type> {
  template<int d> static PyObject* convert(const Vector<T,d>& v) {
    return to_numpy(v);
  }
};
  
template<class T,int d> PyObject* to_python(const Vector<T,d>& vector);
//template<class T,int d> Vector<T,d> FromPython<Vector<T,d> >::convert(PyObject* object);

#define VECTOR_CONVERSIONS(EXPORT,d,...) \
  template<> EXPORT PyObject* to_python<__VA_ARGS__,d>(const Vector<__VA_ARGS__,d>&v) {return VectorToPython<__VA_ARGS__>::convert(v);}\
  template<> EXPORT Vector<__VA_ARGS__,d> FromPython<Vector<__VA_ARGS__,d>>::convert(PyObject*o) { return from_numpy<Vector<__VA_ARGS__,d>>(o); }

#else // non-python stub

#define VECTOR_CONVERSIONS(d,...)

#endif
  
}

#endif

