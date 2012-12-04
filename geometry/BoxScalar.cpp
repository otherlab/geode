//#####################################################################
// Class Box<T>
//#####################################################################
#include <other/core/geometry/BoxScalar.h>
#include <other/core/python/exceptions.h>
namespace other {

#ifdef OTHER_PYTHON

template<class T> PyObject* to_python(const Box<T>& self) {
  const char* format=boost::is_same<T,float>::value?"ff":"dd";
  return Py_BuildValue(format,self.min,self.max);
}

template<class T> Box<T> FromPython<Box<T> >::convert(PyObject* object) {
  Box<T> self;
  const char* format = boost::is_same<T,float>::value?"ff":"dd";
  if (PyArg_ParseTuple(object,format,&self.min,&self.max))
    return self;
  throw_python_error();
}

#define INSTANTIATE(T) \
  template PyObject* to_python(const Box<T>&); \
  template Box<T> FromPython<Box<T> >::convert(PyObject*);
INSTANTIATE(float)
INSTANTIATE(double)

#endif

}
