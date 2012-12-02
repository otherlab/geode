//#####################################################################
// Conversion for functions
//#####################################################################
#include <other/core/python/function.h>
namespace other {

#ifdef OTHER_PYTHON

void throw_callable_error(PyObject* object) {
  PyErr_Format(PyExc_TypeError,"expected callable object, got %s",object->ob_type->tp_name);
  throw PythonError();
}

#endif
}
