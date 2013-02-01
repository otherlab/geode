// Convert to and from python if possible, otherwise throw an exception

#include <other/core/python/try_convert.h>
#include <other/core/utility/format.h>
namespace other {

#ifdef OTHER_PYTHON

void set_to_python_failed(const type_info& type) {
  PyErr_Format(PyExc_TypeError,"C++ type %s can't be converted to Python: no to_python overload exists",type.name());
}

void from_python_failed(PyObject* object, const type_info& type) {
  throw TypeError(format("Python type %s can't be converted to C++ type %s: no from_python overload exists",object->ob_type->tp_name,type.name()));
}

#endif

}
