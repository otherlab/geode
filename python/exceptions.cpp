//#####################################################################
// File Exceptions
//#####################################################################
#include <other/core/python/exceptions.h>
#include <other/core/python/module.h>
#include <other/core/utility/tr1.h>
namespace other{

// map from C++ to python exceptions
typedef unordered_map<const std::type_info*,PyObject*> ExceptionMap;
ExceptionMap exception_map;

void set_python_exception(const std::exception& error) {
  if (typeid(error)!=typeid(PythonError)) {
    ExceptionMap::const_iterator exc=exception_map.find(&typeid(error));
    PyObject* type=exc!=exception_map.end()?exc->second:PyExc_RuntimeError;
    PyErr_SetString(type,error.what());
  }
}

void register_python_exception(const std::type_info& type,PyObject* pytype) {
  exception_map[&type] = pytype;
}

static void redefine_assertion_error(PyObject* exc) {
  Py_INCREF(exc);
  register_python_exception<AssertionError>(exc);
}

void throw_python_error() { // python error must already be set
  throw PythonError();
}

void throw_type_error(PyObject* object, PyTypeObject* type) {
  PyErr_Format(PyExc_TypeError,"%s expected (got %s)",type->tp_name,object->ob_type->tp_name);
  throw PythonError();
}

void unregistered_python_type(PyObject* object, PyTypeObject* type) {
  PyErr_Format(PyExc_TypeError,"can't convert %s to unregistered type deriving from %s",object->ob_type->tp_name,type->tp_name);
  throw PythonError();
}

void throw_arity_mismatch(const int expected, const Py_ssize_t got) {
  if(expected)
    PyErr_Format(PyExc_TypeError, "function takes %d argument%s (%ld given)",expected,(expected==1?"":"s"),got);
  else
    PyErr_Format(PyExc_TypeError, "function takes no arguments (%ld given)",got);
  throw PythonError();
}

void throw_no_keyword_args(PyObject* kwargs) {
  PyErr_Format(PyExc_TypeError, "function takes no keyword arguments (%ld given)",PyDict_Size(kwargs));
  throw PythonError();
}

PythonError::PythonError() {
  if (!PyErr_Occurred())
    throw AssertionError("PythonError thrown without setting a python exception");
}

PythonError::~PythonError() throw ()
{}

const char* PythonError::what() const throw() {
  if (!what_.size()) {
    PyObject *type,*value,*traceback;
    PyErr_Fetch(&type,&value,&traceback);
    if (type && value) {
      what_ = ((PyTypeObject*)type)->tp_name;
      what_ += ": ";
      if (PyObject* str = PyObject_Str(value))
        if (const char* s = PyString_AsString(str))
          what_ += s;
        else
          what_ += "<__str__ didn't return a string>";
      else
        what_ += "<__str__ didn't work>";
    } else
      what_ = "<no python exception set>";
    PyErr_Restore(type,value,traceback);
  }
  return what_.c_str();
}

#define INSTANTIATE(Error) \
  Error::Error(const std::string& message):Base(message){} \
  Error::~Error() throw () {}
INSTANTIATE(IOError)
INSTANTIATE(OSError)
INSTANTIATE(LookupError)
INSTANTIATE(IndexError)
INSTANTIATE(KeyError)
INSTANTIATE(TypeError)
INSTANTIATE(ValueError)
INSTANTIATE(NotImplementedError)
INSTANTIATE(AssertionError)
INSTANTIATE(ArithmeticError)
INSTANTIATE(OverflowError)
INSTANTIATE(ZeroDivisionError)
INSTANTIATE(ReferenceError)

}
using namespace other;

void wrap_exceptions() {
  register_python_exception<IOError>(PyExc_IOError);
  register_python_exception<OSError>(PyExc_OSError);
  register_python_exception<LookupError>(PyExc_LookupError);
  register_python_exception<IndexError>(PyExc_IndexError);
  register_python_exception<KeyError>(PyExc_KeyError);
  register_python_exception<TypeError>(PyExc_TypeError);
  register_python_exception<ValueError>(PyExc_ValueError);
  register_python_exception<NotImplementedError>(PyExc_NotImplementedError);
  register_python_exception<AssertionError>(PyExc_AssertionError);
  register_python_exception<ArithmeticError>(PyExc_ArithmeticError);
  register_python_exception<OverflowError>(PyExc_OverflowError);
  register_python_exception<ZeroDivisionError>(PyExc_ZeroDivisionError);
  register_python_exception<ReferenceError>(PyExc_ReferenceError);

  OTHER_FUNCTION(redefine_assertion_error)
}
