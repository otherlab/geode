//#####################################################################
// Function to_python
//#####################################################################
//
// Conversion from C++ types to PyObject*.
//
// to_python can be extended via normal overloading.  Overloads should take the C++ object, return a PyObject* if possible,
// and otherwise set a python exception and return null.  to_python overloads should never throw exceptions: if this was
// allowed, overloads which call other overloads would require two different sets of error handling code.
//
// to_python does not need to be specialized for classes which are native python objects, since predefined conversions
// from Ref<T> and Ptr<T> are already defined.  The pointer version converts null pointers to None.
//
//#####################################################################
#pragma once

#include <other/core/python/forward.h>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_enum.hpp>
#include <string>
namespace other {

using std::string;

// Conversion for PyObject*
static inline PyObject*
to_python(PyObject* value) {
  Py_INCREF(value);
  return value;
}

// Conversion from bool
static inline PyObject*
to_python(bool value) {
  return PyBool_FromLong(value);
}

// Conversion from integers

static inline PyObject*
to_python(int value) {
  return PyInt_FromLong(value);
}

static inline PyObject*
to_python(unsigned int value) {
  return PyInt_FromSize_t(value);
}

static inline PyObject*
to_python(long value) {
  return PyInt_FromLong(value);
}

static inline PyObject*
to_python(unsigned long value) {
  return PyLong_FromUnsignedLongLong(value);
}

static inline PyObject*
to_python(unsigned long long value) {
  return PyLong_FromUnsignedLongLong(value);
}

// Conversion from float/double

static inline PyObject*
to_python(float value) {
  return PyFloat_FromDouble(value);
}

static inline PyObject*
to_python(double value) {
  return PyFloat_FromDouble(value);
}

// Conversion from string/char*

static inline PyObject*
to_python(const char* value) {
  return PyString_FromString(value);
}

static inline PyObject*
to_python(const string& value) {
  return PyString_FromStringAndSize(value.c_str(),(Py_ssize_t)value.size());
}

// Conversion from char
static inline PyObject*
to_python(char value) {
  char s[2] = {value,0};
  return PyString_FromString(s);
}

// Conversion to enums (see enum.h for the definition)
template<class E> typename boost::enable_if<boost::is_enum<E>,PyObject*>::type to_python(E value) OTHER_EXPORT;

} // namespace other
