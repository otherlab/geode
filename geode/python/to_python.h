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

#include <geode/python/forward.h>
#include <geode/utility/debug.h>
#include <geode/utility/forward.h>
#include <geode/utility/validity.h>
#include <boost/utility/declval.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_enum.hpp>
#include <string>
namespace geode {

using std::string;
namespace mpl = boost::mpl;

#ifdef GEODE_PYTHON

// Conversion for PyObject*
static inline PyObject* to_python(PyObject* value) {
  GEODE_XINCREF(value); // Allow zero so that wrapped functions can return (PyObject*)0 on error
  return value;
}

// Conversion from bool, taking care not to accept pointer arguments and other weird types
template<class T> static inline typename boost::enable_if<boost::is_same<T,bool>,PyObject*>::type to_python(T value) {
  return PyBool_FromLong(value);
}

// Conversion from integers

static inline PyObject* to_python(int value) {
  return PyInt_FromLong(value);
}

static inline PyObject* to_python(unsigned int value) {
  return PyInt_FromSize_t(value);
}

static inline PyObject* to_python(long value) {
  return PyInt_FromLong(value);
}

static inline PyObject* to_python(unsigned long value) {
  return PyLong_FromUnsignedLongLong(value);
}

static inline PyObject* to_python(long long value) {
  return PyLong_FromLongLong(value);
}

static inline PyObject* to_python(unsigned long long value) {
  return PyLong_FromUnsignedLongLong(value);
}

// Conversion from float/double

static inline PyObject* to_python(float value) {
  return PyFloat_FromDouble(value);
}

static inline PyObject* to_python(double value) {
  return PyFloat_FromDouble(value);
}

// Conversion from string/char*

static inline PyObject* to_python(const char* value) {
  return PyString_FromString(value);
}

static inline PyObject* to_python(const string& value) {
  return PyString_FromStringAndSize(value.c_str(),(Py_ssize_t)value.size());
}

// uint8_t is a valuable small integer type to use, and we believe most
// machines and compilers nowadays have signed chars.  Therefore, we are
// going to do something horrible: make char convert from string, and
// uint8_t convert from small integers.
static_assert(!boost::is_same<char,uint8_t>::value, "Different conversions for uint8_t and char (even though they're the same)! Our fault!");

// Conversion from char
static inline PyObject* to_python(char value) {
  char s[2] = {value,0};
  return PyString_FromString(s);
}

// Conversion from uint8_t
static inline PyObject* to_python(uint8_t value) {
  return to_python((int)value);
}

// Conversion for sets
template<class TS> PyObject* to_python_set(const TS& s) {
  PyObject* set = PySet_New(0);
  if (!set) goto fail;
  for (auto it=s.begin(),end=s.end();it!=end;++it) { // Avoid foreach since pcl needs gcc 4.4
    PyObject* o = to_python(*it);
    if (!o) goto fail;
    int r = PySet_Add(set,o);
    Py_DECREF(o);
    if (r<0) goto fail;
  }
  return set;
  fail:
  Py_XDECREF(set);
  return 0;
}

// Declare has_to_python<T>
GEODE_VALIDITY_CHECKER(has_to_python,T,to_python(*(T*)0))
template<> struct has_to_python<void> : public mpl::true_ {};

#endif

} // namespace geode
