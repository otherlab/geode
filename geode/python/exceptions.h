//#####################################################################
// File Exceptions
//#####################################################################
//
// C++ equivalents of python exceptions, and translation between C++ and python.
//
// Exception translate is done through exact typeids, so exceptions derived from these must be separately registered
// with the translation mechanism.  Unregistered exceptions deriving from std::exception will be converted into Exception
// in python.  If C++ code throws an object not deriving from std::exception, it will propagate into the python runtime,
// and then...explode.
//
// To set a python exception directly from C++ code, use PyErr_SetString or the equivalent and then throw PythonError().
//
//#####################################################################
#pragma once

#include <geode/python/config.h>
#include <geode/utility/config.h>
#include <exception>
#include <stdexcept>
#include <typeinfo>
#include <string>
namespace geode {

using std::string;
using std::type_info;
using std::exception;

GEODE_CORE_EXPORT void set_python_exception(const exception& error);
GEODE_CORE_EXPORT void register_python_exception(const type_info& type, PyObject* pytype);
GEODE_CORE_EXPORT void print_and_clear_exception(const string& where, const exception& error);

// Exception throwing functions to reduce code bloat
#ifdef GEODE_PYTHON
GEODE_CORE_EXPORT void GEODE_NORETURN(throw_python_error()); // python error must already be set
GEODE_CORE_EXPORT void GEODE_NORETURN(throw_type_error(PyObject* object, PyTypeObject* type));
GEODE_CORE_EXPORT void GEODE_NORETURN(unregistered_python_type(PyObject* object, PyTypeObject* type, const char* function));
GEODE_CORE_EXPORT void GEODE_NORETURN(throw_arity_mismatch(const int expected, const ssize_t got));
GEODE_CORE_EXPORT void GEODE_NORETURN(throw_no_keyword_args(PyObject* kwargs));
#else
GEODE_CORE_EXPORT void GEODE_NORETURN(throw_no_python());
#endif

template<class TError> static inline void
register_python_exception(PyObject* pytype) {
  register_python_exception(typeid(TError),pytype);
}

// Note: destructors must be in .cpp to avoid shared library name lookup issues

// Python error must have already been set
struct GEODE_CORE_CLASS_EXPORT PythonError:public exception {
  typedef exception Base;
  PythonError();
  virtual ~PythonError() throw ();
  virtual const char* what() const throw();
private:
  mutable string what_;
};

#define GEODE_SIMPLE_EXCEPTION(Error,Base_) \
  struct GEODE_CORE_CLASS_EXPORT Error : public Base_ { \
    typedef Base_ Base; \
    GEODE_CORE_EXPORT Error(const string& message); \
    GEODE_CORE_EXPORT virtual ~Error() throw (); \
  };

typedef std::runtime_error RuntimeError;

GEODE_SIMPLE_EXCEPTION(IOError,RuntimeError)
GEODE_SIMPLE_EXCEPTION(OSError,RuntimeError)
GEODE_SIMPLE_EXCEPTION(LookupError,RuntimeError)
  GEODE_SIMPLE_EXCEPTION(IndexError,LookupError)
  GEODE_SIMPLE_EXCEPTION(KeyError,LookupError)
GEODE_SIMPLE_EXCEPTION(TypeError,RuntimeError)
GEODE_SIMPLE_EXCEPTION(ValueError,RuntimeError)
GEODE_SIMPLE_EXCEPTION(NotImplementedError,RuntimeError)
GEODE_SIMPLE_EXCEPTION(AssertionError,RuntimeError)
GEODE_SIMPLE_EXCEPTION(AttributeError,RuntimeError)
GEODE_SIMPLE_EXCEPTION(ArithmeticError,RuntimeError)
  GEODE_SIMPLE_EXCEPTION(OverflowError,ArithmeticError)
  GEODE_SIMPLE_EXCEPTION(ZeroDivisionError,ArithmeticError)
GEODE_SIMPLE_EXCEPTION(ReferenceError,RuntimeError)
GEODE_SIMPLE_EXCEPTION(ImportError,RuntimeError)

#undef GEODE_SIMPLE_EXCEPTION

}
