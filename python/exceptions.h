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

#include <other/core/python/config.h>
#include <other/core/utility/config.h>
#include <exception>
#include <stdexcept>
#include <typeinfo>
#include <string>
namespace other {

using std::string;

OTHER_EXPORT void set_python_exception(const std::exception& error);
OTHER_EXPORT void register_python_exception(const std::type_info& type,PyObject* pytype);

// Exception throwing functions to reduce code bloat
#ifdef OTHER_PYTHON
OTHER_EXPORT void OTHER_NORETURN(throw_python_error()); // python error must already be set
OTHER_EXPORT void OTHER_NORETURN(throw_type_error(PyObject* object,PyTypeObject* type));
OTHER_EXPORT void OTHER_NORETURN(unregistered_python_type(PyObject* object,PyTypeObject* type));
OTHER_EXPORT void OTHER_NORETURN(throw_arity_mismatch(const int expected,const ssize_t got));
OTHER_EXPORT void OTHER_NORETURN(throw_no_keyword_args(PyObject* kwargs));
#else
OTHER_EXPORT void OTHER_NORETURN(throw_no_python());
#endif

template<class TError> static inline void
register_python_exception(PyObject* pytype) {
  register_python_exception(typeid(TError),pytype);
}

// note: destructors must be in .cpp to avoid shared library name lookup issues

// python error must have already been set
struct OTHER_EXPORT PythonError:public std::exception {
  typedef std::exception Base;
  PythonError();
  virtual ~PythonError() throw ();
  virtual const char* what() const throw();
private:
  mutable string what_;
};

#define OTHER_SIMPLE_EXCEPTION(Error,Base_) \
  struct OTHER_EXPORT Error : public Base_ { \
    typedef Base_ Base; \
    Error(const std::string& message); \
    virtual ~Error() throw (); \
  };

typedef std::runtime_error RuntimeError;

OTHER_SIMPLE_EXCEPTION(IOError,RuntimeError)
OTHER_SIMPLE_EXCEPTION(OSError,RuntimeError)
OTHER_SIMPLE_EXCEPTION(LookupError,RuntimeError)
  OTHER_SIMPLE_EXCEPTION(IndexError,LookupError)
  OTHER_SIMPLE_EXCEPTION(KeyError,LookupError)
OTHER_SIMPLE_EXCEPTION(TypeError,RuntimeError)
OTHER_SIMPLE_EXCEPTION(ValueError,RuntimeError)
OTHER_SIMPLE_EXCEPTION(NotImplementedError,RuntimeError)
OTHER_SIMPLE_EXCEPTION(AssertionError,RuntimeError)
OTHER_SIMPLE_EXCEPTION(ArithmeticError,RuntimeError)
  OTHER_SIMPLE_EXCEPTION(OverflowError,ArithmeticError)
  OTHER_SIMPLE_EXCEPTION(ZeroDivisionError,ArithmeticError)
OTHER_SIMPLE_EXCEPTION(ReferenceError,RuntimeError)

#undef OTHER_SIMPLE_EXCEPTION

}
