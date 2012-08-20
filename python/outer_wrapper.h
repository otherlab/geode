//#####################################################################
// Function outer_wrapper
//#####################################################################
//
// Convert a function that takes Python arguments and returns a C++ type into
// a function that takes Python arguments and returns a Python type.  This is
// one half of the wrapping needed to expose a C++ function to Python.
//
//#####################################################################
#pragma once

#include <other/core/python/config.h>
#include <other/core/python/exceptions.h>
#include <other/core/python/utility.h>
#include <other/core/utility/config.h>
#include <boost/type_traits/is_same.hpp>
#include <boost/utility/enable_if.hpp>
namespace other {

using std::exception;

template<class R,class... Args> struct OuterWrapper {
  template<R inner(Args...)> static PyObject* wrap(Args... args) {
    try {
      return to_python(inner(args...));
    } catch (const exception& error) {
      set_python_exception(error);
      return 0;
    }
  }
};

template<class... Args> struct OuterWrapper<void,Args...> {
  template<void inner(Args...)> static PyObject* wrap(Args... args) {
    try {
      inner(args...);
      Py_RETURN_NONE;
    } catch (const exception& error) {
      set_python_exception(error);
      return 0;
    }
  }
};

}
