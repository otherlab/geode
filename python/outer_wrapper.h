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

#ifdef OTHER_VARIADIC

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

#else // Unpleasant nonvariadic versions

template<class R,class A0=void,class A1=void,class A2=void,class A3=void> struct OuterWrapper;

#define OTHER_OUTER_WRAPPER(ARGS,Args,Argsargs,args) \
  OTHER_OUTER_WRAPPER_2((,OTHER_REMOVE_PARENS(ARGS)),ARGS,(,OTHER_REMOVE_PARENS(Args)),Args,Argsargs,args)

#define OTHER_OUTER_WRAPPER_2(CARGS,ARGS,CArgs,Args,Argsargs,args) \
  template<class R OTHER_REMOVE_PARENS(CARGS)> struct OuterWrapper<R OTHER_REMOVE_PARENS(CArgs)> { \
    template<R inner Args> static PyObject* wrap Argsargs { \
      try { \
        return to_python(inner args); \
      } catch (const exception& error) { \
        set_python_exception(error); \
        return 0; \
      } \
    } \
  }; \
  \
  template<OTHER_REMOVE_PARENS(ARGS)> struct OuterWrapper<void OTHER_REMOVE_PARENS(CArgs)> { \
    template<void inner Args> static PyObject* wrap Argsargs { \
      try { \
        inner args; \
        Py_RETURN_NONE; \
      } catch (const exception& error) { \
        set_python_exception(error); \
        return 0; \
      } \
    } \
  };

OTHER_OUTER_WRAPPER_2((),(),(),(),(),())
OTHER_OUTER_WRAPPER((class A0),(A0),(A0 a0),(a0))
OTHER_OUTER_WRAPPER((class A0,class A1),(A0,A1),(A0 a0,A1 a1),(a0,a1))
OTHER_OUTER_WRAPPER((class A0,class A1,class A2),(A0,A1,A2),(A0 a0,A1 a1,A2 a2),(a0,a1,a2))

#undef OTHER_OUTER_WRAPPER_2
#undef OTHER_OUTER_WRAPPER

#endif

}
