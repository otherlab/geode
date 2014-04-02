//#####################################################################
// Header wrap_constructor
//#####################################################################
//
// Converts a C++ constructor to a python __new__ function.  It is normally used indirectly through Class<T>.
//
//#####################################################################
#pragma once

#include <geode/python/config.h>
#include <geode/python/utility.h>
#include <geode/utility/enumerate.h>
namespace geode {

using std::exception;

GEODE_CORE_EXPORT void set_argument_count_error(int desired, PyObject* args, PyObject* kwds);
GEODE_CORE_EXPORT void handle_constructor_error(PyObject* self, const exception& error);

#ifdef GEODE_VARIADIC

// wrapped_constructor

template<class T,class... Args> static PyObject*
wrapped_constructor(PyTypeObject* type, PyObject* args, PyObject* kwds) {
  const int desired = sizeof...(Args);
  const auto nargs = PyTuple_GET_SIZE(args);
  // Require exact match if we're constructing T, otherwise extra arguments are fine
  if (type==&T::pytype ? nargs!=desired || (kwds && PyDict_Size(kwds)) : nargs<desired) {
    set_argument_count_error(desired,args,kwds);
    return 0;
  }
  PyObject* self = type->tp_alloc(type,0);
  if (self) {
    try {
      new(self+1) T(convert_item<Args>(args)...);
    } catch (const exception& error) {
      handle_constructor_error(self,error);
      return 0;
    }
  }
  return self;
}

// wrap_constructor.  The following are inlined into GEODE_INIT to reduce error message size.

template<class T,class... Args> static newfunc wrap_constructor_helper(Types<Args...>) {
  return wrapped_constructor<T,Args...>;
}

template<class T,class... Args> static newfunc wrap_constructor() {
  return wrap_constructor_helper<T>(typename Enumerate<Args...>::type());
}

#else // Unpleasant nonvariadic versions

template<class T,class Args> struct WrapConstructor;

#define GEODE_WRAP_CONSTRUCTOR(n,ARGS,Args) \
  GEODE_WRAP_CONSTRUCTOR_2(n,(,GEODE_REMOVE_PARENS(ARGS)),(,GEODE_REMOVE_PARENS(Args)),Args)

#define GEODE_WRAP_CONSTRUCTOR_2(n,CARGS,CArgs,Args) \
  template<class T GEODE_REMOVE_PARENS(CARGS)> struct WrapConstructor<T,Types<GEODE_REMOVE_PARENS(Args)>> { \
    static PyObject* wrap(PyTypeObject* type, PyObject* args, PyObject* kwds) { \
      const int desired = n; \
      const int nargs = PyTuple_GET_SIZE(args); \
      /* Require exact match if we're constructing T, otherwise extra arguments are fine */ \
      if (type==&T::pytype ? nargs!=desired || (kwds && PyDict_Size(kwds)) : nargs<desired) { \
        set_argument_count_error(desired,args,kwds); \
        return 0; \
      } \
      PyObject* self = type->tp_alloc(type,0); \
      if (self) { \
        try { \
          new(self+1) T(GEODE_CONVERT_ARGS_##n); \
        } catch (const exception& error) { \
          handle_constructor_error(self,error); \
          return 0; \
        } \
      } \
      return self; \
    } \
  };

GEODE_WRAP_CONSTRUCTOR_2(0,(),(),())
GEODE_WRAP_CONSTRUCTOR(1,(class A0),(A0))
GEODE_WRAP_CONSTRUCTOR(2,(class A0,class A1),(A0,A1))
GEODE_WRAP_CONSTRUCTOR(3,(class A0,class A1,class A2),(A0,A1,A2))
GEODE_WRAP_CONSTRUCTOR(4,(class A0,class A1,class A2,class A3),(A0,A1,A2,A3))
GEODE_WRAP_CONSTRUCTOR(5,(class A0,class A1,class A2,class A3,class A4),(A0,A1,A2,A3,A4))
GEODE_WRAP_CONSTRUCTOR(6,(class A0,class A1,class A2,class A3,class A4,class A5),(A0,A1,A2,A3,A4,A5))
GEODE_WRAP_CONSTRUCTOR(7,(class A0,class A1,class A2,class A3,class A4,class A5,class A6),(A0,A1,A2,A3,A4,A5,A6))

#undef GEODE_WRAP_CONSTRUCTOR_2
#undef GEODE_WRAP_CONSTRUCTOR

#endif

}
