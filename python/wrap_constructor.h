//#####################################################################
// Header wrap_constructor
//#####################################################################
//
// Converts a C++ constructor to a python __new__ function.  It is normally used indirectly through Class<T>.
//
//#####################################################################
#pragma once

#include <other/core/python/config.h>
#include <other/core/python/from_python.h>
#include <other/core/python/utility.h>
#include <other/core/utility/Enumerate.h>
namespace other {

namespace mpl = boost::mpl;
using std::exception;

void set_argument_count_error(int desired,PyObject* args,PyObject* kwds) OTHER_EXPORT;
void handle_constructor_error(PyObject* self,const exception& error) OTHER_EXPORT;

#ifdef OTHER_VARIADIC

// wrapped_constructor

template<class T,class... Args> static PyObject*
wrapped_constructor(PyTypeObject* type,PyObject* args,PyObject* kwds) {
  const int desired = sizeof...(Args);
  if (PyTuple_GET_SIZE(args)!=desired || (kwds && PyDict_Size(kwds))) {
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

// wrap_constructor.  The following are inlined into OTHER_INIT to reduce error message size.

template<class T,class... Args> static newfunc wrap_constructor_helper(Types<Args...>) {
  return wrapped_constructor<T,Args...>;
}

template<class T,class... Args> static newfunc wrap_constructor() {
  return wrap_constructor_helper<T>(typename Enumerate<Args...>::type());
}

#else // Unpleasant nonvariadic versions

template<class T,class Args> struct WrapConstructor;

#define OTHER_WRAP_CONSTRUCTOR(n,ARGS,Args) \
  OTHER_WRAP_CONSTRUCTOR_2(n,(,OTHER_REMOVE_PARENS(ARGS)),(,OTHER_REMOVE_PARENS(Args)),Args)

#define OTHER_WRAP_CONSTRUCTOR_2(n,CARGS,CArgs,Args) \
  template<class T OTHER_REMOVE_PARENS(CARGS)> struct WrapConstructor<T,Types<OTHER_REMOVE_PARENS(Args)>> { \
    static PyObject* wrap(PyTypeObject* type,PyObject* args,PyObject* kwds) { \
      const int desired = n; \
      if (PyTuple_GET_SIZE(args)!=desired || (kwds && PyDict_Size(kwds))) { \
        set_argument_count_error(desired,args,kwds); \
        return 0; \
      } \
      PyObject* self = type->tp_alloc(type,0); \
      if (self) { \
        try { \
          new(self+1) T(OTHER_CONVERT_ARGS_##n); \
        } catch (const exception& error) { \
          handle_constructor_error(self,error); \
          return 0; \
        } \
      } \
      return self; \
    } \
  };

OTHER_WRAP_CONSTRUCTOR_2(0,(),(),())
OTHER_WRAP_CONSTRUCTOR(1,(class A0),(A0))
OTHER_WRAP_CONSTRUCTOR(2,(class A0,class A1),(A0,A1))
OTHER_WRAP_CONSTRUCTOR(3,(class A0,class A1,class A2),(A0,A1,A2))
OTHER_WRAP_CONSTRUCTOR(4,(class A0,class A1,class A2,class A3),(A0,A1,A2,A3))
OTHER_WRAP_CONSTRUCTOR(5,(class A0,class A1,class A2,class A3,class A4),(A0,A1,A2,A3,A4))
OTHER_WRAP_CONSTRUCTOR(6,(class A0,class A1,class A2,class A3,class A4,class A5),(A0,A1,A2,A3,A4,A5))

#undef OTHER_WRAP_CONSTRUCTOR_2
#undef OTHER_WRAP_CONSTRUCTOR

#endif

}
