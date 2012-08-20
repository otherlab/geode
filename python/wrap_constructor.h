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

}
