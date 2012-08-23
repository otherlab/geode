//#####################################################################
// Function wrap_function
//#####################################################################
//
// Converts a free C++ function into a python function.  It is normally used indirectly through function() (see Module.h).
//
// In order to convert a function of type R(...,Ai,...), there must be from_python overloads converting PyObject* to Ai,
// and a to_python function converting R to PyObject*.  See to_python.h and from_python.h for details.
//
// note: function_inner_wrapper unfortunately can't be declared static because gcc disallows static functions as template
// arguments.  Putting it in an unnamed namespace clutters up the stack traces, so we rely on hidden visibility.
//
//#####################################################################
#pragma once

#include <other/core/python/config.h>
#include <other/core/python/exceptions.h>
#include <other/core/python/outer_wrapper.h>
#include <other/core/python/utility.h>
#include <other/core/utility/config.h>
#include <other/core/utility/Enumerate.h>
#include <boost/type_traits/is_same.hpp>
#include <boost/utility/enable_if.hpp>
namespace other {

typedef PyObject* (*FunctionWrapper)(PyObject* args,void* wrapped);
OTHER_EXPORT PyObject* wrap_function_helper(const char* name,FunctionWrapper wrapper,void* function);

// Can't be static because static functions can't be template arguments
template<class F,class R,class... Args> inline R
function_inner_wrapper(PyObject* args,void* wrapped) {
  Py_ssize_t size = PyTuple_GET_SIZE(args);
  const int desired = sizeof...(Args);
  if (size!=desired) throw_arity_mismatch(desired,size);
  return ((F)wrapped)(convert_item<Args>(args)...);
}

template<class F,class R,class... Args> static FunctionWrapper wrapped_function(Types<Args...>) {
  return OuterWrapper<R,PyObject*,void*>::template wrap<function_inner_wrapper<F,R,Args...> >;
}

template<class R,class... Args> static PyObject*
wrap_function(const char* name,R (*function)(Args...)) {
  return wrap_function_helper(name,wrapped_function<decltype(function),R>(typename Enumerate<Args...>::type()),(void*)function);
}

}
