//#####################################################################
// Function wrap_call
//#####################################################################
//
// Wrap operator() as a python __call__ method
//
//#####################################################################
#pragma once

#include <other/core/python/config.h>
#include <other/core/python/exceptions.h>
#include <other/core/python/from_python.h>
#include <other/core/python/to_python.h>
#include <other/core/python/outer_wrapper.h>
#include <other/core/utility/config.h>
#include <boost/mpl/bool.hpp>
namespace other{

namespace mpl=boost::mpl;

// Inner wrapper: convert arguments and call method
template<class R,class T,class... Args> R
call_inner_wrapper(PyObject* self,PyObject* args,PyObject* kwargs) {
  Py_ssize_t size = PyTuple_GET_SIZE(args);
  const int desired = sizeof...(Args);
  if (size!=desired) throw_arity_mismatch(desired,size);
  if (kwargs && PyDict_Size(kwargs)) throw_no_keyword_args(kwargs);
  return (*((T*)(self+1)))(convert_item<Args>(args)...);
}

template<class R,class T,class... Args> static inline ternaryfunc wrap_call_helper(Types<Args...>) {
  return OuterWrapper<R,PyObject*,PyObject*,PyObject*>::template wrap<call_inner_wrapper<R,T,Args...> >;
}

template<class T,class R,class... Args> static inline ternaryfunc
wrap_call() {
  return wrap_call_helper<R,T>(typename Enumerate<Args...>::type());
}

}
