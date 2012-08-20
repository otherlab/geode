//#####################################################################
// Function wrap_method
//#####################################################################
//
// Converts a C++ member function into a python function.  It is normally used indirectly through Class<T>.
//
// In order to convert a function of type R(...,Ai,...), there must be From_Python overloads converting PyObject* to Ai,
// and a to_python function converting R to PyObject*.  See to_python.h and from_python.h for details.
//
// Since python has no notion of constness, there is no difference between the wrapped versions of a method with and
// without const qualification.
//
// note: Method_Inner_Wrapper_* unfortunately can't be declared static because gcc disallows static functions as template
// arguments.  Putting it in an unnamed namespace clutters up the stack traces, so we rely on hidden visibility.
//
//#####################################################################
#pragma once

#include <other/core/python/config.h>
#include <other/core/python/exceptions.h>
#include <other/core/python/from_python.h>
#include <other/core/python/to_python.h>
#include <other/core/python/outer_wrapper.h>
#include <other/core/utility/config.h>
#include <other/core/utility/Enumerate.h>
#include <boost/mpl/bool.hpp>
namespace other {

namespace mpl = boost::mpl;

OTHER_EXPORT PyObject* wrap_method_helper(PyTypeObject* type,const char* name,wrapperfunc wrapper,void* method);

// Can't be static because static functions can't be template arguments
template<class M,class R,class T,class... Args> R
method_inner_wrapper(PyObject* self,PyObject* args,void* method) {
  Py_ssize_t size = PyTuple_GET_SIZE(args);
  const int desired = sizeof...(Args);
  if (size!=desired) throw_arity_mismatch(desired,size);
  return (GetSelf<T>::get(self)->*(*(M*)method))(convert_item<Args>(args)...);
}

// wrap_method for static methods
template<class T,class M,class R,class... Args> static PyObject*
wrap_method(const char* name,R (*method)(Args...)) {
  return wrap_function(name,method);
}

template<class T,class M,class R,class... Args> static wrapperfunc wrapped_method(Types<Args...>) {
  return OuterWrapper<R,PyObject*,PyObject*,void*>::template wrap<method_inner_wrapper<M,R,T,Args...> >;
}

// wrap_method for nonconst methods
template<class T,class M,class R,class B,class... Args> static PyObject*
wrap_method(const char* name,R (B::*method)(Args...)) {
  return wrap_method_helper(&T::pytype,name,wrapped_method<T,M,R>(typename Enumerate<Args...>::type()),(void*)new M(method));
}

// wrap_method for const methods
template<class T,class M,class R,class B,class... Args> static PyObject*
wrap_method(const char* name,R (B::*method)(Args...) const) {
  return wrap_method_helper(&T::pytype,name,wrapped_method<T,M,R>(typename Enumerate<Args...>::type()),(void*)new M(method));
}

}
