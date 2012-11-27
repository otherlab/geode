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
namespace other {

namespace mpl=boost::mpl;

#ifdef OTHER_VARIADIC

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

#else // Unpleasant nonvariadic versions

template<class T,class R,class A0=void,class A1=void,class A2=void,class A3=void,class A4=void,class A5=void> struct WrapCall;

#define OTHER_WRAP_CALL(n,ARGS,Args) \
  OTHER_WRAP_CALL_2(n,(,OTHER_REMOVE_PARENS(ARGS)),(,OTHER_REMOVE_PARENS(Args)))

#define OTHER_WRAP_CALL_2(n,CARGS,CArgs) \
  template<class T,class R OTHER_REMOVE_PARENS(CARGS)> struct WrapCall<T,R OTHER_REMOVE_PARENS(CArgs)> { \
    /* Inner wrapper: convert arguments and call method */ \
    static R inner_wrapper(PyObject* self,PyObject* args,PyObject* kwargs) { \
      Py_ssize_t size = PyTuple_GET_SIZE(args); \
      const int desired = n; \
      if (size!=desired) throw_arity_mismatch(desired,size); \
      if (kwargs && PyDict_Size(kwargs)) throw_no_keyword_args(kwargs); \
      return (*((T*)(self+1)))(OTHER_CONVERT_ARGS_##n); \
    } \
    \
    static inline ternaryfunc wrap() { \
      return OuterWrapper<R,PyObject*,PyObject*,PyObject*>::template wrap<&WrapCall::inner_wrapper>; \
    } \
  };

OTHER_WRAP_CALL_2(0,(),())
OTHER_WRAP_CALL(1,(class A0),(A0))
OTHER_WRAP_CALL(2,(class A0,class A1),(A0,A1))
OTHER_WRAP_CALL(3,(class A0,class A1,class A2),(A0,A1,A2))
OTHER_WRAP_CALL(4,(class A0,class A1,class A2,class A3),(A0,A1,A2,A3))
OTHER_WRAP_CALL(5,(class A0,class A1,class A2,class A3,class A4),(A0,A1,A2,A3,A4))

#undef OTHER_WRAP_CALL_2
#undef OTHER_WRAP_CALL

#endif

}
