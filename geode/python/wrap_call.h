//#####################################################################
// Function wrap_call
//#####################################################################
//
// Wrap operator() as a python __call__ method
//
//#####################################################################
#pragma once

#include <geode/python/config.h>
#include <geode/python/exceptions.h>
#include <geode/python/from_python.h>
#include <geode/python/to_python.h>
#include <geode/python/outer_wrapper.h>
#include <geode/utility/config.h>
#include <boost/mpl/bool.hpp>
namespace geode {

namespace mpl = boost::mpl;

#ifdef GEODE_VARIADIC

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

template<class T,class... Args> static inline ternaryfunc
wrap_call() {
  typedef decltype((*(T*)0)(declval<Args>()...)) R;
  return wrap_call_helper<R,T>(typename Enumerate<Args...>::type());
}

#else // Unpleasant nonvariadic versions

template<class T,class A0=void,class A1=void,class A2=void,class A3=void,class A4=void,class A5=void> struct WrapCall;

#define GEODE_WRAP_CALL(n,ARGS,Args) \
  GEODE_WRAP_CALL_2(n,(,GEODE_REMOVE_PARENS(ARGS)),(,GEODE_REMOVE_PARENS(Args)))

#define GEODE_WRAP_CALL_2(n,CARGS,CArgs,values) \
  template<class T GEODE_REMOVE_PARENS(CARGS)> struct WrapCall<T GEODE_REMOVE_PARENS(CArgs)> { \
    typedef decltype((*(T*)0) values) R; \
    /* Inner wrapper: convert arguments and call method */ \
    static R inner_wrapper(PyObject* self,PyObject* args,PyObject* kwargs) { \
      Py_ssize_t size = PyTuple_GET_SIZE(args); \
      const int desired = n; \
      if (size!=desired) throw_arity_mismatch(desired,size); \
      if (kwargs && PyDict_Size(kwargs)) throw_no_keyword_args(kwargs); \
      return (*((T*)(self+1)))(GEODE_CONVERT_ARGS_##n); \
    } \
    \
    static inline ternaryfunc wrap() { \
      return OuterWrapper<R,PyObject*,PyObject*,PyObject*>::template wrap<&WrapCall::inner_wrapper>; \
    } \
  };

#define V(i) declval<A##i>()
GEODE_WRAP_CALL_2(0,(),(),())
GEODE_WRAP_CALL(1,(class A0),(A0),(V(0)))
GEODE_WRAP_CALL(2,(class A0,class A1),(A0,A1),(V(0),V(1)))
GEODE_WRAP_CALL(3,(class A0,class A1,class A2),(A0,A1,A2),(V(0),V(1),V(2)))
GEODE_WRAP_CALL(4,(class A0,class A1,class A2,class A3),(A0,A1,A2,A3),(V(0),V(1),V(2),V(3)))
GEODE_WRAP_CALL(5,(class A0,class A1,class A2,class A3,class A4),(A0,A1,A2,A3,A4),(V(0),V(1),V(2),V(3),V(4)))
#undef V

#undef GEODE_WRAP_CALL_2
#undef GEODE_WRAP_CALL

#endif

}
