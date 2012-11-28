//#####################################################################
// Conversion for functions
//#####################################################################
#pragma once

#include <other/core/python/from_python.h>
#include <other/core/python/to_python.h>
#include <other/core/utility/config.h>
#include <boost/function.hpp>
#include <boost/type_traits/function_traits.hpp>
#include <other/core/python/utility.h>
namespace other {

using std::string;
using boost::function;

namespace {

template<class R> struct PythonFunctionWrapper {
  Ref<PyObject> f;

  PythonFunctionWrapper(PyObject* object)
    : f(from_python<Ref<PyObject> >(object))
  {}

  R return_(PyObject* r) {
    return from_python<R>(&*steal_ref(*r));
  }

#ifdef OTHER_VARIADIC

  template<class... Args> R operator()(Args&&... args) {
    PyObject* r = PyObject_CallFunctionObjArgs(&*f,&*to_python_check(args)...,0);
    if (!r) throw_python_error();
    return return_(r);
  }

#else // Unpleasant nonvariadic versions

  #define OTHER_FUNCTION_CALL(ARGS,Args,convert) \
    OTHER_FUNCTION_CALL_2((template<OTHER_REMOVE_PARENS(ARGS)>),Args,(OTHER_REMOVE_PARENS(convert),))

  #define OTHER_FUNCTION_CALL_2(TARGS,Args,convertC) \
    OTHER_REMOVE_PARENS(TARGS) R operator() Args { \
      PyObject* r = PyObject_CallFunctionObjArgs(&*f,OTHER_REMOVE_PARENS(convertC) 0); \
      if (!r) throw_python_error(); \
      return return_(r); \
    }

  OTHER_FUNCTION_CALL_2((),(),())
  OTHER_FUNCTION_CALL((class A0),(A0&& a0),(&*to_python_check(a0)))
  OTHER_FUNCTION_CALL((class A0,class A1),(A0&& a0,A1&& a1),(&*to_python_check(a0),&*to_python_check(a1)))
  OTHER_FUNCTION_CALL((class A0,class A1,class A2),(A0&& a0,A1&& a1,A2&& a2),(&*to_python_check(a0),&*to_python_check(a1),&*to_python_check(a2)))
  #undef OTHER_FUNCTION_CALL_2
  #undef OTHER_FUNCTION_CALL

#endif
};

template<> void PythonFunctionWrapper<void>::
return_(PyObject* r) {
  Py_DECREF(r);
}

}

OTHER_EXPORT void OTHER_NORETURN(throw_callable_error(PyObject* object));

template<class F> struct FromPython<function<F> >{static function<F> convert(PyObject* object) {
  if (object==Py_None)
    return function<F>();
  if (!PyCallable_Check(object))
    throw_callable_error(object);
  return PythonFunctionWrapper<typename boost::function_traits<F>::result_type>(object);
}};

template<class F> PyObject* to_python(const boost::function<F>& f) {
  OTHER_NOT_IMPLEMENTED();
}

}
