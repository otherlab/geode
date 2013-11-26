//#####################################################################
// Conversion for functions
//#####################################################################
#pragma once

#include <geode/python/from_python.h>
#include <geode/python/to_python.h>
#include <geode/utility/config.h>
#include <boost/function.hpp>
#include <boost/type_traits/function_traits.hpp>
#include <geode/python/utility.h>
namespace geode {

using std::string;

#ifdef GEODE_PYTHON

namespace {

template<class R> struct PythonFunctionWrapper {
  typedef R result_type;

  Ref<PyObject> f;

  PythonFunctionWrapper(PyObject* object)
    : f(from_python<Ref<PyObject> >(object))
  {}

  R return_(PyObject* r) {
    return from_python<R>(&*steal_ref(*r));
  }

#ifdef GEODE_VARIADIC

  template<class... Args> R operator()(Args&&... args) {
    PyObject* r = PyObject_CallFunctionObjArgs(&*f,&*to_python_check(args)...,0);
    if (!r) throw_python_error();
    return return_(r);
  }

#else // Unpleasant nonvariadic versions

  #define GEODE_FUNCTION_CALL(ARGS,Args,convert) \
    GEODE_FUNCTION_CALL_2((template<GEODE_REMOVE_PARENS(ARGS)>),Args,(GEODE_REMOVE_PARENS(convert),))

  #define GEODE_FUNCTION_CALL_2(TARGS,Args,convertC) \
    GEODE_REMOVE_PARENS(TARGS) R operator() Args { \
      PyObject* r = PyObject_CallFunctionObjArgs(&*f,GEODE_REMOVE_PARENS(convertC) 0); \
      if (!r) throw_python_error(); \
      return return_(r); \
    }

  GEODE_FUNCTION_CALL_2((),(),())
  GEODE_FUNCTION_CALL((class A0),(A0&& a0),(&*to_python_check(a0)))
  GEODE_FUNCTION_CALL((class A0,class A1),(A0&& a0,A1&& a1),(&*to_python_check(a0),&*to_python_check(a1)))
  GEODE_FUNCTION_CALL((class A0,class A1,class A2),(A0&& a0,A1&& a1,A2&& a2),(&*to_python_check(a0),&*to_python_check(a1),&*to_python_check(a2)))
  GEODE_FUNCTION_CALL((class A0,class A1,class A2,class A3),(A0&& a0,A1&& a1,A2&& a2,A3&& a3),(&*to_python_check(a0),&*to_python_check(a1),&*to_python_check(a2),&*to_python_check(a3)))
  #undef GEODE_FUNCTION_CALL_2
  #undef GEODE_FUNCTION_CALL

#endif
};

template<> void PythonFunctionWrapper<void>::return_(PyObject* r) {
  Py_DECREF(r);
}

}

GEODE_CORE_EXPORT void GEODE_NORETURN(throw_callable_error(PyObject* object));

template<class F> struct FromPython<boost::function<F> >{static boost::function<F> convert(PyObject* object) {
  if (object==Py_None)
    return boost::function<F>();
  if (!PyCallable_Check(object))
    throw_callable_error(object);
  return PythonFunctionWrapper<typename boost::function_traits<F>::result_type>(object);
}};

template<class F> PyObject* to_python(const boost::function<F>& f) {
  GEODE_NOT_IMPLEMENTED();
}

#endif
}
