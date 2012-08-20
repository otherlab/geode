//#####################################################################
// Various python interface utilities
//#####################################################################
#pragma once

#include <other/core/python/config.h>
#include <other/core/python/Object.h>
#include <other/core/python/to_python.h>
#include <other/core/python/from_python.h>
namespace other {

template<class T> inline Ref<> to_python_check(const T& o) {
  return steal_ref_check(to_python(o));
}

template<class T> inline Ref<> python_field(const T& o, const char* name) {
  return steal_ref_check(PyObject_GetAttrString(&*to_python_check(o),name));
}

// Given an ITP<n,T> pair, extract the given index and convert to the desired type
template<class A> static inline auto convert_item(PyObject* args)
  -> decltype(from_python<typename A::type>((PyObject*)0)) {
  return from_python<typename A::type>(PyTuple_GET_ITEM(args,A::index));
}

// Call a python function
template<class T,class... Args> inline Ref<> python_call(const T& f, Args&&... args) {
  return steal_ref_check(PyObject_CallFunctionObjArgs(&*to_python_check(f),&*to_python_check(args)...,0));
}

// Call a python method
template<class T,class... Args> inline Ref<> python_call_method(const T& o, const char* name, Args&&... args) {
  return python_call(python_field(o,name),args...);
}

}
