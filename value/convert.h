// Conversion from PyObject* to Value<T>
#pragma once

#include <other/core/value/Compute.h>
#include <other/core/python/from_python.h>
#include <other/core/python/Ptr.h>
#include <other/core/utility/format.h>
#include <iostream>
namespace other {

#ifdef OTHER_PYTHON

template<class T> static T convert_helper(const ValueRef<Ptr<>>& value) {
  Ptr<> v = value();
  return from_python<T>(v ? v.get() : Py_None);
}

template<class T> struct FromPython<ValueRef<T>>{static ValueRef<T> convert(PyObject* object) {
  const ValueBase& base = from_python<ValueBase&>(object);
  if (const Value<T>* exact = base.cast<T>())
    return ValueRef<T>(*exact);
  if (const Value<Ptr<>>* python = base.cast<Ptr<>>())
    return cache(convert_helper<T>,ValueRef<Ptr<>>(*python));
  throw TypeError(format("can't convert '%s' to '%s'",typeid(base).name(),typeid(Value<T>).name()));
}};

#endif
}
