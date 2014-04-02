// Conversion from PyObject* to Value<T>
#pragma once

#include <geode/value/Compute.h>
#include <geode/utility/Ptr.h>
#include <geode/utility/format.h>
#include <iostream>
namespace geode {

#if 0 // Value python support

template<class T> static T convert_helper(const ValueRef<Ptr<>>& value) {
  Ptr<> v = value();
  return try_from_python<T>(v ? v.get() : Py_None);
}

template<class T> struct FromPython<Ref<const Value<T>>>{static Ref<const Value<T>> convert(PyObject* object) {
  const ValueBase& base = from_python<ValueBase&>(object);
  if (const Value<T>* exact = base.cast<T>())
    return geode::ref(*exact);
  if (const Value<Ptr<>>* python = base.cast<Ptr<>>())
    return cache(convert_helper<T>,ValueRef<Ptr<>>(*python)).self;
  throw TypeError(format("can't convert '%s' to '%s'",typeid(base).name(),typeid(Value<T>).name()));
}};

template<class T> struct FromPython<Ref<Value<T>>>{static Ref<Value<T>> convert(PyObject* object) {
  return FromPython<Ref<const Value<T>>>::convert(object).const_cast_();
}};

template<class T> struct FromPython<ValueRef<T>>{static ValueRef<T> convert(PyObject* object) {
  return *FromPython<Ref<const Value<T>>>::convert(object);
}};

#endif
}
