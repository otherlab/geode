//#####################################################################
// Header wrap_constructor
//#####################################################################
//
// Converts a C++ constructor to a python __new__ function.  It is normally used indirectly through Class<T>.
//
//#####################################################################
#pragma once

#include <geode/python/config.h>
#include <geode/python/from_python.h>
#include <geode/python/utility.h>
#include <geode/utility/enumerate.h>
namespace geode {

GEODE_VALIDITY_CHECKER(has_iter,A,&A::iter);
GEODE_VALIDITY_CHECKER(has_iternext,A,&A::iternext);

// wrapped iterator constructor function (uses PyObject *T::iter())
template<bool use_iter> struct wrapped_iter_helper {};

template<> struct wrapped_iter_helper<true> {
  template<class T> GEODE_CORE_EXPORT static PyObject *iter(PyObject*o) {
    return GetSelf<T>::get(o)->iter();
  }
};

template<> struct wrapped_iter_helper<false> {
  template<class T> GEODE_CORE_EXPORT static PyObject *iter(PyObject*o) {
    GEODE_INCREF(o);
    return o;
  }
};

template<class T>
GEODE_CORE_EXPORT PyObject* wrapped_iter(PyObject* o) {
  return wrapped_iter_helper<has_iter<T>::value>::template iter<T>(o);
}

// wrapped iterator increment function (uses PyObject *T::iternext())

template<class T, bool use_iternext = has_iternext<T>::value > struct wrapped_iternext_helper {};

template<class T> struct wrapped_iternext_helper<T, true> {
  GEODE_CORE_EXPORT static PyObject *iter(PyObject*o) {
    return GetSelf<T>::get(o)->iternext();
  }
};

template<class T> struct wrapped_iternext_helper<T, false> {
  constexpr static PyObject* (* const iter)(PyObject *o) = 0;
};

}
